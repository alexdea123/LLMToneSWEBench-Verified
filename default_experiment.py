import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from groq import Groq

from datasets import load_dataset, load_from_disk  # HuggingFace datasets
from tqdm.auto import tqdm

from swebench.inference.make_datasets import create_text_dataset
from swebench.inference.make_datasets.utils import extract_diff, repair_patch

# -----------------------------------------------------------------------------
# Groq helper
# -----------------------------------------------------------------------------

# Instantiate a Groq client using your API key from the environment.
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_processed_dataset_path(
    output_dir: Path,
    dataset_name: str,
    prompt_style: str,
    file_source: str,
) -> Path:
    """Construct the path for the cached processed dataset."""
    if dataset_name.startswith("princeton-nlp"):
        dataset_name = dataset_name.split("/")[-1]
    dataset_name = dataset_name.replace("/", "__")
    output_filename = f"{dataset_name}__{prompt_style}__fs-{file_source}"
    return Path(output_dir) / "datasets" / output_filename


def prepare_dataset(
    dataset_name: str,
    split: str,
    output_dir: Path,
    prompt_style: str,
    file_source: str,
) -> Path:
    """
    Prepare the dataset by running create_text_dataset.py if necessary.
    Returns the path to the processed dataset.
    """
    processed_dataset_path = get_processed_dataset_path(
        output_dir, dataset_name, prompt_style, file_source
    )

    if processed_dataset_path.exists():
        try:
            # Verify that the split exists in the cached dataset
            d = load_from_disk(str(processed_dataset_path))
            if split in d:
                print(f"Using cached processed dataset from {processed_dataset_path}")
                return processed_dataset_path
            print(f"Split {split} not found in cached dataset. Re-creating.")
        except Exception:
            print("Failed to load cached dataset. Re-creating.")

    print(f"Processed dataset not found. Creating at {processed_dataset_path}...")

    processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    create_text_dataset.main(
        dataset_name_or_path=dataset_name,
        splits=[split],
        output_dir=str(processed_dataset_path.parent),
        prompt_style=prompt_style,
        file_source=file_source,
        validation_ratio=0.01,
        retrieval_file=None,
        k=None,
        max_context_len=None,
        tokenizer_name=None,
        push_to_hub_user=None,
    )
    return processed_dataset_path


def chat_completion(model: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
    """Call the Groq Chat Completion endpoint and return the assistant message text."""

    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError(
            "Environment variable GROQ_API_KEY must be set with your Groq API key."
        )

    response = _client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content  # type: ignore[no-any-return]

# -----------------------------------------------------------------------------
# Core generation routine
# -----------------------------------------------------------------------------

def generate_predictions(
    dataset_path: Path,
    split: str,
    model_name: str,
    output_file: Path,
    temperature: float,
    top_p: float,
    max_instances: int | None,
):
    """Iterate over *split* of *dataset_name* and call Groq to obtain patches."""

    # Ensure output directory exists and load any previously completed ids so the
    # script is resumable.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    continue

    # Load dataset from the prepared path.
    dataset = load_from_disk(str(dataset_path))[split]
    if max_instances is not None:
        dataset = dataset.select(range(min(max_instances, len(dataset))))

    print(f"Dataset loaded: {dataset_path} [{split}]  (size={len(dataset)})")
    print(f"Existing predictions detected for {len(existing_ids)} instances  they will be skipped.")

    with open(output_file, "a", encoding="utf-8") as fh_out:
        for row in tqdm(dataset, desc="Generating", unit="inst"):
            instance_id: str = row["instance_id"]
            if instance_id in existing_ids:
                continue

            # `run_api.py` expects each dataset row to provide a complete prompt in
            # the "text" field, where the first newline separates the **system**
            # message from the **user** message.
            if "text" not in row or row["text"] is None:
                # Omit instances that do not have a prompt
                print(f"[WARN] Skipping {instance_id} because it has no prompt.")
                continue

            prompt = row["text"] + "\n\n"
            system_msg, user_msg = prompt.split("\n", 1)
            print(f"System message: {system_msg}")
            print(f"User message: {user_msg}")
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            try:
                assistant_content = chat_completion(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                print(f"[WARN] Generation failed for {instance_id}: {e}")
                continue

            # Extract diff and attempt to automatically repair malformed hunks
            raw_patch = extract_diff(assistant_content)
            model_patch = repair_patch(raw_patch)
            record = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "full_output": assistant_content,
                "model_patch": model_patch,
            }
            fh_out.write(json.dumps(record) + "\n")
            fh_out.flush()

# -----------------------------------------------------------------------------
# Optional evaluation helper
# -----------------------------------------------------------------------------


def evaluate_predictions(
    dataset_name: str,
    split: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
):
    """Invoke the built-in SWE-bench harness to evaluate *predictions_path*."""

    from swebench.harness import run_evaluation  # late import to avoid Docker deps unless needed

    run_evaluation.main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=[],  # evaluate all predictions present
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="all",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
        namespace=None,
        rewrite_reports=False,
        modal=False,
    )

# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Llama 3 on SWE-bench via Groq and optionally evaluate.")
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model name (e.g. llama3-8b-8192)")
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory to write *.jsonl predictions",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="style-3",
        help="Prompt style for constructing the `text` field (see `create_text_dataset.py`)",
    )
    parser.add_argument(
        "--file_source",
        type=str,
        default="oracle",
        help="File source for constructing the `text` field (see `create_text_dataset.py`)",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="If set, only process the first N instances (useful for quick tests)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After generation, run the SWE-bench evaluation harness (Docker required)",
    )
    parser.add_argument("--run_id", default="groq-run", help="Identifier for the evaluation run logs")
    parser.add_argument("--max_workers", type=int, default=14, help="Workers for evaluation harness")

    args = parser.parse_args()

    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("Environment variable GROQ_API_KEY must be set with your Groq API key.")

    dataset_path = prepare_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=Path(args.output_dir),
        prompt_style=args.prompt_style,
        file_source=args.file_source,
    )

    output_file = Path(args.output_dir) / f"{args.model.replace('/', '__')}__{args.dataset_name.split('/')[-1]}__{args.split}.jsonl"

    generate_predictions(
        dataset_path=dataset_path,
        split=args.split,
        model_name=args.model,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        max_instances=args.max_instances,
    )

    if args.evaluate:
        evaluate_predictions(
            dataset_name=args.dataset_name,
            split=args.split,
            predictions_path=output_file,
            run_id=args.run_id,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main() 