import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from groq import Groq

from datasets import load_dataset  # HuggingFace datasets
from tqdm.auto import tqdm

# SWE-bench helper to post-process model outputs into valid unified diffs
from swebench.inference.make_datasets.utils import extract_diff

# -----------------------------------------------------------------------------
# Groq helper
# -----------------------------------------------------------------------------

# Instantiate a Groq client using your API key from the environment.
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    dataset_name: str,
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

    # Load dataset via HF. (Will download if not present in local cache.)
    dataset = load_dataset(dataset_name, split=split)
    if max_instances is not None:
        dataset = dataset.select(range(min(max_instances, len(dataset))))

    print(f"Dataset loaded: {dataset_name} [{split}]  (size={len(dataset)})")
    print(f"Existing predictions detected for {len(existing_ids)} instances  they will be skipped.")

    with open(output_file, "a", encoding="utf-8") as fh_out:
        for row in tqdm(dataset, desc="Generating", unit="inst"):
            instance_id: str = row["instance_id"]
            if instance_id in existing_ids:
                continue

            problem_statement: str = row["problem_statement"]
            system_msg = "You are a helpful software-engineering assistant. Your task is to provide a unified diff patch to fix the given problem."
            user_msg = f"Problem statement: {problem_statement}\n\nPlease provide the patch:"

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

            model_patch = extract_diff(assistant_content)
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
    parser.add_argument("--model", default="llama3-70b-8192", help="Groq model name (e.g. llama3-8b-8192)")
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory to write *.jsonl predictions",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
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
    parser.add_argument("--max_workers", type=int, default=12, help="Workers for evaluation harness")

    args = parser.parse_args()

    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("Environment variable GROQ_API_KEY must be set with your Groq API key.")

    output_file = Path(args.output_dir) / f"{args.model.replace('/', '__')}__{args.dataset_name.split('/')[-1]}__{args.split}.jsonl"

    generate_predictions(
        dataset_name=args.dataset_name,
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