from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Explore the SWE-bench dataset.")
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="If set, only display the first N instances",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.max_instances is not None:
        dataset = dataset.select(range(min(args.max_instances, len(dataset))))

    print(f"Displaying {len(dataset)} instances.\n")

    for i, row in enumerate(dataset):
        print(f"--- Instance {i+1} (ID: {row['instance_id']}) ---")
        print("Problem Statement:")
        print(row["problem_statement"])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 