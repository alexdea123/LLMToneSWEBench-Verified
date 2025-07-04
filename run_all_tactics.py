#!/usr/bin/env python3
"""
Convenience script to run tone_experiment.py with all available influence tactics.
This makes it easy to conduct the full research experiment comparing all tactics.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from tactics import SWEBenchTactics


def run_experiment_for_tactic(tactic_name: str, base_args: list[str]) -> bool:
    """Run the experiment for a specific tactic. Returns True if successful."""
    cmd = ["python", "tone_experiment.py"] + base_args + ["--tactic", tactic_name]
    
    print(f"\n{'='*60}")
    print(f"Running experiment with tactic: {tactic_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully completed experiment with tactic: {tactic_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed experiment with tactic: {tactic_name} (exit code: {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted during tactic: {tactic_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tone_experiment.py with all available influence tactics sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tactics with default settings
  python run_all_tactics.py

  # Run all tactics with custom dataset and max instances
  python run_all_tactics.py --dataset_name princeton-nlp/SWE-bench_Lite --max_instances 10
  
  # Run specific tactics only
  python run_all_tactics.py --tactics NEUTRAL RATIONAL_PERSUASION PRESSURE

  # Run with evaluation enabled
  python run_all_tactics.py --evaluate
        """
    )
    
    # Add option to run only specific tactics
    parser.add_argument(
        "--tactics",
        nargs="+",
        choices=[t.name for t in SWEBenchTactics],
        default=None,
        help="Specific tactics to run (default: all tactics)"
    )
    
    # Add option to continue from a specific tactic (useful if script was interrupted)
    parser.add_argument(
        "--start_from",
        choices=[t.name for t in SWEBenchTactics],
        default=None,
        help="Start from a specific tactic (useful for resuming interrupted runs)"
    )
    
    # Forward all other arguments to tone_experiment.py
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name or local path"
    )
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model name")
    parser.add_argument("--output_dir", default="./outputs", help="Directory to write *.jsonl predictions")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt_style", type=str, default="style-3", help="Prompt style")
    parser.add_argument("--file_source", type=str, default="oracle", help="File source")
    parser.add_argument("--max_instances", type=int, default=None, help="Max instances to process")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after generation")
    parser.add_argument("--run_id", default="groq-run", help="Identifier for evaluation run logs")
    parser.add_argument("--max_workers", type=int, default=14, help="Workers for evaluation")
    
    args = parser.parse_args()
    
    # Determine which tactics to run
    if args.tactics:
        tactics_to_run = args.tactics
    else:
        tactics_to_run = [t.name for t in SWEBenchTactics]
    
    # If starting from a specific tactic, filter the list
    if args.start_from:
        start_idx = tactics_to_run.index(args.start_from)
        tactics_to_run = tactics_to_run[start_idx:]
        print(f"Resuming from tactic: {args.start_from}")
    
    # Build base arguments to pass to tone_experiment.py
    base_args = []
    for key, value in vars(args).items():
        if key in ["tactics", "start_from"]:
            continue  # These are specific to this script
        if value is not None and value is not False:
            if isinstance(value, bool):
                base_args.append(f"--{key}")
            else:
                base_args.extend([f"--{key}", str(value)])
    
    print(f"Will run experiments for {len(tactics_to_run)} tactics: {', '.join(tactics_to_run)}")
    print(f"Base arguments: {' '.join(base_args)}")
    
    # Track results
    successful = []
    failed = []
    
    try:
        for tactic_name in tactics_to_run:
            if run_experiment_for_tactic(tactic_name, base_args):
                successful.append(tactic_name)
            else:
                failed.append(tactic_name)
    except KeyboardInterrupt:
        print("\n⚠ Experiment sequence interrupted by user")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful tactics ({len(successful)}): {', '.join(successful) if successful else 'None'}")
    print(f"✗ Failed tactics ({len(failed)}): {', '.join(failed) if failed else 'None'}")
    
    if failed:
        print(f"\nTo retry failed tactics:")
        print(f"python run_all_tactics.py --tactics {' '.join(failed)} {' '.join(base_args)}")
        sys.exit(1)
    else:
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()