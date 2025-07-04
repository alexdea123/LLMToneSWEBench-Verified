#!/usr/bin/env python3
"""
Analysis script for comparing the effectiveness of different influence tactics
in the SWE-bench experiment.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import statistics

from tactics import SWEBenchTactics


def load_predictions(file_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return predictions


def analyze_single_tactic(predictions: List[Dict[str, Any]], tactic_name: str) -> Dict[str, Any]:
    """Analyze predictions for a single tactic."""
    total = len(predictions)
    
    # Count predictions with valid patches (non-empty model_patch)
    valid_patches = sum(1 for p in predictions if p.get('model_patch', '').strip())
    
    # Analyze output lengths
    output_lengths = [len(p.get('full_output', '')) for p in predictions]
    patch_lengths = [len(p.get('model_patch', '')) for p in predictions if p.get('model_patch', '').strip()]
    
    analysis = {
        'tactic': tactic_name,
        'total_predictions': total,
        'valid_patches': valid_patches,
        'valid_patch_rate': valid_patches / total if total > 0 else 0,
        'avg_output_length': statistics.mean(output_lengths) if output_lengths else 0,
        'median_output_length': statistics.median(output_lengths) if output_lengths else 0,
        'avg_patch_length': statistics.mean(patch_lengths) if patch_lengths else 0,
        'median_patch_length': statistics.median(patch_lengths) if patch_lengths else 0,
    }
    
    return analysis


def find_tactic_files(output_dir: Path, model_name: str, dataset_name: str, split: str) -> Dict[str, Path]:
    """Find all tactic result files in the output directory."""
    tactic_files = {}
    
    # Clean up names for filename matching
    model_clean = model_name.replace('/', '__')
    dataset_clean = dataset_name.split('/')[-1]
    
    for tactic in SWEBenchTactics:
        expected_filename = f"{model_clean}__{dataset_clean}__{split}__tactic-{tactic.name}.jsonl"
        file_path = output_dir / expected_filename
        
        if file_path.exists():
            tactic_files[tactic.name] = file_path
        else:
            print(f"⚠ Missing results for tactic: {tactic.name} (expected: {expected_filename})")
    
    return tactic_files


def main():
    parser = argparse.ArgumentParser(description="Analyze influence tactic experiment results")
    parser.add_argument(
        "--output_dir", 
        default="./outputs", 
        help="Directory containing the prediction JSONL files"
    )
    parser.add_argument(
        "--model", 
        default="llama-3.1-8b-instant", 
        help="Model name used in the experiments"
    )
    parser.add_argument(
        "--dataset_name", 
        default="princeton-nlp/SWE-bench_Verified", 
        help="Dataset name used in the experiments"
    )
    parser.add_argument("--split", default="test", help="Dataset split used")
    parser.add_argument(
        "--output_file", 
        help="Save analysis results to JSON file"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Find all tactic result files
    tactic_files = find_tactic_files(output_dir, args.model, args.dataset_name, args.split)
    
    if not tactic_files:
        print("❌ No tactic result files found!")
        return
    
    print(f"📊 Found results for {len(tactic_files)} tactics: {', '.join(tactic_files.keys())}")
    print()
    
    # Analyze each tactic
    analyses = []
    for tactic_name, file_path in tactic_files.items():
        predictions = load_predictions(file_path)
        analysis = analyze_single_tactic(predictions, tactic_name)
        analyses.append(analysis)
        
        print(f"📈 {tactic_name}:")
        print(f"   Total predictions: {analysis['total_predictions']}")
        print(f"   Valid patches: {analysis['valid_patches']} ({analysis['valid_patch_rate']:.1%})")
        print(f"   Avg output length: {analysis['avg_output_length']:.0f} chars")
        print(f"   Avg patch length: {analysis['avg_patch_length']:.0f} chars")
        print()
    
    # Summary comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort by valid patch rate
    analyses.sort(key=lambda x: x['valid_patch_rate'], reverse=True)
    
    print("📊 Tactics ranked by valid patch rate:")
    for i, analysis in enumerate(analyses, 1):
        print(f"{i:2d}. {analysis['tactic']:20s} - {analysis['valid_patch_rate']:6.1%} "
              f"({analysis['valid_patches']}/{analysis['total_predictions']})")
    
    print()
    print("📝 Summary statistics:")
    patch_rates = [a['valid_patch_rate'] for a in analyses]
    print(f"   Best tactic: {analyses[0]['tactic']} ({analyses[0]['valid_patch_rate']:.1%})")
    print(f"   Worst tactic: {analyses[-1]['tactic']} ({analyses[-1]['valid_patch_rate']:.1%})")
    print(f"   Average patch rate: {statistics.mean(patch_rates):.1%}")
    print(f"   Standard deviation: {statistics.stdev(patch_rates):.1%}")
    print(f"   Range: {max(patch_rates) - min(patch_rates):.1%}")
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        print(f"\n💾 Analysis results saved to: {output_path}")


if __name__ == "__main__":
    main() 