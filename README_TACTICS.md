# Influence Tactics Research Experiment

This directory contains scripts for running and analyzing experiments to determine which influence tactics are most effective for LLM performance on SWE-bench tasks.

## Overview

The experiment tests 9 different influence tactics from social psychology:
- `RATIONAL_PERSUASION` - Appeals to logic and reason
- `EXCHANGE` - Offers benefits in return
- `INSPIRATIONAL_APPEAL` - Appeals to values and ideals  
- `INGRATIATION` - Uses flattery and praise
- `PERSONAL_APPEALS` - Appeals to friendship and personal relationships
- `LEGITIMATING` - Appeals to authority and rules
- `PRESSURE` - Uses demands and threats (direct supervision)
- `PRESSURE_ALTERNATIVE` - Uses demands and threats (post-hoc review)
- `NEUTRAL` - Baseline with no influence tactic

## Files

- `tone_experiment.py` - Main experiment script (modified to support tactics)
- `tactics.py` - Definitions of all influence tactics
- `run_all_tactics.py` - Convenience script to run all tactics sequentially
- `analyze_tactics.py` - Analysis script to compare results across tactics

## Quick Start

### 1. Run a Single Tactic Experiment

```bash
# Run with a specific tactic
python tone_experiment.py --tactic RATIONAL_PERSUASION --max_instances 10

# Run with default neutral tactic
python tone_experiment.py --max_instances 10
```

### 2. Run All Tactics (Recommended for Research)

```bash
# Run all tactics with default settings (small test)
python run_all_tactics.py --max_instances 5

# Run all tactics on full dataset
python run_all_tactics.py

# Run specific tactics only
python run_all_tactics.py --tactics NEUTRAL RATIONAL_PERSUASION PRESSURE

# Resume from a specific tactic if interrupted
python run_all_tactics.py --start_from INSPIRATIONAL_APPEAL
```

### 3. Analyze Results

```bash
# Analyze all tactic results
python analyze_tactics.py

# Save analysis to file
python analyze_tactics.py --output_file tactic_analysis.json
```

## Output Files

Each tactic run creates a separate output file with the tactic name included:
```
outputs/llama-3.1-8b-instant__SWE-bench_Verified__test__tactic-NEUTRAL.jsonl
outputs/llama-3.1-8b-instant__SWE-bench_Verified__test__tactic-RATIONAL_PERSUASION.jsonl
...
```

This ensures no overwriting between different tactic runs.

## Research Best Practices

### Experimental Design
- **Run all tactics**: Use `run_all_tactics.py` to ensure consistent conditions
- **Use same dataset split**: Keep `--split test` consistent across runs
- **Control for randomness**: Use same `--temperature` and `--top_p` values
- **Sample size**: Start with `--max_instances 10-50` for pilot testing

### Data Collection
- **Resumable runs**: The scripts support resuming interrupted experiments
- **Separate files**: Each tactic gets its own output file to prevent contamination
- **Metadata tracking**: Each prediction includes the tactic name for traceability

### Analysis
- **Valid patch rate**: Primary metric (percentage of predictions with non-empty patches)
- **Output characteristics**: Length analysis can reveal tactic effects on verbosity
- **Statistical comparison**: Use the analysis script for quick comparisons

## Example Workflows

### Pilot Study
```bash
# Quick test with 10 instances per tactic
python run_all_tactics.py --max_instances 10 --dataset_name princeton-nlp/SWE-bench_Lite

# Analyze pilot results
python analyze_tactics.py
```

### Full Study
```bash
# Run all tactics on full verified dataset
python run_all_tactics.py --dataset_name princeton-nlp/SWE-bench_Verified

# With evaluation (requires Docker)
python run_all_tactics.py --evaluate

# Analyze final results
python analyze_tactics.py --output_file final_analysis.json
```

### Troubleshooting Failed Tactics
```bash
# If some tactics failed, retry just those
python run_all_tactics.py --tactics PRESSURE LEGITIMATING

# Or resume from where it stopped
python run_all_tactics.py --start_from PRESSURE
```

## Environment Setup

Ensure you have:
- `GROQ_API_KEY` environment variable set
- All required dependencies from the original script
- Sufficient disk space for output files

## Expected Results

The analysis script will rank tactics by:
1. **Valid patch rate** - Most important metric
2. **Output characteristics** - Length, completeness
3. **Statistical summaries** - Mean, standard deviation, range

This allows you to identify which influence tactics are most effective for improving LLM performance on code generation tasks. 