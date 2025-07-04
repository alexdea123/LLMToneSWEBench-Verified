# LLMToneSWEBench-Verified

This repository contains the code for running experiments with Groq models on the SWE-bench dataset.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LLMToneSWEBench-Verified.git
    cd LLMToneSWEBench-Verified
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: `requirements.txt` is not yet provided, but will be needed for `datasets` and `groq`.)

3.  **Set your Groq API key:**
    ```bash
    export GROQ_API_KEY="your_groq_api_key_here"
    ```

## Usage

The **first** time you run generation on a given dataset the script will automatically
construct the prompts (a process that clones repositories, adds line numbers, etc.).
All subsequent runs reuse the cached version so you do **not** need to call any
extra command – the cache lives under `outputs/datasets/`.

### Generate Predictions

```bash
# Quick test – will automatically build and cache the prompt dataset the first time
python experiment.py \
  --model llama-3.1-8b-instant \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --max_instances 10
```

Optional flags:

* `--prompt_style`  (default: `style-3`) – choose an alternative prompt format defined in `swebench/inference/make_datasets/create_instance.py`.
* `--file_source`   (default: `oracle`) – choose which repository files to include in context (`oracle`, `bm25`, `all`, `none`).

### Evaluate Predictions

To evaluate the generated predictions (requires Docker):

```bash
python experiment.py --evaluate --model llama-3.1-8b-instant --dataset_name princeton-nlp/SWE-bench_Verified --split test --run_id my-evaluation-run
```

This will generate a `outputs/llama3-70b-8192__SWE-bench_Verified__test.jsonl` file and then run the SWE-bench evaluation harness.