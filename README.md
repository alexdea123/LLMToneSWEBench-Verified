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

### Generate Predictions

To generate predictions using a Groq model (e.g., `llama3-8b-8192`) on the SWE-bench_Verified dataset:

```bash
python experiment.py --model llama3-8b-8192 --dataset_name princeton-nlp/SWE-bench_Verified --split test --max_instances 10 # For a quick test
```

### Evaluate Predictions

To evaluate the generated predictions (requires Docker):

```bash
python experiment.py --evaluate --model llama3-70b-8192 --dataset_name princeton-nlp/SWE-bench_Verified --split test --run_id my-evaluation-run
```

This will generate a `outputs/llama3-70b-8192__SWE-bench_Verified__test.jsonl` file and then run the SWE-bench evaluation harness.