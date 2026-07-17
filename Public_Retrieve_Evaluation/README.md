## 🚀 Running the Model

> **Overview:**  
> Run inference using either Bedrock API or local GPU deployment.

### Environment

- **vLLM:** `0.10.0`
- **Recommended Docker image:** `verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2`

```bash
# Pull the recommended image
docker pull verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

# Or install vLLM directly
pip install vllm==0.10.0
```

### Using Local GPU (on ASG)

```bash
# Step 1: Merge shared checkpoints
# (Required if multi-GPU training was used with Verl)
pip install verl==0.5
pip install pebble
pip install timeout_decorator
bash model_merge_all.sh # Remember to update the data path

# Step 2: Run inference in batches
bash run_all_step.sh # Remember to update the data path
```

### Using Bedrock API (on Mac)

```bash
# Step 1: Update the API key in retrieve_tool_local_file_api.py

# Step 2: Run all steps
bash run_all_step_api.sh
```