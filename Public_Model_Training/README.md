## 🐳 Docker Setup for Verl Training and Sandbox

> **Overview:**  
> Set up two Docker containers: one for Verl training and one for the sandbox environment. Copy required tool packages and run training.

---

### Start Containers

```bash
# Sandbox container
docker run -it --name hysdb_test -p 8080:8080 volcengine/sandbox-fusion:server-20250609

# Note:
# The port must be consistent with:
# - ./recipe/retool/sandbox_fusion_tool_config.yaml
# - save_traj_manual.py

# Verl training container
docker run -d --name hyenv_test \
  --gpus all \
  --net=host \
  --shm-size=40g \
  -v /home/hongyee/hyenv/:/workspace/hyenv \
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
  sleep infinity
```

---

### Copy Tool Packages to Sandbox

```bash
docker cp /home/hongyee/hyenv/final_collection/Public_Model_Training/convert_trajectory_with_schema.py \
  hysdb_test:/root/miniconda3/envs/sandbox-runtime/lib/python3.10/site-packages/

docker cp /home/hongyee/hyenv/final_collection/Public_Model_Training/data_loader.py \
  hysdb_test:/root/miniconda3/envs/sandbox-runtime/lib/python3.10/site-packages/
```

---

### Prepare Data

```bash
# Enter the Verl container
docker exec -it hyenv_test bash

# Install sandbox dependency
pip install sandbox_fusion

# Save all shopping trajectories inside the sandbox
python save_traj_manual.py
```

---

### Start Training in Verl Container

```bash
docker exec -it hyenv_test bash
cd hyenv

# Run SFT training
wandb login  # Input your Weights & Biases API key
bash run_qwen3_sft_multiturn.sh

# Merge checkpoints
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /workspace/hyenv/final_collection/Public_Model_Training/checkpoint/Qwen/Qwen3-4B_public_sft/global_step_20 \
  --target_dir /workspace/hyenv/final_collection/Public_Model_Training/checkpoint/Qwen/Qwen3-4B_public_sft/global_step_20_consolidate

# Run RLVR training
VLLM_USE_V1=1 ray start --head

# Notes:
# - Remember to update the rollout path in ./recipe/retool/shopping_retool.py
# - Remember to update the model path in run_qwen3_sft_multiturn_rlvr.sh
bash run_qwen3_sft_multiturn_rlvr.sh
```

The training and evaluation data are published on Hugging Face: https://huggingface.co/datasets/hongyeeliu/ShopTrajQA