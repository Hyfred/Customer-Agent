## 📦 Internal Shopping Trajectory Processing

> **Overview:**  
> Process internal shopping trajectories from S3, convert to PG-Eval format, generate training data, and run multi-turn SFT training.

### Prerequisites

```bash
# Step 0: Download internal shopping trajectories from your own S3 bucket
# s3://<your-bucket>/<path-to>/behavioral_playground_data/
```

### Workflow

```bash
# Step 1: Convert BE format to PG-Eval format
python convert_be2pgeval.py

# Generate robust benchmark data
python robust_benchmark_generator.py \
  --input ./data/be_pgeval.json \
  --output ./data/be_training_10q_32k.json \
  --trajectory-mode 32k \
  --num-trajectories 10 \
  --questions-per-trajectory 3 \
  --easy-ratio 0.6 \
  --medium-ratio 0.4 \
  --hard-ratio 0.0 \
  --total-questions 5 \
  --balanced-depth

# Step 2: Build factuality type data
python build_factuality_type_data.py
```

- **Output:** `merged_output.jsonl` (used for executable SQL)

```bash
# Step 3: Run inference to generate SFT pretraining data
cat path/to/dir/* > merged.txt

# Step 4: Convert to Parquet format for SFT data
python convert_parquet.py

# Step 5: Run multi-turn SFT training
bash run_qwen3_4b_sft_multiturn.sh

# Step 6: Merge outputs split across multiple machines
for f in data/output3_splitby4machine/*; do
    sed -e '$d' -e '$d' "$f"
done > data/merge_output3_from4machine.jsonl
```

> **Next Steps:**  
> Proceed to the SFT and RLVR data construction pipeline.