## 📊 Generate Multi-Turn SFT Data

> **Overview:**  
> Copy the evaluation script, run it in the target directory, then proceed to SFT/RLVR data processing.

```bash
# Step 1: Copy the evaluation script to the target directory
cp eval_tool_generate.sh ../Public_Retrieve_Evaluation

# Step 2: Navigate to the target directory and run the script
cd ../Public_Retrieve_Evaluation
bash eval_tool_generate.sh

( sed '$d' data/merged_test_qwen3_4b_merge_12part_public_1.jsonl | sed '$d'
  sed '$d' data/merged_test_qwen3_4b_merge_12part_public_2.jsonl | sed '$d'
) > data/merge_output_sft_pre.jsonl
```

- **Next step:**  
  Proceed to the `process_sft_rlvr_data` step

```bash
python process_sft_rlvr_data/convert_neo_multi_sft.py
python process_sft_rlvr_data/convert_neo_multi_rlvr.py
```

> **Note:**  
> The `internal_data_process` step is not required for public data. The overall logic remains the same.