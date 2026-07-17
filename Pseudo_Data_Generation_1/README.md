# Part I: Prompting the LLM to Generate Shopping Trajectories

> **Overview:**  
> Read the ESCI dataset, randomly sample 300 queries per prompt, and use the LLM to generate corresponding shopping trajectories. Multiple prompt files enable parallel LLM execution.

## Workflow

```bash
cd batch/train

# Step 1: Read ESCI data and randomly select 300 queries per prompt
python random_pick_esci_asin.py

# Step 2: Construct prompt files (one prompt per line)
python synthetic_data_prompt_force.py

# Step 3: Run Claude API (update Bedrock API key; supports up to 12 parallel instances)
python run_claude.py
```

---

# Part II: Filtering Noisy Data and Checking Action Distributions

> **Note:**  
> Both training and test datasets are maintained separately. Token length constraints are applied during test set construction. Focus here is on the training set.

## Data Cleaning Pipeline

```bash
# Step 1: Read Claude-generated trajectories
python step1_read_save_file.py

# Step 2: Fix formatting issues and filter invalid action distributions
python step2_fix_noise_filter_dist.py

# Step 2.5: Verify action distribution
python step2.5_action_distribution.py

# Step 3: Prepare training data for question-answer construction
python step3_train_data.py


# Benchmark Construction

1. Update the API key at **line 34** in: `oss_generation/robust_benchmark_generator.py`
2. Run:
   ```bash
   bash oss_generation/run_multi_robust_generator.sh
   ```

# Step 4: Prepare training data for SFT-data construction
python step4_train_build_factuality_type_data.py
```

> **Next Steps:**  
> After obtaining clean shopping trajectories, proceed to the `Pseudo_Data_Generation_2` folder for SFT and RLVR data construction.
>
> For the test set pipeline, refer to the `test_build` directory.