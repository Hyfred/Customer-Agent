
## Step 1: Read Claude-generated trajectories
`python step1_read_save_file.py`

## Step 2: Fix formatting issues and filter data with incorrect action distributions
`python step2_fix_noise_filter_dist.py`

## Step 2.5: Verify the correctness of the action distribution
`python step2.5_action_distribution.py`

## Step 3: Build datasets with different token lengths (test set only)
`python step3_test_cut2version_pgeval_data.py`

## Step 3.5: Recompute action distributions for benchmark evaluation
`python step3.5_stat_benchmark_data.py`

(refers `oss_generation` to build question-answer then move to step 4)

```bash
python robust_benchmark_generator.py \
  --input /path/to/data/pseudo_data/benchmark_data/step3_32k_new.json \
  --output /path/to/data/pseudo_data/benchmark_data/step3_32k_new_gptoss.json \
  --trajectory-mode 32k \
  --num-trajectories 609 \
  --questions-per-trajectory 2 \
  --easy-ratio 0.6 \
  --medium-ratio 0.4 \
  --hard-ratio 0.0 \
  --total-questions 3045 \
  --balanced-depth

python robust_benchmark_generator.py \
  --input /path/to/data/pseudo_data/benchmark_data/step3_full_new.json \
  --output /path/to/data/pseudo_data/benchmark_data/step3_full_new_gptoss.json \
  --trajectory-mode 32k \
  --num-trajectories 609 \
  --questions-per-trajectory 2 \
  --easy-ratio 0.6 \
  --medium-ratio 0.4 \
  --hard-ratio 0.0 \
  --total-questions 3045 \
  --balanced-depth
```

## Step 4: Build datasets for different factuality types
`python step4_test_build_factuality_type_data.py`