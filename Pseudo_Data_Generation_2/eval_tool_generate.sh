#!/bin/bash
set -e
set -o pipefail

######################################
# Common
######################################
pwd_path=$(pwd)

######################################
# ========== GROUP 1 ==========
######################################
echo "================ GROUP 1 START ================"

output1=qwen3_4b_merge_12part_public_1
output2=qwen3_4b_merge_12part_public_2
model1_name=Qwen/Qwen3-4B
model2_name=Qwen/Qwen3-4B

input="$pwd_path/data/merged_all_12parts.jsonl"
out1="$pwd_path/data/merged_all_12parts_1.jsonl"
out2="$pwd_path/data/merged_all_12parts_2.jsonl"

# split input
total_lines=$(wc -l < "$input")
half=$(( total_lines / 2 ))

head -n "$half" "$input" > "$out1"
tail -n $(( total_lines - half )) "$input" > "$out2"

# ---------- retrieve inference ----------
echo "[Group1] Starting retrieve inference..."

export CUDA_VISIBLE_DEVICES=0,1,2,3
python retrieve_tool_local_file.py \
  --input_path "$out1" \
  --output_path "$pwd_path/rollout/merged_test_${output1}.jsonl" \
  --model_name_or_path "$model1_name" \
  --n 1 \
  --exe_code \
  --batch_size 1 \
  --parallel_size 4 \
  --max_tokens 4096 \
  > "acc/merged_test_${output1}.log" 2>&1 &

pid1=$!

export CUDA_VISIBLE_DEVICES=4,5,6,7
python retrieve_tool_local_file.py \
  --input_path "$out2" \
  --output_path "$pwd_path/rollout/merged_test_${output2}.jsonl" \
  --model_name_or_path "$model2_name" \
  --n 1 \
  --exe_code \
  --batch_size 1 \
  --parallel_size 4 \
  --max_tokens 4096 \
  > "acc/merged_test_${output2}.log" 2>&1 &

pid2=$!

echo "[Group1] Waiting retrieve jobs..."
wait $pid1
wait $pid2
echo "[Group1] Retrieve finished."

# ---------- judge ----------
echo "[Group1] Starting judge..."

export CUDA_VISIBLE_DEVICES=0,1,2,3
python llm_judge.py \
  --file_path "$pwd_path/rollout/merged_test_${output1}.jsonl" \
  --model_name_or_path Qwen/Qwen3-8B \
  --parallel_size 4 &

pid3=$!

export CUDA_VISIBLE_DEVICES=4,5,6,7
python llm_judge.py \
  --file_path "$pwd_path/rollout/merged_test_${output2}.jsonl" \
  --model_name_or_path Qwen/Qwen3-8B \
  --parallel_size 4 &

pid4=$!

wait $pid3
wait $pid4
echo "================ GROUP 1 DONE ================"


######################################
# ========== GROUP 2 ==========
######################################
echo "================ GROUP 2 START ================"

# clean trajectory files
rm -f trajectory_files/*

output1=qwen3_4b_merge_3_public
output2=qwen3_4b_merge_4_public
model1_name=Qwen/Qwen3-4B
model2_name=Qwen/Qwen3-4B

input="$pwd_path/data/merged_train_full_asg.jsonl"
out1="$pwd_path/data/merged_output3_32k_1.jsonl"
out2="$pwd_path/data/merged_output3_32k_2.jsonl"

# split input
total_lines=$(wc -l < "$input")
half=$(( total_lines / 2 ))

head -n "$half" "$input" > "$out1"
tail -n $(( total_lines - half )) "$input" > "$out2"

# ---------- retrieve inference ----------
echo "[Group2] Starting retrieve inference..."

export CUDA_VISIBLE_DEVICES=0,1,2,3
python retrieve_tool_local_file.py \
  --input_path "$out1" \
  --output_path "$pwd_path/rollout/merged_test_${output1}.jsonl" \
  --model_name_or_path "$model1_name" \
  --n 1 \
  --exe_code \
  --batch_size 1 \
  --parallel_size 4 \
  --max_tokens 4096 \
  > "acc/merged_test_${output1}.log" 2>&1 &

pid1=$!

export CUDA_VISIBLE_DEVICES=4,5,6,7
python retrieve_tool_local_file.py \
  --input_path "$out2" \
  --output_path "$pwd_path/rollout/merged_test_${output2}.jsonl" \
  --model_name_or_path "$model2_name" \
  --n 1 \
  --exe_code \
  --batch_size 1 \
  --parallel_size 4 \
  --max_tokens 4096 \
  > "acc/merged_test_${output2}.log" 2>&1 &

pid2=$!

wait $pid1
wait $pid2
echo "[Group2] Retrieve finished."

# ---------- judge ----------
echo "[Group2] Starting judge..."

export CUDA_VISIBLE_DEVICES=0,1,2,3
python llm_judge.py \
  --file_path "$pwd_path/rollout/merged_test_${output1}.jsonl" \
  --model_name_or_path Qwen/Qwen3-8B \
  --parallel_size 4 &

pid3=$!

export CUDA_VISIBLE_DEVICES=4,5,6,7
python llm_judge.py \
  --file_path "$pwd_path/rollout/merged_test_${output2}.jsonl" \
  --model_name_or_path Qwen/Qwen3-8B \
  --parallel_size 4 &

pid4=$!

wait $pid3
wait $pid4
echo "================ GROUP 2 DONE ================"

echo "🎉 ALL JOBS FINISHED SUCCESSFULLY 🎉"
