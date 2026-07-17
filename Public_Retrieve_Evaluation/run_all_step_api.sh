#!/bin/bash

MODEL_DIR="/workspace/hyenv/retool_verl/checkpoint/multiturn-sft-qwen-3-1.7b-train3"

STEP_MODELS=(
   "openai"
)
#    "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-neosft-rlvr/1.7b_global_step_240_consolidate"
PWD_PATH=$(pwd)

# folder
INPUT_32K="$PWD_PATH/data/merged_test_32k.jsonl"
INPUT_64K="$PWD_PATH/data/merged_test_full.jsonl"
ROLL_OUT_DIR="$PWD_PATH/rollout_morestep"
ACC_DIR="$PWD_PATH/acc_morestep"
PARALLEL_SIZE=4
BATCH_SIZE=1

mkdir -p $ROLL_OUT_DIR
mkdir -p $ACC_DIR

for STEP_MODEL in "${STEP_MODELS[@]}"; do
    OUTPUT_PREFIX=$(basename $STEP_MODEL)
    echo "=== Processing step: $OUTPUT_PREFIX ==="

    # -------- 32k inference --------
    echo "Starting 32k inference..."
    python retrieve_tool_local_file_api.py \
        --input_path $INPUT_32K \
        --output_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
        --api_type $STEP_MODEL \
        --n 1 \
        --exe_code \
        --batch_size $BATCH_SIZE \
        --max_tokens 4096 > $ACC_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &

    pid32=$!

    # -------- 128k inference --------
    echo "Starting 128k inference..."
    python retrieve_tool_local_file_api.py \
        --input_path $INPUT_64K \
        --output_path $ROLL_OUT_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.jsonl \
        --api_type $STEP_MODEL \
        --n 1 \
        --exe_code \
        --batch_size $BATCH_SIZE \
        --max_tokens 4096 > $ACC_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &

    pid128=$!

    wait $pid32
    wait $pid128
    echo "Inference done for step $OUTPUT_PREFIX"

    # -------- judge --------
    # echo "Starting judge for step $OUTPUT_PREFIX..."
    # python llm_judge.py \
    #     --file_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
    #     --model_name_or_path Qwen/Qwen3-8B \
    #     --parallel_size $PARALLEL_SIZE &

    # python llm_judge.py \
    #     --file_path $ROLL_OUT_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.jsonl \
    #     --model_name_or_path Qwen/Qwen3-8B \
    #     --parallel_size $PARALLEL_SIZE

    echo "Judge done for step $OUTPUT_PREFIX"
done

echo "=== All steps finished ==="
