#!/bin/bash


STEP_MODELS=(
   "qwen3-30b-code"
)
# "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-neosft-rlvr/1.7b_global_step_240_consolidate"
PWD_PATH=$(pwd)

INPUT_32K="$PWD_PATH/data/merged_test_32k.jsonl"
INPUT_64K="$PWD_PATH/data/merged_test_full.jsonl"
ROLL_OUT_DIR="$PWD_PATH/rollout"
ACC_DIR="$PWD_PATH/acc"
PARALLEL_SIZE=4
BATCH_SIZE=1

mkdir -p $ROLL_OUT_DIR
mkdir -p $ACC_DIR

for STEP_MODEL in "${STEP_MODELS[@]}"; do
    OUTPUT_PREFIX=$(basename $STEP_MODEL)
    echo "=== Processing step: $OUTPUT_PREFIX ==="

    # -------- 32k inference --------
    echo "Starting 32k inference..."
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python retrieve_wotool_infer.py \
        --input_path $INPUT_32K \
        --output_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
        --api_type $STEP_MODEL \
        --n 1 \
        --batch_size $BATCH_SIZE \
        --max_tokens 64000 > $ACC_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &

    pid32=$!

    # -------- 128k inference --------
    echo "Starting 128k inference..."
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    python retrieve_wotool_infer.py \
        --input_path $INPUT_64K \
        --output_path $ROLL_OUT_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.jsonl \
        --api_type $STEP_MODEL \
        --n 1 \
        --batch_size $BATCH_SIZE \
        --max_tokens 128000 > $ACC_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &


    pid128=$!

    wait $pid32
    wait $pid128
    echo "Inference done for step $OUTPUT_PREFIX"

    # -------- judge --------
    # echo "Starting judge for step $OUTPUT_PREFIX..."
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
    # python llm_judge.py \
    #     --file_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
    #     --model_name_or_path Qwen/Qwen3-8B \
    #     --parallel_size $PARALLEL_SIZE &

    # export CUDA_VISIBLE_DEVICES=4,5,6,7
    # python llm_judge.py \
    #     --file_path $ROLL_OUT_DIR/factuality_64k_test_tool_${OUTPUT_PREFIX}.jsonl \
    #     --model_name_or_path Qwen/Qwen3-8B \
    #     --parallel_size $PARALLEL_SIZE

    # echo "Judge done for step $OUTPUT_PREFIX"
done

echo "=== All steps finished ==="
