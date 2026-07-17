#!/bin/bash


MODEL_DIRS=(
   "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-neosft-rlvr"
)

STEPS=(120 180 240 300)


PWD_PATH=$(pwd)

INPUT_32K="$PWD_PATH/data/factuality_32k_test.jsonl"
INPUT_128K="$PWD_PATH/data/factuality_128k_test.jsonl"

ROLL_OUT_DIR="$PWD_PATH/rollout"
ACC_DIR="$PWD_PATH/acc"
mkdir -p $ROLL_OUT_DIR
mkdir -p $ACC_DIR

PARALLEL_SIZE=4
BATCH_SIZE=1

echo "=== Starting evaluation on 4 models (each 5 checkpoints) ==="

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    echo "--------------------------------------------"
    echo "Processing MODEL DIR: $MODEL_DIR"
    echo "--------------------------------------------"

    for STEP in "${STEPS[@]}"; do
        STEP_MODEL="${MODEL_DIR}/global_step_${STEP}_consolidate"

        if [ ! -d "$STEP_MODEL" ]; then
            echo "  Skipping missing checkpoint: $STEP_MODEL"
            continue
        fi

        OUTPUT_PREFIX="$(basename $MODEL_DIR)_step_${STEP}"
        echo "=== Evaluating checkpoint: $OUTPUT_PREFIX ==="

        # ========== 32k inference ==========
        echo "Starting 32k inference..."
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        python retrieve_tool_local_file.py \
            --input_path $INPUT_32K \
            --output_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
            --model_name_or_path $STEP_MODEL \
            --n 1 \
            --exe_code \
            --batch_size $BATCH_SIZE \
            --parallel_size $PARALLEL_SIZE \
            --max_tokens 4096 > $ACC_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &

        pid32=$!

        # ========== 128k inference ==========
        echo "Starting 128k inference..."
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        python retrieve_tool_local_file.py \
            --input_path $INPUT_128K \
            --output_path $ROLL_OUT_DIR/factuality_128k_test_tool_${OUTPUT_PREFIX}.jsonl \
            --model_name_or_path $STEP_MODEL \
            --n 1 \
            --exe_code \
            --batch_size $BATCH_SIZE \
            --parallel_size $PARALLEL_SIZE \
            --max_tokens 4096 > $ACC_DIR/factuality_128k_test_tool_${OUTPUT_PREFIX}.log 2>&1 &

        pid128=$!

        wait $pid32
        wait $pid128
        echo "Inference done for $OUTPUT_PREFIX"

        # ========== judge ==========
        echo "Starting judge for $OUTPUT_PREFIX..."

        export CUDA_VISIBLE_DEVICES=0,1,2,3
        python llm_judge.py \
            --file_path $ROLL_OUT_DIR/factuality_32k_test_tool_${OUTPUT_PREFIX}.jsonl \
            --model_name_or_path Qwen/Qwen3-8B \
            --parallel_size $PARALLEL_SIZE &

        export CUDA_VISIBLE_DEVICES=4,5,6,7
        python llm_judge.py \
            --file_path $ROLL_OUT_DIR/factuality_128k_test_tool_${OUTPUT_PREFIX}.jsonl \
            --model_name_or_path Qwen/Qwen3-8B \
            --parallel_size $PARALLEL_SIZE

        echo "Judge done for $OUTPUT_PREFIX"
    done
done

echo "=== All evaluations finished ==="
