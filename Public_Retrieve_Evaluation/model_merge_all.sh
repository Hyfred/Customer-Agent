#!/bin/bash

MODEL_DIR="/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-publicsft-rlvr"
# multiturn-sft-qwen-3-4b-instruct-pandasql
# loop for global_step_* folder
for STEP_DIR in "$MODEL_DIR"/global_step_*; do
    if [ -d "$STEP_DIR" ]; then
        TARGET_DIR="${STEP_DIR}_consolidate"

        echo "Merging:"
        echo "  from: $STEP_DIR"
        echo "  to:   $TARGET_DIR"

        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$STEP_DIR/actor" \
            --target_dir "$TARGET_DIR"

        if [ $? -ne 0 ]; then
            echo "  Error merging $STEP_DIR. Skipping."
        fi
    fi
done

echo "All merges completed."