#!/bin/bash


MODEL_DIRS=(
    "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-neosft-rlvr"
)

STEPS=(120 180 240 300)

echo "Starting merges..."

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    echo "Processing directory: $MODEL_DIR"

    for STEP in "${STEPS[@]}"; do
        STEP_DIR="${MODEL_DIR}/global_step_${STEP}/actor"

        if [ -d "$STEP_DIR" ]; then
            TARGET_DIR="${MODEL_DIR}/global_step_${STEP}_consolidate"

            echo "--------------------------------------------------"
            echo "Merging:"
            echo "  from: $STEP_DIR"
            echo "  to:   $TARGET_DIR"

            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$STEP_DIR" \
                --target_dir "$TARGET_DIR"

            if [ $? -ne 0 ]; then
                echo "  Error merging $STEP_DIR. Skipping."
            fi
        else
            echo "  Skipping: $STEP_DIR does not exist."
        fi
    done

done

echo "All merges completed."
