#!/bin/bash

MODEL_DIRS=(
   "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-1.7b-neosft-rlvr/1.7b_global_step_60_consolidate"
   "/workspace/hyenv/neo_rlvr_training/checkpoint/qwen3-4b-neosft-rlvr/4b_global_step_100_consolidate"
)


dataset_list=("AIME24" "AIME25")
prompt_template_path="prompt_template.json"
SRC_PATH1="phrase4_public_data_1.7b"
SRC_PATH2="phrase4_public_data_4b"


num_models=${#MODEL_DIRS[@]}

# each batch run two model
for ((i=0; i<$num_models; i+=2)); do

    MODEL1=${MODEL_DIRS[$i]}
    MODEL2=${MODEL_DIRS[$i+1]}

    echo "==== Running batch: $MODEL1 AND $MODEL2 ===="

    # -------- loop for two data --------
    for data in "${dataset_list[@]}"; do
        echo "---- Dataset: $data ----"

        # model 1（GPU 0-3）
        {
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python eval.py \
                --data_name $data \
                --target_path $SRC_PATH1 \
                --model_name_or_path $MODEL1 \
                --max_tokens 32768 \
                --paralle_size 4 \
                --n 8 \
                --prompt_template ${prompt_template_path} \
                --exe_code \
                --prompt retool
        } &
        PID1=$!

        # model 2（GPU 4-7）
        if [ -n "$MODEL2" ]; then
            {
                export CUDA_VISIBLE_DEVICES=4,5,6,7
                python eval.py \
                    --data_name $data \
                    --target_path $SRC_PATH2 \
                    --model_name_or_path $MODEL2 \
                    --max_tokens 32768 \
                    --paralle_size 4 \
                    --n 8 \
                    --prompt_template ${prompt_template_path} \
                    --exe_code \
                    --prompt retool
            } &
            PID2=$!
        fi

        wait $PID1
        if [ -n "$PID2" ]; then
            wait $PID2
        fi

        echo "---- Dataset $data done ----"
    done

    echo "==== Batch finished ===="

done

echo "==== ALL MODELS DONE ===="