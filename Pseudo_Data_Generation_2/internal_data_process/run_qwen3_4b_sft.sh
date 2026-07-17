#!/bin/bash
set -x

nnodes=1
nproc_per_node=8
# master_addr=
# master_port=
# node_rank=${ARNOLD_ID:-0}

project_name=retool
# experiment_name=multiturn-sft-qwen-3-4b-instruct
experiment_name=multiturn-sft-qwen-3-4b-train3

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=./data/neo_shopping_qwen3_30b.parquet
EVAL_DATA=./data/neo_shopping_qwen3_30b.parquet
MODEL_PATH=Qwen/Qwen3-4B
# MODEL_PATH=Qwen/Qwen3-1.7B
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

# torchrun --nnodes=$nnodes \
#      --nproc_per_node=$nproc_per_node \
#      -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=$TRAIN_DATA \
#     data.val_files=$EVAL_DATA \
#     data.max_length=16384 \
#     data.train_batch_size=32 \
#     data.multiturn.enable=true \
#     data.multiturn.messages_key=messages \
#     data.multiturn.tools_key=tools \
#     data.micro_batch_size_per_gpu=4 \
#     model.partial_pretrain=$MODEL_PATH \
#     model.strategy=fsdp \
#     trainer.default_local_dir=$SAVE_PATH \
#     trainer.project_name=wuxibin-multiturn-sft \
#     trainer.experiment_name=$experiment_name \
#     trainer.logger='["console","wandb"]' \
#     trainer.total_epochs=6 \
#     trainer.save_freq=62 \
#     ulysses_sequence_parallel_size=4 \
#     use_remove_padding=true

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=102400 \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=neo-shopping \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=15 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true \
