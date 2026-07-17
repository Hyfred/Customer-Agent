#!/bin/bash
set -x

# nnodes=1
# nproc_per_node=8
# master_addr=
# master_port=
# node_rank=${ARNOLD_ID:-0}

project_name=shopping_retool
experiment_name=Qwen/Qwen3-4B_public_sft #multiturn-sft-qwen-3-4b-instruct

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=./data/public_shopping_multi-turn.parquet
EVAL_DATA=./data/public_shopping_multi-turn.parquet
MODEL_PATH=Qwen/Qwen3-4B #Qwen/Qwen3-1.7B #
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=public-shopping-multiturn \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=4 \
    trainer.save_freq=20 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true