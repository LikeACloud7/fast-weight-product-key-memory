#!/bin/bash

# WANDB_RUN_NAME, MODEL_CFG, CHECKPOINT_PATH, MODEL_TYPE from command line args
WANDB_RUN_NAME=$1
MODEL_CFG=$2
CHECKPOINT_PATH=$3
MODEL_TYPE=$4

WANDB_PROJECT="fwpkm_eval_pile_domain"
MASTER_PORT=$(shuf -i 12300-65535 -n 1)
deepspeed --include localhost:0 --master_port $MASTER_PORT --module src.eval_pile_domain \
    --model_type $MODEL_TYPE \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --pretrained_config_path $MODEL_CFG \
    --pretrained_model_dir $CHECKPOINT_PATH \
    --override_attn_implementation flash_attention_2 \
    --micro_batch_size 8 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME
