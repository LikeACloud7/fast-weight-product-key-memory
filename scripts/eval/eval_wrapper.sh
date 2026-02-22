#!/bin/bash

# WANDB_RUN_NAME, MODEL_CFG, CHECKPOINT_PATH, MODEL_TYPE from command line args
WANDB_RUN_NAME=$1
MODEL_CFG=$2
CHECKPOINT_PATH=$3
MODEL_TYPE=$4

# Decide MODEL_TYPE by checking if "lact" in MODEL_CFG
if [[ "$MODEL_CFG" == *"lact"* ]]; then
    MODEL_TYPE="lact"
else
    MODEL_TYPE="qwen3_next_mem"
fi

# Exit if checkpoint path does not exist
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path $CHECKPOINT_PATH not found."
    exit 1
fi

############### Eval ###############

# Longbench PPL evaluation
bash scripts/eval/longbench_ppl.sh \
    "$WANDB_RUN_NAME" \
    "$MODEL_CFG" \
    "$CHECKPOINT_PATH" \
    "$MODEL_TYPE"

# PPL stream
bash scripts/eval/ppl_stream.sh \
    "$WANDB_RUN_NAME" \
    "$MODEL_CFG" \
    "$CHECKPOINT_PATH" \
    "$MODEL_TYPE"

# Gate
bash scripts/eval/gate.sh \
    "$WANDB_RUN_NAME" \
    "$MODEL_CFG" \
    "$CHECKPOINT_PATH" \
    "$MODEL_TYPE"

# Pile domain
bash scripts/eval/pile_domain.sh \
    "$WANDB_RUN_NAME" \
    "$MODEL_CFG" \
    "$CHECKPOINT_PATH" \
    "$MODEL_TYPE"

# NIAH evaluation
bash scripts/eval/niah.sh \
    "$WANDB_RUN_NAME" \
    "$MODEL_CFG" \
    "$CHECKPOINT_PATH" \
    "$MODEL_TYPE"
