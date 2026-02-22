# Standard training of `l12-gdn-pkm6-fwpkm2_10`.
# Change variable CFG to train other models.
SEED=1; \
CFG=main/l12-gdn-pkm6-fwpkm2_10; \
MASTER_PORT=$(shuf -i 12300-65535 -n 1); \
deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT --module src.pretrain \
    -c \
        cfgs/ike_config/fineweb_lc64/train.cfg \
    --model_type qwen3_next_mem \
    --pretrained_config_path cfgs/model_config/$CFG.json \
    --peak_lr 0.0003 \
    --min_lr 0.00003 \
    --seed $SEED \
    --override_attn_implementation flash_attention_2 \
    --micro_batch_size 8 --micro_valid_batch_size 32 \
    --log_grad_norms \
    --log_weight_norms \
    --save_log --save_model \
    --save_wandb \
    --wandb_project fwpkm_train \
    --wandb_run_name $CFG

# Training of `l12-fa-pkm6-fwpkm2_10` with pSWA=0.9.
SEED=1; \
CFG=main/l12-fa-pkm6-fwpkm2_10; \
MASTER_PORT=$(shuf -i 12300-65535 -n 1); \
deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT --module src.pretrain \
    -c \
        cfgs/ike_config/fineweb_lc64/train.cfg \
    --model_type qwen3_next_mem \
    --pretrained_config_path cfgs/model_config/$CFG.json \
    --peak_lr 0.0003 \
    --min_lr 0.00003 \
    --seed $SEED \
    --override_attn_implementation flash_attention_2 \
    --micro_batch_size 8 --micro_valid_batch_size 32 \
    --log_grad_norms \
    --log_weight_norms \
    --save_log --save_model \
    --save_wandb \
    --wandb_project fwpkm_train \
    --wandb_run_name $CFG/pSWA_0.9 \
    --fa2swa_prob 0.9
