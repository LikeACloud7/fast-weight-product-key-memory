<div align="center">
    <h1>FwPKM: Fast-weight Product Key Memory</h1>
    :scroll:<a href="https://arxiv.org/abs/2601.00671">Paper</a> |
    :octocat:<a href="https://github.com/SakanaAI/fast-weight-product-key-memory">GitHub</a>
</div>
<br><br>
<div align="center">
    <img height="500px" src="assets/FwPKM_arch.svg" />
</div>

---

## Updates

- **2026-02-22**: Initial release.

## Get started

Install dependencies as follows:

```bash
git clone https://github.com/SakanaAI/fast-weight-product-key-memory.git
cd fast-weight-product-key-memory
# Preferably use a virtual environment
# Tested with Python 3.12.11
bash install.sh
```

Prepare pre-training and evaluation data as follows:

```bash
bash scripts/data/fineweb.sh
bash scripts/data/lc64_nanogpt.sh
bash scripts/data/lambada.sh
bash scripts/data/pile_domain.sh
```

## Training

Example command for training `GDN | PKM@6 + FwPKM@2,10` on 4 GPUs:
```bash
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
```

See more examples in the `scripts/train` directory:
- `scripts/train/train_main.sh` - commands for training models in the main experiments.
- `scripts/train/train_baseline.sh` - commands for training baselines.
- `scripts/train/train_ablation.sh` - commands for training models in ablated studies.

## Evaluation

Individual evaluation commands are provided in the `scripts/eval` directory. Results will be logged to wandb.
- `scripts/eval/ppl_stream.sh` - PPL stream evaluation.
- `scripts/eval/longbench_ppl.sh` - LongBench PPL evaluation.
- `scripts/eval/gate.sh` - Gate analysis.
- `scripts/eval/niah.sh` - NIAH evaluation.
- `scripts/eval/pile_domain.sh` - Pile domain evaluation.

To run individual evaluation scripts, provide `WANDB_RUN_NAME, MODEL_CFG, CHECKPOINT_PATH, MODEL_TYPE` as command arguments to the evaluation script. For example, for Pile domain evaluation:

```bash
WANDB_RUN_NAME=main/l12-fa-pkm6-fwpkm2_10/fa2swa_p0.9
MODEL_CFG=cfgs/model_config/main/l12-fa-pkm6-fwpkm2_10.json
CHECKPOINT_PATH=./experiments/lm/main/l12-fa-pkm6-fwpkm2_10/fa2swa_p0.9_2026-02-20-07-05-31/checkpoint/best
MODEL_TYPE=qwen3_next_mem
bash scripts/eval/pile_domain.sh \
    $WANDB_RUN_NAME \
    $MODEL_CFG \
    $CHECKPOINT_PATH \
    $MODEL_TYPE
```

Alternatively, you can run all evaluations for multiple checkpoints in one go with slurm. using the `eval_wrapper.sh` script by providing a **checkpoint list tsv** file, where each line corresponds to a checkpoint and contains the following tab-separated fields: `<WANDB_RUN_NAME> <MODEL_CFG_PATH> <CHECKPOINT_PATH> <MODEL_TYPE>`. For example:

```bash
CHECKPOINT_LIST_FILE=ckpt_lists/example.tsv
bash scripts/eval/eval_wrapper.sh $CHECKPOINT_LIST_FILE
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhao2026fwpkm,
    title={Fast-weight Product Key Memory}, 
    author={Tianyu Zhao and Llion Jones},
    year={2026},
    eprint={2601.00671},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2601.00671}, 
}
```

