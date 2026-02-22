import math
import mmap
import os

from ike import (
    configargparse as argparse,
    get_inference_arguments,
    is_rank_0,
)
from ike.util import build_tokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb


def build_model(config: argparse.Namespace, **kwargs):
    # Config
    if config.model_type == "qwen3_next_mem":
        from src.models.qwen3_next_mem import Qwen3NextMemConfig, Qwen3NextMemForCausalLM

        model_config = Qwen3NextMemConfig.from_pretrained(config.pretrained_config_path)
    elif config.model_type == "lact":
        from src.models.lact_hf_rope import LaCTSWIGLUConfig, LaCTForCausalLM

        model_config = LaCTSWIGLUConfig.from_pretrained(config.pretrained_config_path)

    # Override config
    if config.override_attn_implementation:
        model_config._attn_implementation = config.override_attn_implementation
    # Set RoPE scaling to max length according to config
    factor = math.ceil(config.max_seq_len / 4096)
    if factor > 1:
        model_config.rope_scaling = {"type": "yarn", "factor": factor}

    # Build model
    if config.model_type == "qwen3_next_mem":
        model = Qwen3NextMemForCausalLM(model_config)
    elif config.model_type == "lact":
        model = LaCTForCausalLM(model_config)

    # Load pretrained weights
    if config.pretrained_model_dir:
        if os.path.exists(os.path.join(config.pretrained_model_dir, "pytorch_model.bin")):
            pretrained_model_path = os.path.join(config.pretrained_model_dir, "pytorch_model.bin")
            model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=True)
        elif os.path.exists(os.path.join(config.pretrained_model_dir, "ds", "mp_rank_00_model_states.pt")):
            pretrained_model_path = os.path.join(config.pretrained_model_dir, "ds", "mp_rank_00_model_states.pt")
            model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu")["module"], strict=True)
        else:
            raise FileNotFoundError(f"Pretrained model not found in {config.pretrained_model_dir}")
        if is_rank_0():
            print(f"Loaded pretrained model from {pretrained_model_path}")
    return model


def load_data(
    tokenizer: object,
    filepath: str,
    config: argparse.Namespace = None,
    **kwargs,
):
    # Load data indices
    bin_filepath = f"{filepath}.bin"
    index_filepath = f"{filepath}.idx"
    indices = []
    with open(index_filepath, "r", encoding="utf-8") as f:
        for line in f:
            length, offset = map(int, line.strip().split())
            indices.append((length, offset))

    # Open file
    bin_f = open(bin_filepath, "rb")

    # Load data
    dataset = []
    for line_idx, (length, offset) in tqdm(
        enumerate(indices), desc=f"Loading data from {filepath}", disable=not is_rank_0()
    ):
        # Get dtype
        vocab_size = len(tokenizer)
        if vocab_size <= 2**16:
            dtype = np.dtype(np.uint16)
        elif vocab_size <= 2**32:
            dtype = np.dtype(np.uint32)
        else:
            dtype = np.dtype(np.uint64)
        itemsize = dtype.itemsize
        # Read
        mm = mmap.mmap(bin_f.fileno(), 0, access=mmap.ACCESS_READ)
        # Read the chunk
        num_bytes = length * itemsize
        chunk_bytes = mm[offset : offset + num_bytes]
        line = np.frombuffer(chunk_bytes, dtype=dtype)
        token_ids = line.tolist()
        # Construct
        segment_size = config.max_seq_len + 1
        num_segments = math.ceil(len(token_ids) / segment_size)
        for seg_idx in range(num_segments):
            start_idx = int(seg_idx * segment_size)
            end_idx = min((seg_idx + 1) * config.max_seq_len + 1, len(token_ids))
            segment_token_ids = token_ids[start_idx:end_idx]
            input_ids = segment_token_ids[:-1]
            labels = segment_token_ids[1:]
            doc_id = line_idx
            dataset.append(
                {
                    "text": tokenizer.decode(segment_token_ids),
                    "input_ids": input_ids,
                    "labels": labels,
                    "seg_id": seg_idx,
                    "doc_id": doc_id,
                }
            )
        # if len(dataset) >= 100:
        #     break
    bin_f.close()

    # Stats
    avg_segs_per_doc = len(dataset) / len(indices)
    print(f"Avg segments per document: {avg_segs_per_doc:.2f}")

    return dataset


def get_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--pretrained_config_path", type=str, required=True)
    parser.add_argument(
        "--override_attn_implementation",
        type=str,
        default=None,
        help="Override attention implementation for the model.",
    )
    return parser


def get_custom_arguments(parser: argparse.ArgumentParser):
    parser = get_model_arguments(parser)
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation.")
    parser.add_argument(
        "--disable_fw_update", action="store_true", help="Disable forward update of fwpkm during inference."
    )
    return parser


def compute_addr_stats(model, idcs: torch.LongTensor) -> dict[str, float]:
    addr_stats = {}
    idx_counter = torch.bincount(idcs.view(-1), minlength=model.config.fwpkm_n_subkeys**2)
    addr_stats["collision_ratio"] = idx_counter[idx_counter > 1].sum().item() / (idcs.numel() + 1e-6)
    addr_stats["coverage_ratio"] = (idx_counter > 0).sum().item() / model.config.fwpkm_n_subkeys**2
    addr_stats["kld"] = torch.distributions.kl.kl_divergence(
        torch.distributions.Categorical(probs=idx_counter.float() / (idx_counter.sum() + 1e-6)),
        torch.distributions.Categorical(probs=torch.ones_like(idx_counter).float() / idx_counter.numel()),
    ).item()
    return addr_stats


def main(config: argparse.Namespace, local_rank: int):
    # Print config
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # Initialize distributed
    torch.distributed.init_process_group()

    # Build model
    model = build_model(config)
    model = model.to(torch.bfloat16).to("cuda:0").eval()

    # Build tokenizer
    tokenizer = build_tokenizer(config)

    # Wandb
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            group=config.wandb_group_name,
            name=config.wandb_run_name,
            config=config,
        )

    # Build data
    for filepath in config.filepaths:
        dataset = load_data(tokenizer, filepath, config)
        filename = filepath[filepath.index("data/") + len("data/") :]

        # Init
        stats = []
        addr_stats = {}
        if config.model_type.startswith("qwen3_next_mem"):
            for layer_idx in range(len(model.config.fwpkm_layers)):
                addr_stats.update(
                    {
                        f"collision_ratio_list/{layer_idx}": [],
                        f"coverage_ratio_list/{layer_idx}": [],
                        f"kld_list/{layer_idx}": [],
                    }
                )
            accum_all_layer_fwpkm_idcs = [[] for _ in range(len(model.config.fwpkm_layers))]

        # Inference
        for segment_idx, segment in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            # Extract data
            if config.model_type.startswith("qwen3_next_mem"):
                from src.models.qwen3_next_mem import Qwen3NextMemDynamicCache

                past_key_values = Qwen3NextMemDynamicCache(model.config)
            elif config.model_type.startswith("lact"):
                from fla.models.utils import Cache

                past_key_values = Cache()

            input_ids = segment["input_ids"]
            labels = segment["labels"]
            seg_id = segment["seg_id"]
            doc_id = segment["doc_id"]
            position_ids = list(range(len(input_ids)))

            # Convert to tensor
            input_ids = torch.LongTensor([input_ids]).to(model.device)
            labels = torch.LongTensor([labels]).to(model.device)
            position_ids = torch.LongTensor([position_ids]).to(model.device)

            # Manually disable fwpkm update
            if config.disable_fw_update and hasattr(model, "adjust_fwpkm_update_chunksize"):
                model.adjust_fwpkm_update_chunksize(input_len=int(1e8))

            # Forward
            # # past_key_values is not None: linear attention and fwpkm use past_key_values
            # use_cahce=False: softmax attention doesn't use past_key_values
            model_outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                no_compile=config.no_compile,
                use_cache=False,
                return_dict=True,
                past_key_values=past_key_values,
            )
            logits = model_outputs["logits"]

            if config.disable_fw_update and hasattr(past_key_values, "reset_fwpkm_cache"):
                past_key_values.reset_fwpkm_cache()

            # Addressing stats.
            # Record every 512*512/8/4096=8 segments
            fwpkm_idcs = model_outputs.get("all_fwpkm_idcs", None)  # [B, T, num_fwpkm_layers, heads, topk]

            if fwpkm_idcs is not None:
                fwpkm_idcs = fwpkm_idcs.permute(2, 0, 1, 3, 4)  # [num_fwpkm_layers, B, T, heads, topk]
                for layer_idx in range(fwpkm_idcs.size(0)):
                    layer_fwpkm_idcs = fwpkm_idcs[layer_idx]
                    B, T, heads, topk = layer_fwpkm_idcs.size()
                    accum_all_layer_fwpkm_idcs[layer_idx].append(layer_fwpkm_idcs.view(B * T, heads * topk))
                    if (segment_idx + 1) % 8 == 0:
                        # Compute stats
                        layer_fwpkm_idcs = torch.cat(accum_all_layer_fwpkm_idcs[layer_idx], dim=0)  # [*, heads*topk]
                        layer_addr_stats = compute_addr_stats(model, layer_fwpkm_idcs)
                        for k, v in layer_addr_stats.items():
                            addr_stats[f"{k}_list/{layer_idx}"].append(v)
                        # Reset
                        accum_all_layer_fwpkm_idcs[layer_idx] = []

            # PPL stats
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            for position_id, loss in enumerate(losses.tolist()):
                stats.append(
                    {
                        "doc_id": doc_id,
                        "seg_id": seg_id,
                        "step": len(stats),
                        "position_id": position_id,
                        "loss": loss,
                    }
                )

        # Convert to dataframe
        stats_df = pd.DataFrame(stats)

        if config.wandb_project:
            # Total avg
            total_loss = stats_df["loss"].mean()
            total_ppl = math.exp(total_loss)
            wandb_run.summary.update(
                {
                    f"loss/{filename}": total_loss,
                    f"ppl/{filename}": total_ppl,
                }
            )

            # Addr stats
            if config.model_type.startswith("qwen3_next_mem"):
                for layer_idx in range(len(model.config.fwpkm_layers)):
                    collision_ratio_list = addr_stats[f"collision_ratio_list/{layer_idx}"]
                    coverage_ratio_list = addr_stats[f"coverage_ratio_list/{layer_idx}"]
                    kld_list = addr_stats[f"kld_list/{layer_idx}"]
                    wandb_run.summary.update(
                        {
                            f"collision_ratio/{filename}/layer{layer_idx}": np.mean(collision_ratio_list),
                            f"coverage_ratio/{filename}/layer{layer_idx}": np.mean(coverage_ratio_list),
                            f"kld/{filename}/layer{layer_idx}": np.mean(kld_list),
                        }
                    )

            # History
            wandb_run.define_metric(f"chunk_loss/{filename}", step_metric=f"chunk_step/{filename}")
            wandb_run.define_metric(f"chunk_pos_id/{filename}", step_metric=f"chunk_step/{filename}")
            wandb_run.define_metric(f"chunk_doc_id/{filename}", step_metric=f"chunk_step/{filename}")
            wandb_run.define_metric(f"chunk_seg_id/{filename}", step_metric=f"chunk_step/{filename}")
            # Process at doc level
            doc_boundaries = stats_df.groupby("doc_id")["step"].max().tolist()
            doc_boundaries = [0] + doc_boundaries[:10]
            for i in range(1, len(doc_boundaries)):
                rows = stats_df[(stats_df["step"] >= doc_boundaries[i - 1]) & (stats_df["step"] < doc_boundaries[i])]
                chunk_size = 128
                for start_idx in range(0, len(rows), chunk_size):
                    chunk_rows = rows.iloc[start_idx : start_idx + chunk_size]
                    chunk_doc_id = chunk_rows["doc_id"].iloc[-1]
                    chunk_seg_id = chunk_rows["seg_id"].iloc[-1]
                    chunk_pos_id = chunk_rows["position_id"].iloc[-1]
                    chunk_step = chunk_rows["step"].iloc[-1]
                    chunk_loss = chunk_rows["loss"].mean()
                    wandb.log(
                        {
                            f"chunk_step/{filename}": chunk_step,
                            f"chunk_loss/{filename}": chunk_loss,
                            f"chunk_pos_id/{filename}": chunk_pos_id,
                            f"chunk_doc_id/{filename}": chunk_doc_id,
                            f"chunk_seg_id/{filename}": chunk_seg_id,
                        }
                    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
