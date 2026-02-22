import functools
import math
import mmap
import multiprocessing as mp
import os

from ike import (
    configargparse as argparse,
    get_inference_arguments,
    is_rank_0,
)
from ike.util import build_tokenizer
import numpy as np
from tqdm import tqdm
import torch
import wandb

from src.eval_ppl_stream import build_model, get_custom_arguments


def load_line(index, bin_filepath, tokenizer):
    bin_f = open(bin_filepath, "rb")
    length, offset = index
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
    bin_f.close()
    return token_ids


def load_data(
    tokenizer: object,
    filepath: str,
    config: argparse.Namespace = None,
    num_workers=10,
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
    worker_func = functools.partial(load_line, bin_filepath=bin_filepath, tokenizer=tokenizer)

    dataset = []
    with mp.Pool(processes=num_workers) as pool:
        for token_ids in tqdm(
            pool.imap(worker_func, indices), total=len(indices), desc=f"Parallel loading from {filepath}"
        ):
            # Construct
            segment_size = config.max_seq_len + 1
            num_segments = math.ceil(len(token_ids) / segment_size)
            for seg_idx in range(num_segments):
                start_idx = int(seg_idx * segment_size)
                end_idx = min((seg_idx + 1) * config.max_seq_len + 1, len(token_ids))
                segment_token_ids = token_ids[start_idx:end_idx]
                input_ids = segment_token_ids[:-1]
                labels = segment_token_ids[1:]
                dataset.append(
                    {
                        "text": tokenizer.decode(segment_token_ids),
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )
            # if len(dataset) >= 100:
            #     break

    # Stats
    avg_segs_per_doc = len(dataset) / len(indices)
    print(f"Avg segments per document: {avg_segs_per_doc:.2f}")

    return dataset


def eval(model, domain2valid_data, config: argparse.Namespace):
    # Stop fast weight updates
    if hasattr(model, "adjust_fwpkm_update_chunksize"):
        model.adjust_fwpkm_update_chunksize(input_len=int(1e8))

    # Provide past key values other wise will use forward_wo_past where update always happens
    if config.model_type.startswith("qwen3_next_mem"):
        from src.models.qwen3_next_mem import Qwen3NextMemDynamicCache

        past_key_values = Qwen3NextMemDynamicCache(model.config)
    elif config.model_type.startswith("lact"):
        from fla.models.utils import Cache

        past_key_values = Cache()

    # Run over domain datasets
    domain2ppl = {}
    for domain, data in domain2valid_data.items():
        num_batches = math.ceil(len(data) / config.micro_batch_size)
        domain_losses = []
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating domain {domain}"):
            batch_data = data[batch_idx * config.micro_batch_size : (batch_idx + 1) * config.micro_batch_size]
            # Prepare batch
            input_ids = []
            labels = []
            for item in batch_data:
                input_ids.append(item["input_ids"])
                labels.append(item["labels"])

            # LaCT requires consistent batch size for caching, so we skip the last batch if it's not full
            if len(batch_data) != config.micro_batch_size:
                continue

            # Tensor
            input_ids = torch.LongTensor(input_ids).to(model.device)
            labels = torch.LongTensor(labels).to(model.device)
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=model.device).unsqueeze(0).expand_as(input_ids)
            )
            # Forward
            model_outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                no_compile=config.no_compile,
                use_cache=False,
                return_dict=True,
                past_key_values=past_key_values,
            )
            logits = model_outputs["logits"]
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            domain_losses.append(losses.detach())

            # Clear past key values
            if hasattr(past_key_values, "reset_fwpkm_cache"):
                past_key_values.reset_fwpkm_cache()

        domain_losses = torch.cat(domain_losses, dim=0)
        domain_loss = domain_losses.mean()
        domain_ppl = domain_loss.exp().item()
        domain2ppl[domain] = domain_ppl
    return domain2ppl


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
    domains = [
        "dm_mathematics",
        "freelaw",
        "philpapers",
        "pubmed_central",
        "ubuntu_irc",
        "uspto_backgrounds",
    ]
    domain2train_data = {}
    domain2valid_data = {}
    for domain in domains:
        filepath = f"data/pile_domain/encoded/mistral32k/l4096/{domain}.8mt"
        dataset = load_data(tokenizer, filepath, config)
        num_samples = len(dataset)
        split_idx = int(0.5 * num_samples)
        domain2train_data[domain] = dataset[:split_idx]
        domain2valid_data[domain] = dataset[split_idx:]

    # Results
    domain2ppl_list = []

    # Eval before test-time training
    domain2ppl = eval(model, domain2valid_data, config)
    domain2ppl_list.append(domain2ppl)

    # Test-time training on each domain
    for domain in domains:
        domain_train_data = domain2train_data[domain]

        if hasattr(model, "adjust_fwpkm_update_chunksize"):
            model.adjust_fwpkm_update_chunksize(input_len=512)

        num_batches = math.ceil(len(domain_train_data) / config.micro_batch_size)
        for batch_idx in tqdm(range(num_batches), desc=f"TTT on {domain}"):
            batch_data = domain_train_data[
                batch_idx * config.micro_batch_size : (batch_idx + 1) * config.micro_batch_size
            ]
            if len(batch_data) != config.micro_batch_size:
                continue
            # Prepare batch
            input_ids = [item["input_ids"] for item in batch_data]
            # Tensor
            input_ids = torch.LongTensor(input_ids).to(model.device)
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=model.device).unsqueeze(0).expand_as(input_ids)
            )

            # Forward
            _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                no_compile=config.no_compile,
                use_cache=False,
                return_dict=True,
            )

        # Eval after domain TTT
        domain2ppl = eval(model, domain2valid_data, config)
        domain2ppl_list.append(domain2ppl)

    if config.wandb_project:
        ppl_list = [[domain2ppl[domain] for domain in domains] for domain2ppl in domain2ppl_list]
        wandb_run.summary.update({"ppl_list": ppl_list, "domains": domains})
        wandb.finish()


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
