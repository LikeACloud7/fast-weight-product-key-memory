import math

from ike import (
    configargparse as argparse,
    get_inference_arguments,
)
from ike.util import build_tokenizer
from tqdm import tqdm
import torch
import wandb

from src.eval_ppl_stream import build_model, get_custom_arguments, load_data


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

        fwpkm_gates = []

        # Inference
        num_batches = math.ceil(len(dataset) / config.micro_batch_size)
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            batch_start = batch_idx * config.micro_batch_size
            batch_end = min((batch_idx + 1) * config.micro_batch_size, len(dataset))
            batch_segments = dataset[batch_start:batch_end]

            # Extract data
            input_ids = [segment["input_ids"] for segment in batch_segments]
            position_ids = [list(range(len(segment["input_ids"]))) for segment in batch_segments]

            # Convert to tensor
            input_ids = torch.LongTensor(input_ids).to(model.device)
            position_ids = torch.LongTensor(position_ids).to(model.device)

            # Forward
            model_outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                no_compile=config.no_compile,
                use_cache=False,
                return_dict=True,
                past_key_values=None,
            )

            batch_fwpkm_gates = model_outputs["all_fwpkm_gates"]  # [batch_size, seq_len, num_fwpkm_layers, 1]
            fwpkm_gates.append(batch_fwpkm_gates)

        fwpkm_gates = torch.cat(fwpkm_gates, dim=0)  # [data_size, seq_len, num_fwpkm_layers, 1]
        fwpkm_gates = fwpkm_gates.squeeze(-1).view(-1, fwpkm_gates.size(2))  # [total_tokens, num_fwpkm_layers]
        fwpkm_gates = fwpkm_gates.transpose(0, 1)  # [num_fwpkm_layers, total_tokens]
        fwpkm_gates = fwpkm_gates.float()

        if config.wandb_project:
            for layer_idx, layer_fwpkm_gates in enumerate(fwpkm_gates):
                layer_fwpkm_gate_histo = torch.histc(layer_fwpkm_gates, bins=10, min=0.0, max=1.0).cpu().numpy()
                wandb_run.summary.update(
                    {f"fwpkm_gate_histo/layer_{layer_idx}/{filename}": layer_fwpkm_gate_histo.tolist()}
                )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
