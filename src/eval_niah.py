import functools
import json
from multiprocessing import Pool
import os

from ike import (
    autoregressive_decode,
    configargparse as argparse,
    get_inference_arguments,
    is_rank_0,
)
from ike.util import build_tokenizer
from tqdm import tqdm
import torch
import wandb


def build_model(config: argparse.Namespace, filename: str, **kwargs):
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
    # Set RoPE scaling to max length according to filename (base model uses 4K length)
    if "128k" in filename:
        model_config.rope_scaling = {"type": "yarn", "factor": 64}
    elif "32k" in filename:
        model_config.rope_scaling = {"type": "yarn", "factor": 16}
    elif "16k" in filename:
        model_config.rope_scaling = {"type": "yarn", "factor": 8}
    elif "8k" in filename:
        model_config.rope_scaling = {"type": "yarn", "factor": 4}

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


def load_niah_line(line_str, tokenizer):
    """
    Worker function to process a single JSON line.
    """
    line = json.loads(line_str)

    haystack = line["haystack"]
    question_key = line["question_key"]
    answer = line["answer"]
    needles = line["all_inserted_needles"]
    target_depth = line["metadata"]["target_depth"]
    question = f"The secret number for {question_key} is "
    input_text = haystack + " " + question

    haystack_ids = tokenizer.encode(haystack, add_special_tokens=False)
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)

    return {
        "haystack_ids": haystack_ids,
        "input_ids": input_ids,
        "label": answer,
        "needles": needles,
        "target_depth": target_depth,
        "question_key": question_key,
    }


def load_niah_data(filepath: str, tokenizer: object, num_workers: int = 10):
    raw_lines = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            raw_lines.append(line)

    worker_func = functools.partial(load_niah_line, tokenizer=tokenizer)

    dataset = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap(worker_func, raw_lines), total=len(raw_lines), desc=f"Parallel loading from {filepath}"
        ):
            dataset.append(result)

    return dataset


def get_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--pretrained_config_path", type=str, required=True)
    parser.add_argument("--keep_chunk_size", type=int, help="Stick to training fwpkm chunk size during inference.")
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
    parser.add_argument("--max_num_pre_iterations", type=int, default=3)
    return parser


def main(config: argparse.Namespace, local_rank: int):
    # Print config
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # Initialize distributed
    torch.distributed.init_process_group()

    # Build tokenizer
    tokenizer = build_tokenizer(config)

    for num_pre_iterations in range(0, config.max_num_pre_iterations + 1):
        # Wandb
        if config.wandb_project:
            run_name = f"{config.wandb_run_name}/iter{num_pre_iterations}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                group=config.wandb_group_name,
                name=run_name,
                config=config,
            )

        for filepath in config.filepaths:
            filename = os.path.basename(filepath)

            # Build model
            model = build_model(config, filename)
            model = model.to(torch.bfloat16).to("cuda:0").eval()

            # Build data
            dataset = load_niah_data(filepath, tokenizer)

            # Init cache
            if config.model_type.startswith("qwen3_next_mem"):
                from src.models.qwen3_next_mem import Qwen3NextMemDynamicCache

                past_key_values = Qwen3NextMemDynamicCache(model.config)
            elif config.model_type.startswith("lact"):
                from fla.models.utils import Cache

                past_key_values = Cache()

            # Inference
            stats = {
                "num_exact_match": 0,
                "num_digit_match": 0,
                "num_cand_match": 0,
                "em_history": [],
            }
            for segment in tqdm(dataset, desc="Evaluating", disable=False):
                # Extract data
                haystack_ids = segment["haystack_ids"]
                input_ids = segment["input_ids"]
                candidates = [n["value"] for n in segment["needles"]]

                # Reset KV cache for softmax attentions for a new sample
                if hasattr(past_key_values, "reset_kv_cache"):
                    past_key_values.reset_kv_cache()

                # Pre forward
                pre_iter_input_ids = torch.LongTensor([haystack_ids]).to(model.device)
                for iter_idx in range(num_pre_iterations):
                    # Update after the entire sequence
                    if not config.keep_chunk_size and hasattr(model, "adjust_fwpkm_update_chunksize"):
                        model.adjust_fwpkm_update_chunksize(
                            input_len=pre_iter_input_ids.size(1), past_key_values=past_key_values
                        )
                    # Forward
                    model_outputs = model(
                        input_ids=pre_iter_input_ids,
                        no_compile=config.no_compile,
                        use_cache=False,
                        return_dict=True,
                        past_key_values=past_key_values,
                    )

                # Update after the entire sequence
                if not config.keep_chunk_size and hasattr(model, "adjust_fwpkm_update_chunksize"):
                    model.adjust_fwpkm_update_chunksize(input_len=len(input_ids), past_key_values=past_key_values)
                # Forward
                model_outputs = autoregressive_decode(
                    model=model,
                    past_key_values=past_key_values,
                    tokenizer=tokenizer,
                    input_ids=[input_ids],
                    min_new_tokens=6,
                    max_new_tokens=6,
                    do_sample=False,
                    model_kwargs={"no_compile": True},
                )
                output_id_seqs = model_outputs["output_id_seqs"]
                output_text = tokenizer.decode(output_id_seqs[0])
                label = segment["label"]

                # print([output_text], [label], candidates)
                correctness = output_text.strip() == label.strip()

                stats["num_exact_match"] += int(correctness)
                stats["num_digit_match"] += sum(1 for o, l in zip(output_text.strip(), label.strip()) if o == l)
                stats["num_cand_match"] += int(output_text in candidates)
                stats["em_history"].append(int(correctness))

            for k, v in stats.items():
                print(f"{k}: {v}")

            if config.wandb_project:
                # Metrics
                wandb_run.summary.update({f"exact_match/{filename}": stats["num_exact_match"]})
                wandb_run.summary.update({f"digit_match/{filename}": stats["num_digit_match"]})
                wandb_run.summary.update({f"cand_match/{filename}": stats["num_cand_match"]})
                # EM history
                wandb_run.define_metric(f"em_progress/{filename}", step_metric=f"samples_seen/{filename}")
                accum_em = 0
                for idx, em in enumerate(stats["em_history"]):
                    accum_em += em
                    wandb.log(
                        {
                            f"samples_seen/{filename}": idx + 1,
                            f"em_progress/{filename}": accum_em,
                        }
                    )

        if config.wandb_project:
            wandb_run.finish()


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
