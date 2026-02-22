from collections import Counter, defaultdict
import functools
from multiprocessing import Pool
import os

from datasets import load_dataset
from ike import (
    configargparse as argparse,
    get_inference_arguments,
    is_rank_0,
)
from ike.util import build_tokenizer
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
    if "fwmlp" in config.pretrained_config_path:
        config.keep_chunk_size = True  # Otherwise fw states become nan
    # Set RoPE scaling (base model uses 4K length)
    if config.rope_scaling_type:
        model_config.rope_scaling = {"type": config.rope_scaling_type, "factor": config.rope_scaling_factor}

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


def task2prompt(task, context, question):
    if task == "narrativeqa":
        context_prompt = f"You are given a story, which can be either a novel or a movie script, and a question.\nAnswer the question as concisely as you can, using a single phrase if possible. Do not provide any  explanation.\nStory:\n{context}\n"
        question_prompt = f"Now, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\nQuestion:\n{question}\nAnswer:\n"
    elif task == "multifieldqa_en":
        context_prompt = f"Read the following text and answer briefly.\n{context}\n"
        question_prompt = f"Now, answer the following question based on the above text, only give me the answer and do not output any other words.\nQuestion:\n{question}\nAnswer:\n"
    elif task == "hotpotqa":
        context_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\nThe following are given passages.\n{context}\n"
        question_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion:\n{question}\nAnswer:\n"
    elif task == "2wikimqa":
        context_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\nThe following are given passages.\n{context}\n"
        question_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion:\n{question}\nAnswer:\n"
    elif task == "musique":
        context_prompt = f"Answer the question based on the given context. Only give me the answer and do not output any other words.\nContext:\n{context}\n"
        question_prompt = f"Answer the question based on the given context. Only give me the answer and do not output any other words.\nQuestion:\n{question}\nAnswer:\n"
    elif task == "triviaqa":
        context_prompt = f"Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n{context}\n"
        question_prompt = f"{question}"
    else:
        raise ValueError(f"Unknown task: {task}")
    return context_prompt, question_prompt


def load_line(line, tokenizer, metric):
    question = line["input"]
    context = line["context"]
    answers = line["answers"]
    length = line["length"]
    dataset = line["dataset"]
    language = line["language"]

    # Filter: only certain metrics
    if metric not in ["f1"]:
        return None
    # Filter: use only English samples
    if language != "en":
        return None
    # Filter: # character > 64K
    if length > 65536:
        return None

    context, question = task2prompt(dataset, context, question)

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    answer_ids = [tokenizer.encode(ans, add_special_tokens=False) for ans in answers]
    max_answer_id_len = max(len(aid) for aid in answer_ids)

    # Filter: # tokens > 128K
    if len(context_ids) + len(question_ids) + max_answer_id_len > 131072:
        return None

    return {
        "context_ids": context_ids,
        "question_ids": question_ids,
        "answers": answers,
        "max_answer_id_len": max_answer_id_len,
        "task": dataset,
        "metric": metric,
    }


def clean_output(output: str):
    # Truncate at the first newline
    if "\n" in output:
        output = output.split("\n")[0]
    # Strip leading and trailing spaces
    output = output.strip()
    return output


def compute_metric(hyp: str, refs: list[str], metric: str):
    if metric == "f1":
        hyp_words = hyp.split()
        f1_scores = []
        for ref in refs:
            ref_words = ref.split()
            common = Counter(hyp_words) & Counter(ref_words)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                precision = 1.0 * num_same / len(hyp)
                recall = 1.0 * num_same / len(ref)
                f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def load_longbench_data(tokenizer: object, num_workers: int = 10):
    tasks = [
        ("narrativeqa", "f1"),
        ("multifieldqa_en", "f1"),
        ("hotpotqa", "f1"),
        ("2wikimqa", "f1"),
        ("musique", "f1"),
    ]

    dataset = []
    for task, metric in tasks:
        task_data = load_dataset("THUDM/LongBench", task, split="test")

        worker_func = functools.partial(load_line, tokenizer=tokenizer, metric=metric)

        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap(worker_func, task_data), total=len(task_data), desc=f"Parallel loading from {task}"
            ):
                if result is not None:
                    dataset.append(result)
                # if len(dataset) >= 100:
                #     break

    # Stats
    task2stats = defaultdict(lambda: {"num_samples": 0, "avg_length": 0, "max_length": 0})
    task2dataset = defaultdict(list)
    for segment in dataset:
        task = segment["task"]
        context_len = len(segment["context_ids"])
        question_len = len(segment["question_ids"])
        answer_len = segment["max_answer_id_len"]
        total_len = context_len + question_len + answer_len
        task2stats[task]["num_samples"] += 1
        task2stats[task]["avg_length"] += total_len
        task2stats[task]["max_length"] = max(task2stats[task]["max_length"], total_len)
        task2stats["total"]["num_samples"] = len(dataset)
        task2stats["total"]["avg_length"] += total_len
        task2stats["total"]["max_length"] = max(task2stats["total"]["max_length"], total_len)
        task2dataset[task].append(segment)
    for task, stats in task2stats.items():
        stats["avg_length"] /= stats["num_samples"]
        print(
            f"Task: {task}, Num samples: {stats['num_samples']}, Avg length: {stats['avg_length']}, Max length: {stats['max_length']}"
        )

    return task2dataset


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
    parser.add_argument("--rope_scaling_type", type=str, default=None, help="Type of RoPE scaling to use.")
    parser.add_argument("--rope_scaling_factor", type=int, default=64, help="Factor for RoPE scaling.")
    return parser


def main(config: argparse.Namespace, local_rank: int):
    # Print config
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # Initialize distributed
    torch.distributed.init_process_group()

    # Build tokenizer
    tokenizer = build_tokenizer(config)

    # Build data
    task2dataset = load_longbench_data(tokenizer)
    # dataset = dataset[:200]

    # Wandb
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            group=config.wandb_group_name,
            name=config.wandb_run_name,
            config=config,
        )

    for task, dataset in task2dataset.items():
        # Build model
        model = build_model(config)
        model = model.to(torch.bfloat16).to("cuda:0").eval()

        # Stats
        score_list = []

        # Init cache
        if config.model_type.startswith("qwen3_next_mem"):
            from src.models.qwen3_next_mem import Qwen3NextMemDynamicCache

            past_key_values = Qwen3NextMemDynamicCache(model.config)
        elif config.model_type.startswith("lact"):
            from fla.models.utils import Cache

            past_key_values = Cache()

        # Evaluate
        for segment in tqdm(dataset, desc="Evaluating", disable=False):
            # Extract data
            context_ids = segment["context_ids"]
            question_ids = segment["question_ids"]
            answers = segment["answers"]
            max_answer_id_len = segment["max_answer_id_len"]
            task = segment["task"]
            metric = segment["metric"]

            # Reset KV cache for softmax attentions for a new sample
            if hasattr(past_key_values, "reset_kv_cache"):
                past_key_values.reset_kv_cache()

            # Reset FwPKM cache for a new sample
            if hasattr(past_key_values, "reset_fwpkm_cache"):
                past_key_values.reset_fwpkm_cache()

            ##### PPL ######
            # Update after the entire context
            if not config.keep_chunk_size and hasattr(model, "adjust_fwpkm_update_chunksize"):
                model.adjust_fwpkm_update_chunksize(input_len=len(context_ids), past_key_values=past_key_values)
            answer_ids = tokenizer.encode(answers[0], add_special_tokens=False)
            input_ids = torch.LongTensor(context_ids + question_ids + answer_ids[:-1]).unsqueeze(0).to(model.device)
            labels = torch.LongTensor(answer_ids).unsqueeze(0).to(model.device)
            model_outputs = model(
                input_ids=input_ids,
                no_compile=config.no_compile,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            logits = model_outputs["logits"][:, -len(answer_ids) :, :]
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            score_list.append(losses.detach())

        losses = torch.cat(score_list, dim=0)
        loss = losses.mean()
        ppl = torch.exp(loss).item()
        print(f"Task: {task}, PPL: {ppl} ({len(losses)} samples)")

        if config.wandb_project:
            wandb_run.summary.update({f"{task}_ppl": ppl})

    if config.wandb_project:
        wandb_run.finish()


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
