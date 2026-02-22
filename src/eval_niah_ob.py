import json
import os

from ike import (
    autoregressive_decode,
    configargparse as argparse,
    get_inference_arguments,
)
from ike.util import build_tokenizer
from tqdm import tqdm
import torch

from src.eval_niah import build_model, get_custom_arguments, load_niah_data
from src.models.qwen3_next_mem import Qwen3NextMemDynamicCache


def show_mem_slot(tokenizer, model, layer_idx, slot_idx):
    layer = model.model.layers[layer_idx]
    ob_idx2log = layer.fwpkm.ob_idx2log
    slot_log = ob_idx2log[slot_idx]
    slot_log = slot_log[-1:]
    slot_contents = []
    for entry in slot_log:
        q_id, v_id, v_prev_ctx_ids, v_next_ctx_ids, score = entry
        q = tokenizer.decode([q_id])
        v = tokenizer.decode([v_id])
        v_prev_ctx = tokenizer.decode(v_prev_ctx_ids)
        v_next_ctx = tokenizer.decode(v_next_ctx_ids)
        slot_contents.append(
            {
                "query": q,
                "value": v,
                "prev_context": v_prev_ctx,
                "next_context": v_next_ctx,
                "score": score,
            }
        )
    return slot_contents


def main(config: argparse.Namespace, local_rank: int):
    # Print config
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # Initialize distributed
    torch.distributed.init_process_group()

    # Build tokenizer
    tokenizer = build_tokenizer(config)

    for filepath in config.filepaths:
        filename = os.path.basename(filepath)

        # Build model
        model = build_model(config, filename)
        model = model.to(torch.bfloat16).to("cuda:0").eval()

        # Turn on observation mode
        for layer in model.model.layers:
            if layer.fwpkm is not None:
                layer.fwpkm.ob_mode = True

        # Build data
        dataset = load_niah_data(filepath, tokenizer)

        # Init cache
        past_key_values = Qwen3NextMemDynamicCache(model.config)

        for segment in tqdm(dataset, desc="Evaluating", disable=False):
            # Extract data
            haystack_ids = segment["haystack_ids"]
            input_ids = segment["input_ids"]
            candidates = [n["value"] for n in segment["needles"]]

            # Reset KV cache for softmax attentions for a new sample
            past_key_values.reset_kv_cache()

            # Pre forward
            pre_iter_input_ids = torch.LongTensor([haystack_ids]).to(model.device)
            for iter_idx in range(config.max_num_pre_iterations):
                # Update after the entire sequence
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
            model.adjust_fwpkm_update_chunksize(input_len=len(input_ids), past_key_values=past_key_values)

            # Forward
            def other_step_output_extract(step_outputs, batch_idx):
                all_fwpkm_idcs = step_outputs.all_fwpkm_idcs  # [batch_size, 1, num_fwpkm_layers, topk]
                all_fwpkm_gates = step_outputs.all_fwpkm_gates  # [batch_size, 1, num_fwpkm_layers, 1]
                all_fwpkm_scores = step_outputs.all_fwpkm_scores  # [batch_size, 1, num_fwpkm_layers, topk]
                num_fwpkm_layers = all_fwpkm_idcs.size(2)
                ret = {}
                for layer_idx in range(num_fwpkm_layers):
                    ret[f"fwpkm_idcs/{layer_idx}"] = all_fwpkm_idcs[batch_idx, 0, layer_idx].cpu().tolist()
                    ret[f"fwpkm_gates/{layer_idx}"] = all_fwpkm_gates[batch_idx, 0, layer_idx].item()
                    ret[f"fwpkm_scores/{layer_idx}"] = all_fwpkm_scores[batch_idx, 0, layer_idx].cpu().tolist()
                return ret

            model_outputs = autoregressive_decode(
                model=model,
                past_key_values=past_key_values,
                tokenizer=tokenizer,
                input_ids=[input_ids],
                min_new_tokens=6,
                max_new_tokens=6,
                do_sample=False,
                model_kwargs={"no_compile": True},
                other_step_output_extract_fn=other_step_output_extract,
            )
            output_id_seqs = model_outputs["output_id_seqs"]
            output_text = tokenizer.decode(output_id_seqs[0])
            label = segment["label"]
            correctness = output_text.strip() == label.strip()

            if correctness:
                output_data = []
                for output_idx in range(6):
                    output_id = output_id_seqs[0][output_idx]
                    output_token = tokenizer.decode([output_id])
                    label_token = label[output_idx]
                    # print(f"----- Step {output_idx} -----")
                    # print(f"Output Token: {output_token}")
                    token_data = []
                    for fwpkm_layer_idx in range(len(model.config.fwpkm_layers)):
                        layer_idx = model.config.fwpkm_layers[fwpkm_layer_idx]
                        # print(f"\tfwpkm Layer {layer_idx}:")
                        fwpkm_idcs = model_outputs[f"fwpkm_idcs/{fwpkm_layer_idx}"][0][output_idx][0]
                        fwpkm_scores = model_outputs[f"fwpkm_scores/{fwpkm_layer_idx}"][0][output_idx][0]
                        fwpkm_layer_data = []
                        for fwpkm_slot_idx, slot_score in zip(fwpkm_idcs, fwpkm_scores):
                            # print(f"\t\tSlot {fwpkm_slot_idx}:")
                            slot_content = show_mem_slot(tokenizer, model, layer_idx, fwpkm_slot_idx)
                            fwpkm_layer_data.append(
                                {"slot_idx": fwpkm_slot_idx, "slot_score": slot_score, "content": slot_content}
                            )
                        token_data.append({"layer_idx": layer_idx, "slots": fwpkm_layer_data})
                    output_data.append(
                        {"output_token": output_token, "label_token": label_token, "fwpkm_layers": token_data}
                    )
                sample_data = {
                    "needles": segment["needles"],
                    "question_key": segment["question_key"],
                    "reference_answer": label,
                    "predicted_answer": output_text,
                    "fwpkm_data": output_data,
                }

                output_filepath = os.path.join(config.output_dir, f"ob_data.{filename}.json")
                os.makedirs(config.output_dir, exist_ok=True)
                with open(output_filepath, "w") as f:
                    json.dump(sample_data, f, indent=4)

                exit(0)


if __name__ == "__main__":
    config = get_inference_arguments(get_custom_arguments)
    with torch.no_grad():
        main(config, config.local_rank)
