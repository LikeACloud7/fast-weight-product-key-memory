import logging
import mmap
import random

from ike import (
    TrainingPipeline,
    configargparse as argparse,
    get_arguments,
    is_rank_0,
)
import numpy as np
import torch

from src.data import (
    load_data_from_np_idx,
    DATA_PROCESSOR_CLASSES,
)


logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True


def build_optimizer(
    model,
    config: argparse.Namespace,
    **kwargs,
):
    def get_optimizer_grouped_parameters(module):
        param_optimizer = list(module.named_parameters())
        no_decay = ["bias", "ln", "norm"]

        parameters_with_decay = []
        parameters_without_decay = []
        for n, p in param_optimizer:
            if not p.requires_grad:
                continue
            if "fwpkm" in n:  # Ignore fast weights, which will be updated in "inner loop"
                continue
            elif any(nd in n for nd in no_decay):
                parameters_without_decay.append((n, p))
            else:
                parameters_with_decay.append((n, p))

        for name, _ in parameters_with_decay:
            logger.info(f"Optimizer group with weight decay: {name}")
        for name, _ in parameters_without_decay:
            logger.info(f"Optimizer group without weight decay: {name}")

        optimizer_grouped_parameters = []
        if len(parameters_with_decay) > 0:
            optimizer_grouped_parameters.append(
                {"params": [p for n, p in parameters_with_decay], "weight_decay": config.weight_decay}
            )
        if len(parameters_without_decay) > 0:
            optimizer_grouped_parameters.append(
                {"params": [p for n, p in parameters_without_decay], "weight_decay": 0.0}
            )
        return optimizer_grouped_parameters

    if config.optimizer_type == "adam":
        from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

        AdamOptimizer = DeepSpeedCPUAdam if config.offload_adam else FusedAdam
        optimizer = AdamOptimizer(
            get_optimizer_grouped_parameters(model),
            lr=config.peak_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    else:
        raise Exception(f"Unknown optimizer type: {config.optimizer_type}")
    return optimizer


def build_model(
    config: argparse.Namespace,
    **kwargs,
):
    if config.model_type == "qwen3_next_mem":
        from src.models.qwen3_next_mem import Qwen3NextMemConfig, Qwen3NextMemForCausalLM

        model_config = Qwen3NextMemConfig.from_pretrained(config.pretrained_config_path)
        if config.override_attn_implementation:
            model_config._attn_implementation = config.override_attn_implementation
        model = Qwen3NextMemForCausalLM(model_config)
    elif config.model_type == "lact":
        from src.models.lact_hf_rope import LaCTSWIGLUConfig, LaCTForCausalLM

        model_config = LaCTSWIGLUConfig.from_pretrained(config.pretrained_config_path)
        if config.override_attn_implementation:
            model_config._attn_implementation = config.override_attn_implementation
        model = LaCTForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    if is_rank_0():
        logger.info(model)
        logger.info(model.config)

    return model


def compute_metrics(
    data: dict,
    **kwargs,
):
    ret_data, ret_stat = {}, {}

    num_valid_losses_list = data["num_valid_losses"]
    nlls_list = data["nlls"]
    nlls = []
    for item in nlls_list:
        nlls.append(item.view(-1, 1))
    nlls = torch.cat(nlls, dim=0)
    num_valid_losses = sum(num_valid_losses_list)
    mean_nll = nlls.sum() / num_valid_losses
    ppl = mean_nll.exp()
    ret_stat["ppl"] = ppl.item()

    return ret_data, ret_stat


def train_forward_step(
    step: int,
    model: object,
    tokenizer: object,
    batch_data: list,
    config: argparse.Namespace,
    is_training: bool = True,
    **kwargs,
):
    ret_data = {}
    ret_stat = {}

    # Inputs
    batch_input_ids = []
    batch_labels = []
    # Data only provides index
    # Get dtype
    vocab_size = len(tokenizer)
    if vocab_size <= 2**16:
        dtype = np.dtype(np.uint16)
    elif vocab_size <= 2**32:
        dtype = np.dtype(np.uint32)
    else:
        dtype = np.dtype(np.uint64)
    itemsize = dtype.itemsize
    # Read data
    filename2f = {}
    filename2mm = {}
    for item in batch_data:
        length = item["length"]
        offset = item["offset"]
        filepath = item["filepath"]
        if filepath not in filename2f:
            f = open(filepath, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            filename2f[filepath] = f
            filename2mm[filepath] = mm
        else:
            mm = filename2mm[filepath]
        # Read the chunk
        num_bytes = length * itemsize
        chunk_bytes = mm[offset : offset + num_bytes]
        line = np.frombuffer(chunk_bytes, dtype=dtype)
        token_ids = line.tolist()
        # Construct input_ids, labels
        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        # Append to batch
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
    # Close files
    for f in filename2f.values():
        f.close()
    for mm in filename2mm.values():
        mm.close()

    # Padding
    max_seq_len = max(len(seq) for seq in batch_input_ids)
    batch_input_ids = [[tokenizer.pad_token_id] * (max_seq_len - len(seq)) + seq for seq in batch_input_ids]
    batch_labels = [[-100] * (max_seq_len - len(seq)) + seq for seq in batch_labels]
    batch_position_ids = [[0] * (max_seq_len - len(seq)) + list(range(len(seq))) for seq in batch_input_ids]

    # Convert to tensor
    batch_input_ids = torch.LongTensor(batch_input_ids).to(model.device)
    batch_labels = torch.LongTensor(batch_labels).to(model.device)
    batch_position_ids = torch.LongTensor(batch_position_ids).to(model.device)

    # Construct attention mask
    batch_attn_mask = (batch_input_ids != tokenizer.pad_token_id).long().to(model.device)

    # Number of valid loss positions
    num_valid_losses = (batch_labels != -100).sum().item()

    # Temporarily change FA to SWA
    if is_training and config.fa2swa_prob:
        run_fa2swa = random.random() < config.fa2swa_prob
        # Synchronize across ranks
        run_fa2swa_tensor = torch.tensor(run_fa2swa, dtype=torch.bool, device=model.device)
        torch.distributed.broadcast(run_fa2swa_tensor, src=0)
        run_fa2swa = run_fa2swa_tensor.item()
        if run_fa2swa:
            for layer in model.module.model.layers:
                if layer.layer_type == "full_attention":
                    layer.self_attn.sliding_window = model.module.config.fwpkm_update_chunk_size

    # Forward
    model_outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        position_ids=batch_position_ids,
        no_compile=True if config.no_compile or not is_training else False,
        use_cache=False,
        return_dict=True,
    )

    # Revert SWA to FA
    if is_training and config.fa2swa_prob:
        if run_fa2swa:
            for layer in model.module.model.layers:
                if layer.layer_type == "full_attention":
                    layer.self_attn.sliding_window = None

    # Backwrad
    if config.model_type == "qwen3_next_mem":
        batch_size, seq_len, vocab_size = model_outputs["logits"].size()
        logits = model_outputs["logits"]
        fwpkm_losses = model_outputs.get(
            "all_fwpkm_losses", None
        )  # [batch_size, seq_len, num_fwpkm_layers, hidden_size]
        fwpkm_addr_stats = model_outputs.get("all_fwpkm_addr_stats", None)  # num_fwpkm_layers x dict
        fwpkm_grad_norms = model_outputs.get("all_fwpkm_grad_norms", None)  # num_fwpkm_layers * [num_fw]
        pkm_idcs = model_outputs.get("all_pkm_idcs", None)  # [batch_size, seq_len, num_pkm_layers, knn]
        fwpkm_gates = model_outputs.get("all_fwpkm_gates", None)  # [batch_size, seq_len, num_fwpkm_layers, hidden_size]

        if fwpkm_losses is not None:
            # [batch_size, seq_len, num_pkm_layers, hidden_size]
            for layer in range(fwpkm_losses.size(2)):
                with torch.no_grad():
                    mean_layer_fwpkm_loss = fwpkm_losses[:, :, layer].contiguous().mean()
                ret_stat[f"fwpkm_loss/{layer}"] = mean_layer_fwpkm_loss.item()

        if pkm_idcs is not None:
            for layer in range(pkm_idcs.size(2)):
                with torch.no_grad():
                    layer_pkm_idcs = pkm_idcs[:, :, layer, :].contiguous().view(-1)
                    unique_pkm_idx_count = torch.unique(layer_pkm_idcs).numel()
                    pkm_coverage_ratio = unique_pkm_idx_count / model.module.config.pkm_n_subkeys**2
                ret_stat[f"pkm_coverage_ratio/{layer}"] = pkm_coverage_ratio

        if fwpkm_addr_stats is not None:
            for layer in range(len(fwpkm_addr_stats)):
                with torch.no_grad():
                    layer_fwpkm_addr_stats = fwpkm_addr_stats[layer]
                    for k, v in layer_fwpkm_addr_stats.items():
                        ret_stat[f"fwpkm_addressing_{k}/{layer}"] = v

        if fwpkm_grad_norms is not None:
            for layer_idx, layer in enumerate(model.module.config.fwpkm_layers):
                fwpkm_layer = model.module.model.layers[layer].fwpkm
                i = 0
                for n, p in fwpkm_layer.get_fw_named_params().items():
                    grad_norm = fwpkm_grad_norms[layer_idx][i]
                    ret_stat[f"fwpkm_grad_norm/{layer}/{n}"] = grad_norm.item()
                    i += 1

        if fwpkm_gates is not None:
            for layer in range(fwpkm_gates.size(2)):
                with torch.no_grad():
                    layer_fwpkm_gate = fwpkm_gates[:, :, layer].contiguous()
                    mean_layer_fwpkm_gate = layer_fwpkm_gate.mean()
                ret_stat[f"fwpkm_gate/{layer}"] = mean_layer_fwpkm_gate.item()
                # if not is_training:
                layer_fwpkm_gate = layer_fwpkm_gate.float()
                # Add a dummy batch dim to comply with batch-averaging in statistics tracker
                layer_fwpkm_gate_histo = torch.histc(layer_fwpkm_gate, bins=10, min=0.0, max=1.0).view(1, -1)
                layer_fwpkm_gate_histo /= layer_fwpkm_gate_histo.sum()
                layer_fwpkm_gate_histo = layer_fwpkm_gate_histo.cpu().numpy().tolist()
                ret_stat[f"fwpkm_gate_histo/{layer}"] = layer_fwpkm_gate_histo

        logits = logits.float()
        if is_training:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), batch_labels.view(-1), ignore_index=-100, reduction="mean"
            )

            with torch.no_grad():
                ppl = loss.exp()

            ret_stat["ppl"] = ppl.item()
            ret_stat["loss"] = loss.item()
        else:
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), batch_labels.view(-1), ignore_index=-100, reduction="none"
            ).view(batch_size, seq_len)
            loss = losses.sum() / num_valid_losses

            ret_stat["loss"] = loss.item()
            ret_data["nlls"] = losses.detach().cpu()
            ret_data["num_valid_losses"] = num_valid_losses
    else:
        batch_size, seq_len, vocab_size = model_outputs["logits"].size()
        logits = model_outputs["logits"]
        logits = logits.float()
        if is_training:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), batch_labels.view(-1), ignore_index=-100, reduction="mean"
            )
            with torch.no_grad():
                ppl = loss.exp()
            ret_stat["loss"] = loss.item()
            ret_stat["ppl"] = ppl.item()
        else:
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), batch_labels.view(-1), ignore_index=-100, reduction="none"
            ).view(batch_size, seq_len)
            loss = losses.sum() / num_valid_losses
            ret_stat["loss"] = loss.item()
            ret_data["nlls"] = losses.detach().cpu()
            ret_data["num_valid_losses"] = num_valid_losses

    if hasattr(model, "stats"):
        for key, value in model.stats.items():
            ret_stat[key] = value

    return loss, ret_data, ret_stat


def valid_forward_step(
    step: int,
    model: object,
    tokenizer: object,
    batch_data: list,
    config: argparse.Namespace,
    **kwargs,
):
    _, ret_data, ret_stat = train_forward_step(
        step=step,
        model=model,
        tokenizer=tokenizer,
        batch_data=batch_data,
        config=config,
        is_training=False,
        **kwargs,
    )
    return ret_data, ret_stat


def get_custom_arguments(parser: argparse.ArgumentParser):
    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen3_next_mem",
        help="Type of model to use.",
        choices=["qwen3_next_mem", "lact"],
    )
    parser.add_argument("--pretrained_config_path", type=str, help="Path to the pretrained config file.")
    parser.add_argument(
        "--override_attn_implementation",
        type=str,
        default=None,
        help="Override attention implementation for the model.",
    )
    parser.add_argument(
        "--fa2swa_prob", type=float, default=0.0, help="Probability of switching FA to SWA during training."
    )

    # Optimization
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation.")

    return parser


def main(config: argparse.Namespace, local_rank: int):
    assert len(config.data_processor_types) == len(config.train_filepaths), (
        "Number of data processor types must match the number of filepaths."
    )
    data_processor_classes = [
        DATA_PROCESSOR_CLASSES[data_processor_class_type] for data_processor_class_type in config.data_processor_types
    ]

    if config.valid_data_processor_types:
        valid_data_processor_classes = [
            DATA_PROCESSOR_CLASSES[data_processor_class_type]
            for data_processor_class_type in config.valid_data_processor_types
        ]
    else:
        valid_data_processor_classes = [data_processor_classes[0]] * len(config.valid_filepaths)

    training_pipeline = TrainingPipeline(
        config=config,
        world_size=config.world_size,
        local_rank=local_rank,
        global_rank=config.global_rank,
    )

    training_pipeline.run(
        build_model_fn=build_model,
        build_optimizer_fn=build_optimizer,
        load_data_from_filepath_fn=load_data_from_np_idx,
        data_processor_classes=data_processor_classes,
        valid_data_processor_classes=valid_data_processor_classes,
        train_forward_step_fn=train_forward_step,
        valid_forward_step_fn=valid_forward_step,
        compute_metrics_fn=compute_metrics,
    )


if __name__ == "__main__":
    config = get_arguments(get_custom_arguments)
    main(config, config.local_rank)
