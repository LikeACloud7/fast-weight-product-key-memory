import argparse
import logging
from typing import Type

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from ike import (
    DataProcessor,
    is_rank_0,
)


logger = logging.getLogger(__name__)


def load_data_from_one_np_idx(
    tokenizer: AutoTokenizer,
    stage: str,
    filepath: str,
    data_processor_class: Type[DataProcessor],
    config: argparse.Namespace = None,
    **kwargs,
):
    # Config
    debug_mode = config.debug_mode
    debug_mode_data_size = config.debug_mode_data_size
    multiple_samples_per_jsonl_line = config.multiple_samples_per_jsonl_line
    max_seq_len = config.max_seq_len

    # Initialize processor
    data_processor = data_processor_class(config=config, tokenizer=tokenizer, filepath=filepath, stage=stage, **kwargs)

    # Load data indices
    bin_filepath = f"{filepath}.bin"
    index_filepath = f"{filepath}.idx"
    indices = []
    with open(index_filepath, "r", encoding="utf-8") as f:
        for line in f:
            length, offset = map(int, line.strip().split())
            indices.append((length, offset))

    # Load data
    dataset = []
    for line_idx, (length, offset) in tqdm(
        enumerate(indices), desc=f"Loading data from {filepath}", disable=not is_rank_0()
    ):
        result = data_processor.line2data(
            (
                line_idx,
                {
                    "length": length,
                    "offset": offset,
                    "filepath": bin_filepath,
                },
            )
        )
        if result is not None:
            if multiple_samples_per_jsonl_line:
                dataset.extend(result)
            else:
                dataset.append(result)

        if debug_mode and len(dataset) >= debug_mode_data_size:
            break

    if is_rank_0():
        logger.info(f"Loaded {len(dataset)} data from {filepath}")

    return dataset


def load_data_from_one_np_bin(
    tokenizer: AutoTokenizer,
    stage: str,
    filepath: str,
    data_processor_class: Type[DataProcessor],
    config: argparse.Namespace = None,
    **kwargs,
):
    # Config
    debug_mode = config.debug_mode
    debug_mode_data_size = config.debug_mode_data_size
    multiple_samples_per_jsonl_line = config.multiple_samples_per_jsonl_line
    max_seq_len = config.max_seq_len
    # Get dtype
    vocab_size = len(tokenizer)
    if vocab_size <= 2**16:
        dtype = np.dtype(np.uint16)
    elif vocab_size <= 2**32:
        dtype = np.dtype(np.uint32)
    else:
        dtype = np.dtype(np.uint64)
    itemsize = dtype.itemsize

    # Initialize processor
    data_processor = data_processor_class(config=config, tokenizer=tokenizer, filepath=filepath, stage=stage, **kwargs)

    # Load data indices
    bin_filepath = f"{filepath}.bin"
    index_filepath = f"{filepath}.idx"
    index = []
    with open(index_filepath, "r", encoding="utf-8") as f:
        for line in f:
            length, offset = map(int, line.strip().split())
            index.append((length, offset))

    # token_ids2loss_mask fn
    if hasattr(data_processor, "token_ids2loss_mask"):
        token_ids2loss_mask_fn = data_processor.token_ids2loss_mask
    else:
        token_ids2loss_mask_fn = lambda token_ids: [1] * len(token_ids)

    # Load data
    dataset = []
    cur_token_id_seqs = []
    cur_loss_mask_seqs = []
    cur_seg_id_seqs = []
    with open(bin_filepath, "rb") as f:
        for line_idx, (length, offset) in tqdm(
            enumerate(index), desc=f"Loading data from {filepath}", disable=not is_rank_0()
        ):
            f.seek(offset)
            num_bytes = length * itemsize
            line = f.read(num_bytes)
            line = np.frombuffer(line, dtype=dtype)

            token_ids = line.tolist()
            loss_mask = token_ids2loss_mask_fn(token_ids)

            assert len(cur_token_id_seqs) in [0, 1]
            # If cur_token_id_seqs has a remaining seq, try to append it until max_seq_len+1
            if len(cur_token_id_seqs) == 1:
                cur_seq = cur_token_id_seqs[-1]
                cur_loss_masks = cur_loss_mask_seqs[-1]
                cur_seg_ids = cur_seg_id_seqs[-1]
                # Append full seq if it fits
                if len(cur_seq) + len(token_ids) <= max_seq_len + 1:
                    cur_seq.extend(token_ids)
                    cur_loss_masks.extend(loss_mask)
                    cur_seg_ids.extend([cur_seg_ids[-1] + 1] * len(token_ids))
                    token_ids = []
                # Otherwise, save the current seq and start a new one
                else:
                    cur_seq.extend(token_ids[: max_seq_len + 1 - len(cur_seq)])
                    cur_loss_masks.extend(loss_mask[: max_seq_len + 1 - len(cur_seq)])
                    cur_seg_ids.extend([cur_seg_ids[-1] + 1] * (max_seq_len + 1 - len(cur_seq)))
                    token_ids = token_ids[max_seq_len + 1 - len(cur_seq) :]
            # Split token_ids into chunks of max_seq_len + 1
            while len(token_ids) > 0:
                cur_seq = token_ids[: max_seq_len + 1]
                cur_loss_masks = loss_mask[: max_seq_len + 1]
                cur_seg_ids = [0] * len(cur_seq)
                if len(cur_seq) > 0:
                    cur_token_id_seqs.append(cur_seq)
                    cur_loss_mask_seqs.append(cur_loss_masks)
                    cur_seg_id_seqs.append(cur_seg_ids)
                if len(token_ids) <= max_seq_len + 1:
                    break
                else:
                    token_ids = token_ids[max_seq_len + 1 :]
                    loss_mask = loss_mask[max_seq_len + 1 :]
            # Add full-length token ids seqs to dataset
            if len(cur_token_id_seqs) > 0:
                num_seqs = len(cur_token_id_seqs)
                for _ in range(num_seqs):
                    if len(cur_token_id_seqs[0]) == max_seq_len + 1:
                        cur_token_ids = cur_token_id_seqs.pop(0)
                        cur_loss_masks = cur_loss_mask_seqs.pop(0)
                        cur_seg_ids = cur_seg_id_seqs.pop(0)
                        result = data_processor.line2data(
                            (
                                line_idx,
                                {
                                    "token_ids": cur_token_ids,
                                    "loss_mask": cur_loss_masks,
                                    "segment_ids": cur_seg_ids,
                                },
                            )
                        )
                        if result is not None:
                            if multiple_samples_per_jsonl_line:
                                dataset.extend(result)
                            else:
                                dataset.append(result)

            if debug_mode and len(dataset) >= debug_mode_data_size:
                break

    if is_rank_0():
        logger.info(f"Loaded {len(dataset)} data from {filepath}")

    return dataset


def load_data_from_np_idx(
    tokenizer: AutoTokenizer,
    stage: str,
    filepaths: list[str],
    data_processor_classes: list[Type[DataProcessor]],
    data_reformatter_class: Type[DataProcessor] = None,
    config: argparse.Namespace = None,
    one_dataset_per_input: bool = False,
    **kwargs,
):
    data_filepath2dataset = {}
    assert len(data_processor_classes) == len(filepaths)

    for data_processor_class, filepath in zip(data_processor_classes, filepaths):
        data = load_data_from_one_np_idx(
            tokenizer=tokenizer,
            stage=stage,
            filepath=filepath,
            data_processor_class=data_processor_class,
            config=config,
            **kwargs,
        )
        data_filepath2dataset[filepath] = data

    if not one_dataset_per_input:
        data_filepath2dataset = {"all": sum(data_filepath2dataset.values(), [])}

    return data_filepath2dataset


def load_data_from_np_bin(
    tokenizer: AutoTokenizer,
    stage: str,
    filepaths: list[str],
    data_processor_classes: list[Type[DataProcessor]],
    data_reformatter_class: Type[DataProcessor] = None,
    config: argparse.Namespace = None,
    one_dataset_per_input: bool = False,
    **kwargs,
):
    data_filepath2dataset = {}
    assert len(data_processor_classes) == len(filepaths)

    for data_processor_class, filepath in zip(data_processor_classes, filepaths):
        data = load_data_from_one_np_bin(
            tokenizer=tokenizer,
            stage=stage,
            filepath=filepath,
            data_processor_class=data_processor_class,
            data_reformatter_class=data_reformatter_class,
            config=config,
            **kwargs,
        )
        data_filepath2dataset[filepath] = data

    if not one_dataset_per_input:
        data_filepath2dataset = {"all": sum(data_filepath2dataset.values(), [])}

    return data_filepath2dataset


class BasePretrainDataProcessor(DataProcessor):
    def __init__(self, config: argparse.Namespace, **kwargs):
        self.config = config

    def initializer(self):
        pass

    def token_ids2loss_mask(self, token_ids):
        loss_mask = [1] * len(token_ids)
        return loss_mask

    def line2data(self, line):
        line_idx, line_data = line
        return line_data


DATA_PROCESSOR_CLASSES = {
    "base_pretrain": BasePretrainDataProcessor,
}
