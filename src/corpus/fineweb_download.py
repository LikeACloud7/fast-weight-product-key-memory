import argparse
import json
import os
from typing import IO, Callable

from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.readers import ParquetReader
# from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    """Write data to datafolder (local or remote) in JSONL format

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        compression: if any compression scheme should be used. By default, "infer" - will be guessed from the filename
        adapter: a custom function to "adapt" the Document format to the desired output format
        expand_metadata: save each metadata entry in a different column instead of as a dictionary
    """

    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 Jsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "gzip",
        adapter: Callable = None,
        expand_metadata: bool = False,
        max_file_size: int = -1,  # in bytes. -1 for unlimited
    ):
        super().__init__(
            output_folder,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            expand_metadata=expand_metadata,
            mode="wt",
            max_file_size=max_file_size,
        )

    def _write(self, document: dict, file_handler: IO, _filename: str):
        file_handler.write(json.dumps(document, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_filepath", type=str, default="data/fineweb/10BT/100000.jsonl", help="Path to save the output JSONL file.")
    parser.add_argument("--lines", type=int, default=-1, help="Number of lines to read from the Parquet file.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_filepath)
    filename = os.path.basename(args.output_filepath)

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", limit=args.lines),
            JsonlWriter(output_dir, filename, compression=None)
        ],
        tasks=1
    )
    pipeline_exec.run()