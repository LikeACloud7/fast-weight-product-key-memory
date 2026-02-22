import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Global variables for worker processes
tokenizer = None
add_eos_to_line = False


def init_worker(tokenizer_dir, add_eos_flag):
    """
    Initializer for the multiprocessing pool.
    Loads the tokenizer and sets the EOS flag for each worker process.
    """
    global tokenizer, add_eos_to_line
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    add_eos_to_line = add_eos_flag


def encode_line(line):
    """
    Takes a single line of JSON text, tokenizes it, and optionally appends the EOS token.
    Returns a list of token IDs.
    """
    # Load the JSON data from the line
    line_data = json.loads(line)
    text = line_data["text"]

    # Encode the text; do not add special tokens as we handle that manually
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Conditionally append the end-of-sentence token ID
    if add_eos_to_line:
        token_ids.append(tokenizer.eos_token_id)

    # Return a standard list of integers
    return token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a text file into fixed-size chunks and save in binary format."
    )
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--input_filepath", type=str, required=True, help="Path to the input JSONL text file")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the output binary and index files"
    )
    parser.add_argument(
        "--max_seq_len", type=int, help="The base sequence length for training (chunks will be max_seq_len + 1)."
    )
    parser.add_argument("--max_total_tokens", type=int)

    parser.add_argument("--output_prefix", type=str, help="Prefix for the output binary and index files")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of worker processes to use")
    parser.add_argument("--add_eos", action="store_true", help="Add an EOS token at the end of each document (line).")
    args = parser.parse_args()

    # --- Setup ---

    # Get the total number of lines for the progress bar
    with open(args.input_filepath, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in tqdm(f))

    # Determine the output file prefix if not provided
    if not args.output_prefix:
        input_filename = os.path.basename(args.input_filepath)
        args.output_prefix = input_filename.replace(".jsonl", "")

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # In the main process, load the tokenizer to determine the vocabulary size and
    # the appropriate NumPy dtype for storing tokens.
    main_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    vocab_size = len(main_tokenizer)
    if vocab_size <= 2**16:
        dtype = np.uint16
    elif vocab_size <= 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    del main_tokenizer  # Free up memory

    # --- Main Processing ---

    # Define the target size for each token chunk.
    # This is max_seq_len + 1 to prepare data for autoregressive model training,
    # where input_ids are tokens[0...T-1] and labels are tokens[1...T].
    if args.max_seq_len:
        chunk_size = args.max_seq_len + 1

    # Open the input file for reading
    with open(args.input_filepath, "r", encoding="utf-8") as f:
        # Create a multiprocessing pool
        num_procs = min(args.num_workers, num_lines)
        init_args = (args.tokenizer_dir, args.add_eos)
        with mp.Pool(processes=num_procs, initializer=init_worker, initargs=init_args) as pool:
            output_file_path = os.path.join(args.output_dir, f"{args.output_prefix}.bin")
            index_file_path = os.path.join(args.output_dir, f"{args.output_prefix}.idx")

            with open(output_file_path, "wb") as out_f, open(index_file_path, "w") as idx_f:
                current_offset = 0
                if args.max_seq_len:
                    token_buffer = []  # Buffer to accumulate tokens from processed lines

                # Use pool.imap for ordered processing.
                for token_ids in tqdm(pool.imap(encode_line, f, chunksize=100), total=num_lines):
                    if args.max_seq_len:
                        token_buffer.extend(token_ids)

                        # While the buffer has enough tokens for at least one full chunk
                        while len(token_buffer) >= chunk_size:
                            # Extract a chunk of size (max_seq_len + 1)
                            chunk_ids = token_buffer[:chunk_size]
                            token_buffer = token_buffer[chunk_size:]

                            # Convert the chunk to a NumPy array for efficient binary saving
                            chunk_array = np.array(chunk_ids, dtype=dtype)

                            # Write the numpy array to the binary file
                            out_f.write(chunk_array.tobytes())

                            # Write the number of tokens and the current offset to the index file
                            idx_f.write(f"{len(chunk_array)} {current_offset}\n")

                            # Update the offset by the number of bytes written
                            current_offset += chunk_array.nbytes
                    else:
                        # If not chunking, write the token IDs directly
                        token_ids = np.array(token_ids, dtype=dtype)
                        out_f.write(token_ids.tobytes())
                        idx_f.write(f"{len(token_ids)} {current_offset}\n")
                        current_offset += token_ids.nbytes

                    if args.max_total_tokens and (current_offset // dtype().nbytes) >= args.max_total_tokens:
                        print(f"\nReached max_total_tokens limit of {args.max_total_tokens}. Stopping.")
                        break

            pool.close()

            # After processing all lines, write any remaining tokens in the buffer as a final chunk
            # if token_buffer:
            #     chunk_array = np.array(token_buffer, dtype=dtype)
            #     out_f.write(chunk_array.tobytes())
            #     idx_f.write(f"{len(chunk_array)} {current_offset}\n")

    print(f"\nProcessing complete. ✅")
    print(f"Binary data saved to: {output_file_path}")
    print(f"Index file saved to: {index_file_path}")
    print(f"Total number of lines processed: {num_lines}")
    if args.max_seq_len:
        print(f"Total number of chunks written: {current_offset // (chunk_size * dtype().nbytes)}")
    print(f"Total number of tokens written: {current_offset // dtype().nbytes}")
