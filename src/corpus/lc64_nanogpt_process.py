import argparse
import os
import shutil
import multiprocessing
import numpy as np
import tiktoken
import json
import time
from tqdm import tqdm

# Configuration
TEMP_DIR = "temp_parts"
BATCH_SIZE = 10_000_000  # Read 10M tokens (~20MB) at a time to keep RAM low


def decode_chunk(args):
    """
    Worker function to decode a specific chunk of the binary file.
    """
    input_path, chunk_id, start_offset, end_offset, model_name, output_file = args

    # Re-init tokenizer per process (tiktoken objects cannot be pickled)
    try:
        enc = tiktoken.get_encoding(model_name)
    except Exception:
        # Fallback if model name is custom or fails
        enc = tiktoken.get_encoding("gpt2")

    eot_token = enc.eot_token

    # Get file size and open memmap
    file_size = os.path.getsize(input_path)
    total_tokens = file_size // 2
    data = np.memmap(input_path, dtype=np.uint16, mode="r", shape=(total_tokens,))

    # --- 1. Determine Start Position ---
    current_idx = start_offset

    # If we are not the first worker, we must skip the first partial document.
    # The previous worker is responsible for the document that straddles the boundary.
    if chunk_id > 0:
        # We simply scan forward until we hit an EOT.
        # We do this in a loop to handle rare cases where a doc is huge (>BATCH_SIZE)
        while current_idx < total_tokens:
            search_limit = min(current_idx + BATCH_SIZE, total_tokens)
            chunk_view = data[current_idx:search_limit]
            local_eot_indices = np.where(chunk_view == eot_token)[0]

            if len(local_eot_indices) > 0:
                # Found the boundary. The *next* doc starts after this EOT.
                current_idx += local_eot_indices[0] + 1
                break
            else:
                # No EOT in this batch, jump forward and try again
                current_idx = search_limit

        # Safety check: if we ran off the end of the file
        if current_idx >= total_tokens:
            return

    # --- 2. Process Documents ---
    with open(output_file, "w", encoding="utf-8") as f:
        while current_idx < end_offset:
            # Define our viewing window
            search_limit = min(current_idx + BATCH_SIZE, total_tokens)
            block_view = data[current_idx:search_limit]

            # Find all EOTs in this block
            eot_indices = np.where(block_view == eot_token)[0]

            # Edge Case: Massive document (larger than BATCH_SIZE)
            # If no EOT is found, we cannot process anything yet.
            # We must expand the search window by jumping to the next iteration
            # WITHOUT advancing current_idx (unless we are at EOF).
            if len(eot_indices) == 0:
                if search_limit == total_tokens:
                    break  # EOF reached

                # We need to search a larger chunk.
                # However, numpy memmap is fast, so we can just iterate the
                # outer loop. But we must ensure we don't infinite loop.
                # If we found nothing, it means the document extends BEYOND search_limit.
                # We can't decode it yet. We need a way to handle this.
                # Simple fix: Scan linearly for the next EOT if block search fails.

                # Fallback for massive docs: scan ahead block by block just to find the END
                scan_cursor = search_limit
                found_huge_end = False
                while scan_cursor < total_tokens:
                    scan_end = min(scan_cursor + BATCH_SIZE, total_tokens)
                    scan_view = data[scan_cursor:scan_end]
                    scan_eots = np.where(scan_view == eot_token)[0]
                    if len(scan_eots) > 0:
                        # Found the end at: scan_cursor + scan_eots[0]
                        abs_end = scan_cursor + scan_eots[0]
                        # Now we can decode the huge doc
                        full_doc = data[current_idx:abs_end]
                        text = enc.decode(full_doc.tolist())
                        f.write(f'{{"text": {json.dumps(text)}}}\n')
                        current_idx = abs_end + 1
                        found_huge_end = True
                        break
                    scan_cursor = scan_end

                if found_huge_end:
                    if current_idx >= end_offset:
                        return
                    continue
                else:
                    break  # End of file during huge doc scan

            # Standard Case: We found EOTs in the buffer
            relative_start = 0
            for rel_eot in eot_indices:
                # Slice directly from the view for speed
                doc_tokens = block_view[relative_start:rel_eot]

                # Filter empty documents if necessary
                if len(doc_tokens) > 0:
                    # Decode
                    text = enc.decode(doc_tokens.tolist())
                    # Write manual JSON line (faster than constructing dicts)
                    f.write(f'{{"text": {json.dumps(text)}}}\n')

                # Check if we have passed our assigned chunk limit
                # We calculate the absolute position of the EOT we just processed
                abs_eot_idx = current_idx + rel_eot

                if abs_eot_idx >= end_offset:
                    return  # Done with our share

                # Move relative start to next token after EOT
                relative_start = rel_eot + 1

            # Advance global pointer
            current_idx += relative_start


def main():
    parser = argparse.ArgumentParser(description="Fast parallel decoder for binary token files.")
    parser.add_argument("input", help="Input binary file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--model", default="gpt2", help="Tokenizer model name (e.g., gpt2, cl100k_base)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    # Setup temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    file_size = os.path.getsize(args.input)
    total_tokens = file_size // 2
    chunk_size = total_tokens // args.workers

    print(f"Total Tokens: {total_tokens:,}")
    print(f"Workers: {args.workers} | Chunk Size: ~{chunk_size:,}")

    # Prepare worker arguments
    worker_args = []
    for i in range(args.workers):
        start = i * chunk_size
        # Last worker goes to the very end
        end = (i + 1) * chunk_size if i < args.workers - 1 else total_tokens
        part_file = os.path.join(TEMP_DIR, f"part_{i:03d}.jsonl")
        worker_args.append((args.input, i, start, end, args.model, part_file))

    print("Starting decoding...")
    t0 = time.time()

    with multiprocessing.Pool(args.workers) as pool:
        # Use tqdm to track worker completion
        list(tqdm(pool.imap(decode_chunk, worker_args), total=args.workers))

    print("Merging temporary files...")
    with open(args.output, "wb") as outfile:
        for i in range(args.workers):
            part_file = os.path.join(TEMP_DIR, f"part_{i:03d}.jsonl")
            if os.path.exists(part_file):
                with open(part_file, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

    shutil.rmtree(TEMP_DIR)
    duration = time.time() - t0
    print(f"Finished in {duration:.2f}s. Output: {args.output}")


if __name__ == "__main__":
    main()
