import argparse
import json
import os
import random
import uuid


def generate_random_digits(length):
    """Generates a string of random digits."""
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def generate_key():
    """Generates a short unique key (e.g., 'ID-a1b2')."""
    return f"ID-{str(uuid.uuid4())[:4]}"


def insert_needles(context, needles, depths=None):
    """
    Inserts needles into the context.

    Args:
        context (str): The haystack text.
        needles (list of dict): [{'key': '...', 'value': '...', 'sentence': '...'}]
        depths (list): Optional list of depths.

    Returns:
        tuple: (modified_context, list_of_insertion_metadata)
    """
    tokens = context.split(" ")
    total_tokens = len(tokens)

    insertion_plan = []

    # Plan where to insert each needle
    for i, needle_obj in enumerate(needles):
        if depths and i < len(depths):
            target_depth = depths[i]
        else:
            target_depth = random.random()

        target_index = int(total_tokens * target_depth)
        insertion_plan.append((target_index, needle_obj, target_depth))

    # Sort descending by index to insert safely from back to front
    insertion_plan.sort(key=lambda x: x[0], reverse=True)

    modified_tokens = tokens.copy()
    realized_depths = []

    for index, needle_obj, depth in insertion_plan:
        modified_tokens.insert(index, needle_obj["sentence"])

        realized_depths.append(
            {
                "key": needle_obj["key"],
                "value": needle_obj["value"],
                "target_depth_percent": round(depth * 100, 2),
                "inserted_at_word_index": index,
            }
        )

    return " ".join(modified_tokens), realized_depths


def process_documents(args):
    created_keys = set()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        doc_count = 0
        chunk_count = 0

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                text = row.get("text", "")
            except json.JSONDecodeError:
                continue

            if not text:
                continue

            words = text.split()
            total_words = len(words)

            step = args.context_length
            if args.overlap:
                step = int(args.context_length * (1 - args.overlap))

            for i in range(0, total_words, step):
                chunk_words = words[i : i + args.context_length]
                if len(chunk_words) < (args.context_length * 0.5):
                    continue

                chunk_text = " ".join(chunk_words)

                # --- 1. Generate Key-Value Needles ---
                current_needles = []

                for _ in range(args.num_needles):
                    while True:
                        k = generate_key()
                        if k not in created_keys:
                            created_keys.add(k)
                            break
                    v = generate_random_digits(args.needle_length)

                    # Construct the sentence: "The secret code for ID-XYZ is 12345."
                    sentence = f"{args.template_prefix} {k} is {v}{args.template_suffix}"

                    current_needles.append({"key": k, "value": v, "sentence": sentence})

                # --- 2. Insert All Needles ---
                target_depths = args.depths if args.depths else None
                haystack, insertion_info = insert_needles(chunk_text, current_needles, target_depths)

                # --- 3. Construct Queries ---
                # We have inserted N needles. We create a question for ONE of them
                # to ensure the model isn't just reciting all numbers it found.

                # If you want to generate a row for EVERY needle, loop here.
                # For now, we pick one random target from the inserted set.
                target_needle = random.choice(insertion_info)

                out_record = {
                    "haystack": haystack,
                    "question_key": target_needle["key"],
                    "answer": target_needle["value"],  # The correct digits for THAT key
                    "all_inserted_needles": insertion_info,  # Debug info showing distractions
                    "metadata": {
                        "source_doc_id": row.get("id", doc_count),
                        "target_depth": target_needle["target_depth_percent"],
                    },
                }

                fout.write(json.dumps(out_record) + "\n")
                chunk_count += 1

                if args.num_samples and chunk_count >= args.num_samples:
                    print(f"Reached target of {args.num_samples} samples. Stopping.")
                    return

            doc_count += 1
            if doc_count % 100 == 0:
                print(f"Processed {doc_count} docs, {chunk_count} samples generated...")

            if args.num_samples and chunk_count >= args.num_samples:
                return

    print(f"Done. Output written to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # File Paths
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int)

    # Context Settings
    parser.add_argument("--context_length", type=int, default=4000, help="Chunk size in words")
    parser.add_argument("--overlap", type=float, default=0.0)

    # Needle Settings
    parser.add_argument("--num_needles", type=int, default=5, help="Total needles to insert (1 target + N distractors)")
    parser.add_argument("--needle_length", type=int, default=6)

    # Text Templates
    parser.add_argument("--template_prefix", type=str, default="The secret number for", help="Start of needle sentence")
    parser.add_argument("--template_suffix", type=str, default=".", help="End of needle sentence")
    parser.add_argument("--question_key_phrase", type=str, default="secret number", help="Phrase used in the question")

    # Depths
    parser.add_argument("--depths", type=float, nargs="+", help="Specific depths list. Ex: 0.1 0.5 0.9")

    args = parser.parse_args()
    process_documents(args)
