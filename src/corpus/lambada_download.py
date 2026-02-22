import json
import os

from datasets import load_dataset


def main():
    for split in ["train", "validation", "test"]:
        dataset = load_dataset("cimec/lambada", split=split)
        output_path = f"data/lambada/{split}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item["text"]
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
