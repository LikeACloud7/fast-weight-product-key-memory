# Download nanogpt-style data from https://manifestai.com/articles/longcrawl64/

# Move to the following locations
mv data/lc64/train.bin data/lc64/50BT.bin
mv data/lc64/val.bin data/lc64/test.bin

# Subsample 10BT train set
head -c 20GB data/lc64/train.bin > data/lc64/10BT.bin

# Decode
python src/corpus/lc64_nanogpt_process.py data/lc64/train.10BT.bin data/lc64/10BT.jsonl  --workers 64
python src/corpus/lc64_nanogpt_process.py data/lc64/test.bin data/lc64/test.jsonl --workers 64

# Subsample a 100L valid set, the rest for training
head -n 100 data/lc64/10BT.jsonl > data/lc64/10BT.valid.jsonl
tail -n +101 data/lc64/10BT.jsonl > data/lc64/10BT.train.jsonl

# l4k
for FILE in data/lc64/10BT.valid.jsonl \
            data/lc64/10BT.train.jsonl; do
    python src/corpus/encode_lm_data.py \
        --input_filepath $FILE \
        --tokenizer_dir mistralai/Mistral-7B-v0.1 \
        --output_dir data/lc64/encoded/mistral32k/l4096 \
        --max_seq_len 4096 \
        --num_workers 64 \
        --add_eos
done

# Test l4k
python src/corpus/encode_lm_data.py \
    --input_filepath data/lc64/test.jsonl \
    --output_prefix test.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/lc64/encoded/mistral32k/l4096 \
    --max_seq_len 4096 \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000

# Test Stream
python src/corpus/encode_lm_data.py \
    --input_filepath data/lc64/test.jsonl \
    --output_prefix test.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/lc64/encoded/mistral32k/stream \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000

