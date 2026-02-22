python src/corpus/fineweb_download.py --output_filepath data/fineweb/10BT.jsonl

head -n 9000000 data/fineweb/10BT.jsonl > data/fineweb/10BT.train9m.jsonl
tail -n 100000 data/fineweb/10BT.jsonl | head -n 10000 > data/fineweb/10BT.valid10k.jsonl
tail -n 100000 data/fineweb/10BT.jsonl | tail -n 10000 > data/fineweb/10BT.test10k.jsonl

# L 4096
for FILE in data/fineweb/10BT.train9m.jsonl \
             data/fineweb/10BT.valid10k.jsonl; do
    python src/corpus/encode_lm_data.py \
        --input_filepath $FILE \
        --tokenizer_dir mistralai/Mistral-7B-v0.1 \
        --output_dir data/fineweb/encoded/mistral32k/l4096 \
        --max_seq_len 4096 \
        --add_eos
done

head -n 1220704 data/fineweb/encoded/mistral32k/l4096/10BT.train9m.idx > data/fineweb/encoded/mistral32k/l4096/10BT.train9m.5bt.idx

# Test
python src/corpus/encode_lm_data.py \
    --input_filepath data/fineweb/10BT.test10k.jsonl \
    --output_prefix test10k.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/fineweb/encoded/mistral32k/l4096 \
    --max_seq_len 4096 \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000

# Test - Stream
python src/corpus/encode_lm_data.py \
    --input_filepath data/fineweb/10BT.test10k.jsonl \
    --output_prefix test10k.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/fineweb/encoded/mistral32k/stream \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000
