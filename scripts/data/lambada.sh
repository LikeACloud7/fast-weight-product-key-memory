python src/corpus/lambada_download.py

# L 4096
python src/corpus/encode_lm_data.py \
    --input_filepath data/lambada/train.jsonl \
    --output_prefix train.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/lambada/encoded/mistral32k/l4096 \
    --max_seq_len 4096 \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000

# Stream
python src/corpus/encode_lm_data.py \
    --input_filepath data/lambada/train.jsonl \
    --output_prefix train.8mt \
    --tokenizer_dir mistralai/Mistral-7B-v0.1 \
    --output_dir data/lambada/encoded/mistral32k/stream \
    --add_eos \
    --num_workers 8 \
    --max_total_tokens 8192000

# NIAH
python src/corpus/create_niah.py \
  --input_file data/lambada/train.jsonl \
  --output_file data/lambada/niah/train.ctx4k_5n.500.jsonl \
  --num_needles 5 \
  --context_length 4000 \
  --num_samples 500

python src/corpus/create_niah.py \
  --input_file data/lambada/train.jsonl \
  --output_file data/lambada/niah/train.ctx8k_5n.500.jsonl \
  --num_needles 5 \
  --context_length 8000 \
  --num_samples 500

python src/corpus/create_niah.py \
  --input_file data/lambada/train.jsonl \
  --output_file data/lambada/niah/train.ctx32k_5n.500.jsonl \
  --num_needles 5 \
  --context_length 32000 \
  --num_samples 500

python src/corpus/create_niah.py \
  --input_file data/lambada/train.jsonl \
  --output_file data/lambada/niah/train.ctx128k_5n.500.jsonl \
  --num_needles 5 \
  --context_length 128000 \
  --num_samples 500
