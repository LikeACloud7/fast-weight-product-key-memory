python src/corpus/pile_domain_download.py \
  --out_dir data/pile_domain/ \
  --target_chars 80000000

# Test Stream
for domain in dm_mathematics freelaw philpapers pubmed_central ubuntu_irc uspto_backgrounds; do
  python src/corpus/encode_lm_data.py \
      --input_filepath data/pile_domain/${domain}/data.jsonl \
      --output_prefix ${domain}.8mt \
      --tokenizer_dir mistralai/Mistral-7B-v0.1 \
      --output_dir data/pile_domain/encoded/mistral32k/stream \
      --add_eos \
      --num_workers 8 \
      --max_total_tokens 8192000
done

# Test l4k
for domain in dm_mathematics freelaw philpapers pubmed_central ubuntu_irc uspto_backgrounds; do
  python src/corpus/encode_lm_data.py \
      --input_filepath data/pile_domain/${domain}/data.jsonl \
      --tokenizer_dir mistralai/Mistral-7B-v0.1 \
      --output_dir data/pile_domain/encoded/mistral32k/l4096 \
      --output_prefix ${domain}.8mt \
      --max_seq_len 4096 \
      --num_workers 8 \
      --add_eos \
      --max_total_tokens 8192000
done
