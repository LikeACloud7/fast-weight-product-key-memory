pip install typing-extensions==4.12.2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.3 tensorboard
pip install wheel packaging
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install deepspeed==0.17.6
pip install transformers==4.57.1
pip install flash-linear-attention==0.3.2

pip install -r requirements.txt
