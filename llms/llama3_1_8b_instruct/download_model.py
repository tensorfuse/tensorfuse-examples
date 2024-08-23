import os

from huggingface_hub import snapshot_download
access_token = '<YOUR-HUGGINGFACE_TOKEN>'


if __name__=='__main__':
    # download the meta/llama3 model
    os.makedirs('./models', exist_ok=True)
    snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", local_dir="models",ignore_patterns=["*.pt", "*.bin"], token=access_token)
