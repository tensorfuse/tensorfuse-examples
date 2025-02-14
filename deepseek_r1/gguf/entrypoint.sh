#!/bin/bash
set -e

# Download model shards if missing
if [ ! -d "/app/DeepSeek-R1-GGUF" ]; then
  echo "Downloading model..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id='unsloth/DeepSeek-R1-GGUF',
  local_dir='DeepSeek-R1-GGUF',
  allow_patterns=['*UD-IQ1_S*']
)"
fi

echo "Downloading model finished. Now waiting to start the llama server with optimisations for one batch latency"

# Start server with single-request optimizations
./llama-server \
  --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf\
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 62 \
  --parallel 4 \
  --ctx-size 5120 \
  --mlock \
  --threads 42 \
  --tensor-split 1,1,1,1 \
  --no-mmap \
  --rope-freq-base 1000000 \
  --rope-freq-scale 0.25 \
  --metrics
