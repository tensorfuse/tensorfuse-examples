FROM ghcr.io/ggerganov/llama.cpp:full-cuda

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV GGML_CUDA_MAX_STREAMS=16
ENV GGML_CUDA_MMQ_Y=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install huggingface_hub hf-transfer

# Copy and set permissions
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh 

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
