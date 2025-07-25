# vllm base image
FROM vllm/vllm-openai:v0.9.2

# Install hf-xet instead of hf-transfer
RUN pip install hf-xet huggingface_hub

# Optional: Enable high performance mode for hf-xet
ENV HF_XET_HIGH_PERFORMANCE=1

ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING True

# ENV VLLM_LOGGING_LEVEL DEBUG

ENV VLLM_USE_V1 1

# Expose port 80
EXPOSE 80

# Entrypoint with API key
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            # name of the model
           "--model", "Qwen/Qwen3-8B", \
           # set the data type to float 16 - requires 140GB of GPU memory
           "--dtype", "bfloat16", \
           # below runs the model on 4 GPUs
           # Maximum number of tokens, this can lead to OOM errors if overestimated
           "--max-model-len", "32768", \
           # Port on which to run the vLLM server
           "--port", "80", \
           "--enable-lora", \
           # API key for authentication to the server stored in tensorfuse secrets
           "--api-key", "test"]
