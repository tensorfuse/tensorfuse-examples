# vllm base image
FROM vllm/vllm-openai:latest

# Enable HF Hub Transfer
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Expose port 80
EXPOSE 80

# Entrypoint with API key
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            # name of the model
           "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", \
           # set the data type to float 16 - requires 140GB of GPU memory
           "--dtype", "bfloat16", \
           # Maximum number of tokens, this can lead to OOM errors if overestimated
           "--max-model-len", "4096", \
           # Port on which to run the vLLM server
           "--port", "80"]
