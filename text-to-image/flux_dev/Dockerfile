# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:25.01-pyt-python-py3

RUN pip install --no-cache-dir \
    torch \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    Pillow \
    hf_transfer \
    protobuf \
    bitsandbytes \
    sentencepiece \
    numpy


RUN mkdir -p /models/flux/1

COPY models/flux/1/model.py /models/flux/1
COPY models/flux/config.pbtxt /models/flux/config.pbtxt


# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose Triton gRPC and HTTP ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Start Triton Server
CMD ["tritonserver", "--model-repository=/models", "--allow-gpu-metrics=false", "--allow-metrics=false", "--metrics-port=0", "--http-restricted-api=inference:API_KEY=random" ] 