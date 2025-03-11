# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install prerequisites, add deadsnakes PPA, and install Python 3.11 and development packages
RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11 using the official get-pip.py script
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Upgrade pip and install required Python packages using Python 3.11
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir transformers torch && python3.11 -m pip install accelerate unsloth vllm pillow diffusers hf_transfer huggingface_hub tensorkube wandb

ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Set working directory
WORKDIR /code

# Copy the code files
COPY train.py /code/train.py
COPY reward_functions.py  /code/reward_functions.py
COPY hugging_face_upload.py /code/hugging_face_upload.py

# Run the application
CMD ["python3.11", "train.py"]