# Use the nvidia cuda base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variable for HF Transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Update and install required packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip && pip install uvicorn fastapi huggingface_hub[cli] hf_transfer


# Set working directory
WORKDIR /code

# Clone the Janus library
RUN git clone https://github.com/deepseek-ai/Janus.git /code/janus

# Install the Janus library using pyproject.toml
RUN pip install /code/janus
RUN pip install python-multipart


# Copy the application code
COPY main.py /code/main.py

EXPOSE 80

# Start a uvicorn server on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
