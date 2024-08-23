# Use the nvidia cuda base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# Update and install required packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*
# Set Python 3.10 as the default Python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python
# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip install transformers \
    && pip install --upgrade torch fastapi uvicorn pydantic huggingface_hub torchvision packaging setuptools python-multipart \
    && git clone https://github.com/facebookresearch/segment-anything-2.git \
    && pip install -e segment-anything-2
# Set working directory
WORKDIR /code
# Run the downloader script to download the model
RUN mkdir checkpoints \
    && cd checkpoints \
    && wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    && cd ..
COPY main.py /code/main.py
EXPOSE 80
# Start a uvicorn server on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
