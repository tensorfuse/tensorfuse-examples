# Use the nvidia cuda base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Update and install required packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip && pip install transformers && pip install torch fastapi uvicorn pydantic accelerate && pip install -q --upgrade transformers autoawq accelerate


# Set working directory
WORKDIR /code

# Copy the code files
COPY main.py /code/main.py
COPY download_model.py /code/download_model.py

# Run the downloader script to download the model
RUN python download_model.py

EXPOSE 80

# Start a uvicorn server on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
