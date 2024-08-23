# Use the Hugging Face base image
FROM ghcr.io/huggingface/text-embeddings-inference:86-1.5

# Install Python 3.10, Nginx, and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install the requests library
RUN pip3 install --no-cache-dir --upgrade pip && pip install requests

# Copy the Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80 for Nginx
EXPOSE 80

# Start both Nginx and the text-embeddings-router
CMD ["sh", "-c", "nginx && text-embeddings-router --json-output --max-batch-tokens 163840 --model-id jinaai/jina-embeddings-v2-base-code --port 8000"]
