# Use the nvidia cuda base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 80
CMD ["uvicorn", "tts_deploy:app", "--host", "0.0.0.0", "--port", "80"]
