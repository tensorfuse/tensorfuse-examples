# Use the NVIDIA Triton Inference Server base image
FROM nvcr.io/nvidia/tritonserver:25.02-pyt-python-py3

# Clone the model repository from GitHub
RUN mkdir -p /model_repository/csm_1b/1
RUN git clone https://github.com/SesameAILabs/csm.git /model_repository/csm_1b/1

# Install Python dependencies
RUN pip install --no-cache-dir --ignore-installed -r /model_repository/csm_1b/1/requirements.txt \
    && pip install --no-cache-dir --ignore-installed numpy hf_transfer

# Copy the code files
COPY model_repository/csm_1b/1/model.py /model_repository/csm_1b/1
COPY model_repository/csm_1b/config.pbtxt /model_repository/csm_1b/config.pbtxt

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose Triton gRPC and HTTP ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Start Triton Server
CMD ["tritonserver", "--model-repository=/model_repository", "--allow-gpu-metrics=false", "--allow-metrics=false", "--metrics-port=0" ]