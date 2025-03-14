FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy DeepGEMM into container
COPY . /workspace/DeepGEMM

# Install Python dependencies (if any, e.g., torch)
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install -e /workspace/DeepGEMM

# Environment variables (optional)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command
CMD ["/bin/bash"]
