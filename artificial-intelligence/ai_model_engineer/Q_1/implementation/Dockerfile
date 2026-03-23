FROM ubuntu:24.04
# Dockerfile for container v1.1.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    sudo \
    cmake \
    nano \
    wget \
    tmux \
    curl \
    gdb \
    ninja-build \
    dpkg \
    dkms \
    libfmt-dev \
    libopencv-dev \
    nlohmann-json3-dev \
    apt clean && rm -rf /var/lib/apt/lists/*

# CUDA 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin && \
    mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    
RUN apt-get update && \
    apt-get -y install cuda-toolkit-12-8 && \
    rm cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb


# cuDNN 9.7.1
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2404-9.12.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cudnn && \
    rm cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb


# TensorRT 10.8.1
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/local_repo/nv-tensorrt-local-repo-ubuntu2404-10.8.0-cuda-12.8_1.0-1_amd64.deb && \
    dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.8.0-cuda-12.8_1.0-1_amd64.deb && \
    cp /var/nv-tensorrt-local-repo-ubuntu2404-10.8.0-cuda-12.8/*-keyring.gpg /usr/share/keyrings/ && \
    apt update && \
    apt install -y tensorrt-dev && \
    rm nv-tensorrt-local-repo-ubuntu2404-10.8.0-cuda-12.8_1.0-1_amd64.deb && \
    apt clean && rm -rf /var/lib/apt/lists/*


