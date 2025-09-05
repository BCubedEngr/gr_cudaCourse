ARG CUDA_MAJOR=12
ARG CUDA_MINOR=6
ARG CUDA_PATCH=3
ARG UB_VERSION=22.04

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.${CUDA_PATCH}-base-ubuntu${UB_VERSION}
ARG CUDA_MAJOR
ARG CUDA_MINOR

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y gnuradio cmake git cuda-minimal-build-${CUDA_MAJOR}-${CUDA_MINOR} libgtest-dev vim \
    libgtk-3-dev python3-pip && \
    pip3 install pygccxml --upgrade

WORKDIR /work

COPY . .

