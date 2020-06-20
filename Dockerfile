FROM ubuntu:latest as base

WORKDIR /tmp

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -y install \
    curl \
    cmake \
    g++ \
    libomp-dev\
    libeigen3-dev \
    liblapack-dev \
    libopenblas-dev\
    pkg-config \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

