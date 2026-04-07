FROM ubuntu:22.04

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV TZ="Europe/Moscow"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang lld \
    build-essential \
    cmake \
    ccache \
    curl \
    git \
    nano \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    sudo \
    unzip \
    vim \
    wget \
    zip \
    ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.46.1

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

ENV TORCH_NPU_WHL="torch_npu-2.5.1.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
RUN wget -O /tmp/${TORCH_NPU_WHL} https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0.1-pytorch2.5.1/${TORCH_NPU_WHL} \
    && python3 -m pip install --no-cache-dir /tmp/${TORCH_NPU_WHL} \
    && rm /tmp/${TORCH_NPU_WHL}

RUN groupadd -g 1000 hwgroup

RUN if [ ${GROUP_ID} -eq 1000 ]; then \
        useradd -m -u ${USER_ID} -g hwgroup -G sudo user; \
    else \
        groupadd -g ${GROUP_ID} user && \
        useradd -m -u ${USER_ID} -g user -G hwgroup,sudo user; \
    fi && \
    echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER user

WORKDIR /home/user

ENTRYPOINT [ "/bin/bash" ]