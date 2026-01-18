# PyPTO Dockerfile for Compilation
# This Dockerfile provides a clean Linux environment for building PyPTO
# Based on macos_support branch with all necessary packages and tools

FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python version (Python 3.9 is minimum requirement, but Ubuntu 22.04 has Python 3.10)
ENV PYTHON_VERSION=3.10

# Set working directory
WORKDIR /workspace

# Install system dependencies required for PyPTO compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    make \
    ninja-build \
    ccache \
    # Compilers (GNU GCC is recommended)
    gcc \
    g++ \
    # Python and development headers (REQUIRED for compilation)
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    # System libraries
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    zlib1g-dev \
    libsqlite3-dev \
    # Math libraries
    libblas-dev \
    liblapack-dev \
    gfortran \
    libblas3 \
    # Additional tools
    pciutils \
    net-tools \
    openssh-client \
    llvm \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyPTO Python dependencies from requirements.txt
# Build dependencies
RUN pip3 install --no-cache-dir \
    setuptools>=77.0.3 \
    pybind11>=2.13.6 \
    pybind11-stubgen

# Build system dependencies (for build_ci.py)
RUN pip3 install --no-cache-dir \
    pip \
    build \
    packaging \
    tomli

# Testing dependencies (for UTest/STest)
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-forked \
    pytest-xdist

# PyPTO runtime dependencies
RUN pip3 install --no-cache-dir \
    sympy \
    PyYAML

# Scientific computing and visualization packages
# Required for swim lane diagrams and plotting
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    plotly \
    tabulate

# Additional development tools
RUN pip3 install --no-cache-dir \
    ml_dtypes \
    jinja2 \
    cloudpickle \
    tornado \
    attrs \
    cython \
    decorator \
    cffi \
    pathlib2 \
    psutil \
    protobuf \
    scipy \
    requests \
    absl-py

# Install PyTorch (CPU version for compatibility)
# Note: For NPU devices, torch-npu should be installed separately
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 \
    torchvision \
    torchaudio

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CC=gcc
ENV CXX=g++
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

# Verify installations
RUN python3 --version && \
    cmake --version && \
    gcc --version && \
    g++ --version && \
    python3 -m pip list | grep -E "(setuptools|pybind11|sympy|PyYAML|pytest)"

# Default command
CMD ["/bin/bash"]
