# Manylinux Docker Setup Guide

This guide explains how to use the manylinux Docker setup to build Python wheels that are compatible with many Linux distributions.

## What is Manylinux?

Manylinux is a standardized Docker image that provides a consistent Linux environment for building Python wheels. Wheels built in manylinux containers are compatible with most Linux distributions, making them ideal for distribution on PyPI.

## Prerequisites

1. **Docker installed and running**
   - macOS: `brew install --cask docker`
   - Linux: `sudo apt-get install docker.io`

2. **Verify Docker is running:**
   ```bash
   docker --version
   docker info
   ```

## Quick Start

### 1. Build the Manylinux Docker Image

```bash
# Build the default manylinux2014 image
./build-manylinux-docker.sh

# Or specify custom version
MANYLINUX_VERSION=2_24 ./build-manylinux-docker.sh
```

### 2. Run the Container

```bash
# Start interactive container
./run-manylinux-docker.sh start

# Or enter shell directly
./run-manylinux-docker.sh shell
```

### 3. Build Python Wheels

```bash
# Build wheel for Python 3.11
./run-manylinux-docker.sh build-wheel 3.11

# Build wheels for all Python versions
./run-manylinux-docker.sh build-all-wheels
```

## Available Commands

### Build Script (`build-manylinux-docker.sh`)

Builds the manylinux Docker image.

**Environment Variables:**
- `MANYLINUX_IMAGE_NAME`: Docker image name (default: `pypto-manylinux`)
- `MANYLINUX_VERSION`: Manylinux version (default: `2014`)
- `PLATFORM`: Platform architecture (default: `x86_64`)
- `DOCKERFILE`: Dockerfile path (default: `Dockerfile.manylinux`)

**Examples:**
```bash
# Default build
./build-manylinux-docker.sh

# Custom image name
MANYLINUX_IMAGE_NAME=my-manylinux ./build-manylinux-docker.sh

# Use manylinux_2_24
MANYLINUX_VERSION=2_24 ./build-manylinux-docker.sh
```

### Run Script (`run-manylinux-docker.sh`)

Manages the manylinux Docker container.

**Commands:**
- `start` or `run`: Start a new interactive container
- `shell` or `bash`: Enter the container shell
- `build-wheel [VERSION]`: Build wheel for specific Python version (3.9, 3.10, 3.11, 3.12)
- `build-all-wheels`: Build wheels for all Python versions
- `audit-wheel FILE`: Audit and repair wheel for manylinux compatibility
- `stop`: Stop the running container
- `clean`: Stop and remove the container
- `help`: Show help message

**Examples:**
```bash
# Start container
./run-manylinux-docker.sh start

# Build wheel for Python 3.11
./run-manylinux-docker.sh build-wheel 3.11

# Build all wheels
./run-manylinux-docker.sh build-all-wheels

# Audit a wheel
./run-manylinux-docker.sh audit-wheel dist/pypto-0.1.0-cp311-cp311-linux_x86_64.whl
```

## Manylinux Versions

### manylinux2014 (Default)
- Based on CentOS 7
- Maximum compatibility
- Supports older Linux distributions
- Tag: `quay.io/pypa/manylinux2014_x86_64`

### manylinux_2_24
- Based on Debian 9
- Good compatibility
- Tag: `quay.io/pypa/manylinux_2_24_x86_64`

### manylinux_2_28
- Based on Debian 12
- Newer, but less compatible
- Tag: `quay.io/pypa/manylinux_2_28_x86_64`

## Python Versions Available

The manylinux images include multiple Python versions:
- Python 3.9: `/opt/python/cp39-cp39/bin/python`
- Python 3.10: `/opt/python/cp310-cp310/bin/python`
- Python 3.11: `/opt/python/cp311-cp311/bin/python`
- Python 3.12: `/opt/python/cp312-cp312/bin/python`

## Building Wheels

### Method 1: Using the Script

```bash
# Build for specific Python version
./run-manylinux-docker.sh build-wheel 3.11

# Build for all versions
./run-manylinux-docker.sh build-all-wheels
```

### Method 2: Manual Build Inside Container

```bash
# Enter container
./run-manylinux-docker.sh shell

# Inside container
cd /workspace
/opt/python/cp311-cp311/bin/python -m pip install build
/opt/python/cp311-cp311/bin/python -m build --wheel
```

### Method 3: Using pip wheel

```bash
# Inside container
/opt/python/cp311-cp311/bin/python -m pip wheel . -w dist/
```

## Auditing Wheels

After building a wheel, you should audit it to ensure manylinux compatibility:

```bash
./run-manylinux-docker.sh audit-wheel dist/pypto-0.1.0-cp311-cp311-linux_x86_64.whl
```

This will:
1. Check the wheel for manylinux compatibility
2. Repair any issues
3. Output a repaired wheel in `dist/`

## Workflow Example

```bash
# 1. Build the Docker image (first time only)
./build-manylinux-docker.sh

# 2. Build wheels for all Python versions
./run-manylinux-docker.sh build-all-wheels

# 3. Audit each wheel
for wheel in dist/*.whl; do
    ./run-manylinux-docker.sh audit-wheel "$wheel"
done

# 4. Check the repaired wheels in dist/
ls -lh dist/
```

## Troubleshooting

### Image Build Fails

```bash
# Check Docker is running
docker info

# Try pulling the base image manually
docker pull quay.io/pypa/manylinux2014_x86_64:latest
```

### Container Won't Start

```bash
# Check if image exists
docker images | grep pypto-manylinux

# Rebuild if needed
./build-manylinux-docker.sh
```

### Wheel Build Fails

```bash
# Enter container and debug
./run-manylinux-docker.sh shell

# Check Python version
/opt/python/cp311-cp311/bin/python --version

# Install dependencies manually
/opt/python/cp311-cp311/bin/python -m pip install -e .
```

### Permission Issues

If you encounter permission issues, you may need to adjust file ownership:

```bash
# Inside container
chown -R $(id -u):$(id -g) /workspace
```

## Advanced Usage

### Custom Dockerfile

You can create a custom Dockerfile based on `Dockerfile.manylinux`:

```bash
# Use custom Dockerfile
DOCKERFILE=Dockerfile.manylinux.custom ./build-manylinux-docker.sh
```

### Multi-Platform Builds

For ARM64 builds, use:

```bash
PLATFORM=aarch64 ./build-manylinux-docker.sh
```

Note: ARM64 manylinux images may have different tags.

### Environment Variables

Set environment variables for the container:

```bash
docker run -it --rm \
    -v "$(pwd):/workspace" \
    -e CUSTOM_VAR=value \
    pypto-manylinux:latest \
    /bin/bash
```

## Notes

- Wheels built in manylinux containers are only compatible with Linux
- For macOS wheels, use a macOS build environment
- For Windows wheels, use a Windows build environment
- The container mounts the current directory to `/workspace`
- All build artifacts are stored in the mounted directory
- The container is ephemeral (removed when stopped) unless you use `--name` and don't use `--rm`

## References

- [Manylinux Project](https://github.com/pypa/manylinux)
- [PEP 513](https://www.python.org/dev/peps/pep-0513/) - manylinux1
- [PEP 571](https://www.python.org/dev/peps/pep-0571/) - manylinux2010
- [PEP 599](https://www.python.org/dev/peps/pep-0599/) - manylinux2014
- [PEP 600](https://www.python.org/dev/peps/pep-0600/) - manylinux platform tags
