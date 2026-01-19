#!/bin/bash
# Build PyPTO Docker image
# This script builds a Docker image with all necessary packages and tools for PyPTO compilation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="pypto-build"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

echo "=========================================="
echo "PyPTO Docker Build"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info &>/dev/null; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

# Build Docker image
echo "Building Docker image: $IMAGE_NAME"
echo "Dockerfile: $DOCKERFILE"
echo ""

docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "Docker image built successfully!"
echo "=========================================="
echo ""
echo "Image name: $IMAGE_NAME"
echo ""
echo "Included packages and tools:"
echo "  System:"
echo "    - Ubuntu 22.04 Linux environment"
echo "    - Python 3.10 with python3-dev (development headers)"
echo "    - Build tools: CMake, make, ninja-build, ccache"
echo "    - Compilers: GCC, G++ (GNU Compiler Collection)"
echo ""
echo "  Python Build Dependencies:"
echo "    - setuptools>=77.0.3"
echo "    - pybind11>=2.13.6"
echo "    - build, packaging, tomli"
echo ""
echo "  PyPTO Runtime Dependencies:"
echo "    - sympy, PyYAML"
echo ""
echo "  Testing Dependencies:"
echo "    - pytest, pytest-forked, pytest-xdist"
echo ""
echo "  Scientific Computing:"
echo "    - numpy, scipy"
echo "    - PyTorch 2.6.0 (CPU version)"
echo ""
echo "  Visualization:"
echo "    - matplotlib, pandas, plotly, tabulate"
echo ""
echo "  Additional Tools:"
echo "    - ml_dtypes, jinja2, cloudpickle, tornado"
echo ""
echo "To start the container:"
echo "  ./docker-run.sh"
echo ""
echo "Or manually:"
echo "  docker run --rm -it -v \$(pwd):/workspace -w /workspace $IMAGE_NAME"
echo ""
