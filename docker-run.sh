#!/bin/bash
# Run PyPTO Docker container
# This script starts an interactive Docker container with the PyPTO project mounted

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

IMAGE_NAME="pypto-build"
CONTAINER_NAME="pypto-build-container"

echo "=========================================="
echo "PyPTO Docker Container"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info &>/dev/null; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if image exists, build if not
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' not found."
    echo "Building image..."
    echo ""
    ./docker-build.sh
    echo ""
fi

echo "Starting Docker container..."
echo "  Image: $IMAGE_NAME"
echo "  Project directory: $PROJECT_ROOT"
echo "  Mounted at: /workspace"
echo ""

# Check if container already exists and is running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' is already running."
        echo "Attaching to existing container..."
        echo ""
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0
    else
        echo "Container '$CONTAINER_NAME' exists but is not running. Removing it..."
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
fi

echo "Inside the container, you can:"
echo "  1. Build PyPTO:"
echo "     python3 -m pip install -e . --verbose"
echo ""
echo "  2. Or use the build script:"
echo "     ./build.sh"
echo ""
echo "  3. Run tests:"
echo "     python3 -m pytest tests/"
echo ""
echo "  4. Run Python:"
echo "     python3"
echo ""
echo "  5. Exit:"
echo "     exit"
echo ""
echo "=========================================="
echo ""

# Run container
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    /bin/bash
