#!/bin/bash
# Run an arbitrary command on the NPU server, in the project directory
# Usage: ./scripts/npu_run.sh "<command>"
# Example: ./scripts/npu_run.sh "python -c 'import pypto; print(pypto.__version__)'"
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    echo "Example: $0 'hostname && uname -a'"
    exit 1
fi

NPU_HOST="${NPU_HOST:-npu}"
NPU_PATH="${NPU_PATH:-~/pypto-multi}"

echo "🔧 Running on ${NPU_HOST}: $*"
echo "---"
ssh "$NPU_HOST" "cd ${NPU_PATH} && $*"
