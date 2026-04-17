#!/bin/bash
# Interactive shell on the NPU server, landing in the project directory
# Usage: ./scripts/npu_shell.sh
set -euo pipefail

NPU_HOST="${NPU_HOST:-npu}"
NPU_PATH="${NPU_PATH:-~/pypto-multi}"

exec ssh -t "$NPU_HOST" "cd ${NPU_PATH} && exec \$SHELL -l"
