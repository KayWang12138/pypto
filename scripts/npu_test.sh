#!/bin/bash
# Sync + run tests for an operator on NPU + pull logs back
# Usage: ./scripts/npu_test.sh <op_name> [extra pytest args]
# Example: ./scripts/npu_test.sh relu
#          ./scripts/npu_test.sh matmul -k precision
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <op_name> [extra pytest args]"
    exit 1
fi

OP="$1"
shift
EXTRA_ARGS="$*"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

NPU_HOST="${NPU_HOST:-npu}"
NPU_PATH="${NPU_PATH:-~/pypto-multi}"

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="logs/${OP}_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="logs/${OP}_latest.log"

# 1. Sync
./scripts/npu_sync.sh

# 2. Run on NPU (adjust the test command to your project's convention)
echo "🧪 Running tests for operator: $OP"
ssh "$NPU_HOST" "mkdir -p ${NPU_PATH}/logs && cd ${NPU_PATH} && \
    (python -m pytest custom/${OP}/ -v ${EXTRA_ARGS} 2>&1 | tee logs/${OP}.log); \
    echo \"--- validate_custom_kernel_layout ---\"; \
    bash .agents/skills/ci-and-pr/ci-and-layout-check/scripts/run_validate_layout.sh custom/${OP}/ 2>&1 | tee -a logs/${OP}.log || true" \
    | tee "$LOG_FILE"

# 3. Pull logs back
echo "📥 Pulling logs..."
rsync -avz "${NPU_HOST}:${NPU_PATH}/logs/" "$PROJECT_ROOT/logs/"

ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"
echo "✅ Done. Full log: $LOG_FILE"
echo "   Latest symlink: $LATEST_LINK"
