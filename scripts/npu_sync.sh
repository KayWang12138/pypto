#!/bin/bash
# Sync local project to NPU server
# Usage: ./scripts/npu_sync.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

NPU_HOST="${NPU_HOST:-npu}"
NPU_PATH="${NPU_PATH:-~/pypto-multi}"

EXCLUDE_FILE="$PROJECT_ROOT/.rsync-exclude"
if [ ! -f "$EXCLUDE_FILE" ]; then
    cat > "$EXCLUDE_FILE" << 'EOF'
.git/
__pycache__/
*.pyc
.DS_Store
node_modules/
.venv/
logs/
*.log
*.swp
.opencode/
.claude/
build/
dist/
EOF
    echo "📝 Created default .rsync-exclude"
fi

echo "🔄 Syncing to ${NPU_HOST}:${NPU_PATH} ..."
rsync -avz --delete --exclude-from="$EXCLUDE_FILE" ./ "${NPU_HOST}:${NPU_PATH}/"
echo "✅ Sync complete"
