#!/usr/bin/env bash
# Run automated layout validation for custom/<operator>/ (no NPU).
# Safe to run from any cwd; resolves repository root via git or directory traversal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$SCRIPT_DIR/../../../../.." && pwd)"
exec python3 "$SCRIPT_DIR/scripts/validate_custom_kernel_layout.py" --repo-root "$REPO_ROOT"
