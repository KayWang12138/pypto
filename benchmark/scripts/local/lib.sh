#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# 内部 helper, 只供 scripts/local/*.sh source. 不要直接执行.
#
# 提供:
#   - 路径推断 (PYPTO_ROOT / BRIDGE_ROOT / SCRIPTS_DIR / FIXTURES_DIR)
#   - 日志目录 (BENCHMARK_LOG_DIR, 默认 /tmp/benchmark_test_<ts>)
#   - 错误处理 (set -euo pipefail + ERR trap 打印行号)
#   - 块标题 (section "...") + PASS/FAIL 标记 (pass / fail)

set -euo pipefail

# ---- 路径 ----
_LIB_FILE="${BASH_SOURCE[0]}"
SCRIPTS_DIR="$(cd "$(dirname "${_LIB_FILE}")/.." && pwd)"
FIXTURES_DIR="${SCRIPTS_DIR}/local/fixtures"
BRIDGE_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
PYPTO_ROOT="$(cd "${BRIDGE_ROOT}/.." && pwd)"

# ---- 日志目录 ----
BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-/tmp/benchmark_test_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${BENCHMARK_LOG_DIR}"

# ---- 输出 ----
section() {
  echo
  echo "============================================================"
  echo "  $*"
  echo "============================================================"
}

pass() { echo "  [PASS] $*"; }
# fail 直接 exit 1 — 避免 return 1 被 set -e 二次捕获到 ERR trap, 也保证调用栈干净.
fail() { echo "  [FAIL] $*" >&2; exit 1; }

# ---- 错误位置打印 (set -u 下要给 BASH_SOURCE[1] 兜底, 否则顶层 caller 时 unbound) ----
# rc >= 128 是信号退出码 (130=SIGINT, 137=SIGKILL, 143=SIGTERM),
# 属于用户主动中断, 不打印 ERROR.
trap 'rc=$?; if [ $rc -ge 128 ]; then exit $rc; fi; echo "[lib.sh] ERROR at ${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}:${BASH_LINENO[0]:-?} (rc=${rc})" >&2; exit ${rc}' ERR

# ---- 通用前置 ----
require_npu() {
  if [ -n "${DEVICE:-}" ]; then
    export TILE_FWK_DEVICE_ID="${DEVICE}"
    echo "[lib.sh] DEVICE=${DEVICE} -> TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID}"
  elif [ -z "${TILE_FWK_DEVICE_ID:-}" ]; then
    export TILE_FWK_DEVICE_ID=0
    echo "[lib.sh] TILE_FWK_DEVICE_ID 未设, 默认 0"
  fi
}

require_opencode() {
  if ! command -v opencode >/dev/null 2>&1; then
    echo "[lib.sh] FAIL: 未找到 opencode CLI; opencode 模式跳过" >&2
    return 1
  fi
}

# ---- 进 pypto cwd 是大多数 import 的前提 ----
cd "${PYPTO_ROOT}"

echo "[lib.sh] PYPTO_ROOT=${PYPTO_ROOT}"
echo "[lib.sh] LOG_DIR   =${BENCHMARK_LOG_DIR}"
echo "[lib.sh] python3   =$(command -v python3)  ($(python3 --version 2>&1))"
