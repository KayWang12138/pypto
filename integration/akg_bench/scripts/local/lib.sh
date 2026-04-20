#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#
# 内部 helper, 只供 scripts/local/*.sh source. 不要直接执行.
#
# 提供:
#   - 路径推断 (PYPTO_ROOT / BRIDGE_ROOT / SCRIPTS_DIR / FIXTURES_DIR)
#   - conda env 自动激活 (CONDA_ENV 默认 torch2.6, 可被外部覆盖)
#   - 日志目录 (AKG_BENCH_LOG_DIR, 默认 /tmp/akg_bench_test_<ts>)
#   - 错误处理 (set -euo pipefail + ERR trap 打印行号)
#   - 块标题 (section "...") + PASS/FAIL 标记 (pass / fail)

set -euo pipefail

# ---- 路径 ----
_LIB_FILE="${BASH_SOURCE[0]}"
SCRIPTS_DIR="$(cd "$(dirname "${_LIB_FILE}")/.." && pwd)"
FIXTURES_DIR="${SCRIPTS_DIR}/local/fixtures"
BRIDGE_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
PYPTO_ROOT="$(cd "${BRIDGE_ROOT}/../.." && pwd)"

# ---- 日志目录 ----
AKG_BENCH_LOG_DIR="${AKG_BENCH_LOG_DIR:-/tmp/akg_bench_test_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${AKG_BENCH_LOG_DIR}"

# ---- conda 自动激活 (若 python3 找不到 pypto, 也尝试激活) ----
_activate_conda() {
  local env_name="${CONDA_ENV:-torch2.6}"
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${env_name}" ]; then
    return
  fi
  local conda_sh
  for conda_sh in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "${conda_sh}" ]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      conda activate "${env_name}" 2>/dev/null && return
    fi
  done
  echo "[lib.sh] WARN: 没找到 conda 或激活 ${env_name} 失败; 继续用当前 python3" >&2
}
_activate_conda

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
trap 'rc=$?; echo "[lib.sh] ERROR at ${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}:${BASH_LINENO[0]:-?} (rc=${rc})" >&2; exit ${rc}' ERR

# ---- 通用前置 ----
require_npu() {
  if [ -z "${TILE_FWK_DEVICE_ID:-}" ]; then
    export TILE_FWK_DEVICE_ID=0
    echo "[lib.sh] TILE_FWK_DEVICE_ID 未设, 默认 0"
  fi
  if [ ! -d /usr/local/Ascend ] && [ ! -d /home/HwHiAiUser/Ascend ]; then
    echo "[lib.sh] WARN: 未发现 /usr/local/Ascend, NPU 测试可能失败" >&2
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
echo "[lib.sh] LOG_DIR   =${AKG_BENCH_LOG_DIR}"
echo "[lib.sh] python3   =$(command -v python3)  ($(python3 --version 2>&1))"
