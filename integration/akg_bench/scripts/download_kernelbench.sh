#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#
# 下载上游 KernelBench (PyTorch 原版) 并固定到指定 commit, 落地到
# pypto/integration/akg_bench/.cache/KernelBench/.
#
# 用法:
#   bash pypto/integration/akg_bench/scripts/download_kernelbench.sh
#
# 自定义下载位置:
#   KERNELBENCH_DIR=/path/to/elsewhere bash .../download_kernelbench.sh
#
# 升级 commit:
#   修改下方 KERNELBENCH_COMMIT 常量, 并同步 case_loader.py docstring 里的引用.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(dirname "${SCRIPT_DIR}")"
DEFAULT_TARGET_DIR="${BRIDGE_ROOT}/.cache/KernelBench"

KERNELBENCH_REPO_URL="https://github.com/ScalingIntelligence/KernelBench.git"
KERNELBENCH_COMMIT="21fbe5a642898cd60b8f60c7aefb43d475e11f33"

TARGET_DIR="${KERNELBENCH_DIR:-${DEFAULT_TARGET_DIR}}"

if ! command -v git &> /dev/null; then
  echo "ERROR: git not found in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "[$(date +%H:%M:%S)] KernelBench 已存在, 更新远端信息: ${TARGET_DIR}"
  git -C "${TARGET_DIR}" fetch --tags origin
else
  if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: 目录 ${TARGET_DIR} 已存在且非空, 但不是 git 仓; 请手动清理后重试." >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] 克隆 KernelBench 到 ${TARGET_DIR}..."
  git clone "${KERNELBENCH_REPO_URL}" "${TARGET_DIR}"
fi

if ! git -C "${TARGET_DIR}" rev-parse --verify "${KERNELBENCH_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] 拉取目标 commit ${KERNELBENCH_COMMIT}..."
  git -C "${TARGET_DIR}" fetch origin "${KERNELBENCH_COMMIT}" || true
fi

if git -C "${TARGET_DIR}" rev-parse --verify "${KERNELBENCH_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] checkout ${KERNELBENCH_COMMIT}..."
  git -C "${TARGET_DIR}" -c advice.detachedHead=false checkout "${KERNELBENCH_COMMIT}"
else
  echo "ERROR: 未能找到目标 commit ${KERNELBENCH_COMMIT}" >&2
  exit 1
fi

echo ""
echo "[OK] KernelBench ready"
echo "  path:   ${TARGET_DIR}"
echo "  commit: ${KERNELBENCH_COMMIT}"

LEVELS_DIR="${TARGET_DIR}/KernelBench"
if [ -d "${LEVELS_DIR}" ]; then
  echo "  levels:"
  for d in "${LEVELS_DIR}"/level*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name '*.py' | wc -l)
    echo "    - $(basename "$d") (${n} cases)"
  done
fi
