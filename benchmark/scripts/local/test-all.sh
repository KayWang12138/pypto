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
# 本机一键跑全部测试. 默认按 4 步走, 任一步失败立即退出:
#   1. test-unit          (无 NPU 无 LLM, ~10 s)
#   2. test-direct        (NPU, 不烧 LLM, ~30 s) — 须有 custom/ReLU/ 现成产物
#   3. test-skill-legit   (NPU + LLM, ~3-5 min) — 须有 custom/ReLU/ 现成产物
#   4. test-skill-cheat   (NPU + LLM, ~3-5 min) — 自动部署 fixture
#   5. test-integration   (NPU + LLM 全套, ~50 min) — 真 pypto 7 阶段, 默认带
#
# 通过环境变量裁剪:
#   SKIP_NPU=1               跳所有需要 NPU 的步骤 (只剩 test-unit).
#   SKIP_LLM=1               跳所有 opencode/skill 步骤 (剩 unit + direct).
#   SKIP_INTEGRATION=1       跳 test-integration (FULL pypto 工作流, 50 min).
#                            适合 CI / 快速回归; 默认值视 BUDGET 而定:
#                              BUDGET=fast (~5 min)  -> SKIP_INTEGRATION=1
#                              BUDGET=full (~60 min) -> SKIP_INTEGRATION=0 (默认)
#   BUDGET=fast              一键设 SKIP_INTEGRATION=1.
#   BENCHMARK_LOG_DIR=/path  统一 log 根, 各步骤自动落子目录.
#
# 用法:
#   bash benchmark/scripts/local/test-all.sh             # FULL ~60 min
#   BUDGET=fast bash ...test-all.sh                                  # ~10 min, 跳 integration
#   SKIP_NPU=1 bash ...test-all.sh                                   # 只跑 unit (CI 离线)
#
# 退出 0 表示所有未跳过的步骤都过.

set -euo pipefail
set -o pipefail   # 保证 `bash step.sh | tee` 的 exit code 跟随 bash, 不被 tee 吞

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-/tmp/benchmark_test_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${BENCHMARK_LOG_DIR}"

if [ "${BUDGET:-}" = "fast" ]; then
  SKIP_INTEGRATION="${SKIP_INTEGRATION:-1}"
fi
SKIP_NPU="${SKIP_NPU:-0}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_INTEGRATION="${SKIP_INTEGRATION:-0}"

declare -a STEPS_OK=()
declare -a STEPS_SKIP=()
declare -a STEPS_FAIL=()

run_step() {
  local name="$1"
  local script="$2"
  local skip_reason="${3:-}"

  if [ -n "${skip_reason}" ]; then
    echo
    echo "############################################################"
    echo "##  SKIP: ${name}  (${skip_reason})"
    echo "############################################################"
    STEPS_SKIP+=("${name} (${skip_reason})")
    return 0
  fi

  echo
  echo "############################################################"
  echo "##  RUN: ${name}"
  echo "##  log dir: ${BENCHMARK_LOG_DIR}"
  echo "############################################################"

  local step_log="${BENCHMARK_LOG_DIR}/${name}.log"
  if BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR}/${name}" \
     bash "${script}" 2>&1 | tee "${step_log}"; then
    STEPS_OK+=("${name}")
    echo "##  ${name} PASSED"
  else
    STEPS_FAIL+=("${name}")
    echo "##  ${name} FAILED, see ${step_log}" >&2
    print_summary
    exit 1
  fi
}

print_summary() {
  echo
  echo "============================================================"
  echo "  test-all summary"
  echo "============================================================"
  for s in "${STEPS_OK[@]}";   do echo "  [PASS] ${s}";   done
  for s in "${STEPS_SKIP[@]}"; do echo "  [SKIP] ${s}";   done
  for s in "${STEPS_FAIL[@]}"; do echo "  [FAIL] ${s}";   done
  echo
  echo "  log root: ${BENCHMARK_LOG_DIR}"
}

# ---------- 走步骤 ----------
run_step "test-unit"        "${SCRIPT_DIR}/test-unit.sh"            ""
run_step "test-direct"      "${SCRIPT_DIR}/test-direct.sh"          "$([ "${SKIP_NPU}" = "1" ] && echo 'SKIP_NPU=1')"
run_step "test-skill-legit" "${SCRIPT_DIR}/test-skill-legit.sh"     "$([ "${SKIP_NPU}" = "1" ] && echo 'SKIP_NPU=1' || ([ "${SKIP_LLM}" = "1" ] && echo 'SKIP_LLM=1'))"
run_step "test-skill-cheat" "${SCRIPT_DIR}/test-skill-cheat.sh"     "$([ "${SKIP_NPU}" = "1" ] && echo 'SKIP_NPU=1' || ([ "${SKIP_LLM}" = "1" ] && echo 'SKIP_LLM=1'))"
run_step "test-integration" "${SCRIPT_DIR}/test-integration.sh"     "$([ "${SKIP_NPU}" = "1" ] && echo 'SKIP_NPU=1' || ([ "${SKIP_LLM}" = "1" ] && echo 'SKIP_LLM=1' || ([ "${SKIP_INTEGRATION}" = "1" ] && echo 'SKIP_INTEGRATION=1')))"

print_summary
echo
echo "ALL DONE."
