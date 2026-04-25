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
# 真集成测试 — KernelBench → pypto 7 阶段 → skill 验证 → batch 报告 全程不跳.
# 默认跑 19_ReLU 一个正例; 默认 verifier-mode=opencode (走 LLM skill).
# 默认 mode=performance, 这样 test-all 能覆盖到性能链路.
#
# 两种模式:
#   FULL=1 (默认)    — 不带 --skip-pypto-gen, 让 pypto-op-orchestrator 真跑
#                      Stage 1-7 算子开发. 单 case ~50 min (大头是 LLM + Stage 7
#                      perf-tune 跑 NPU profile). 这是真正的端到端集成.
#   FULL=0 (cheap)   — 带 --skip-pypto-gen, 复用现成 custom/<op>/ 产物, 只测
#                      verifier 这一段. 单 case ~3 min. 给开发期回归用.
#
# 用法:
#   bash integration/benchmark/scripts/local/test-integration.sh             # FULL
#   FULL=0 bash integration/benchmark/scripts/local/test-integration.sh      # cheap
#   CASES=19_ReLU,20_LeakyReLU bash ...                                      # 多 case
#   VERIFIER_MODE=direct bash ...                                            # 不烧 LLM 验证
#   BENCHMARK_LOG_DIR=/path bash ...                                         # 自定义 log
#   CONCURRENCY=4 bash ...                                                  # 并行 case 数 (默认 1)
#
# 退出 0 表示 1/1 (或 N/N) 通过.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_npu

CASES="${CASES:-19_ReLU}"
FULL="${FULL:-1}"
VERIFIER_MODE="${VERIFIER_MODE:-opencode}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-1500}"
PYPTO_TIMEOUT="${PYPTO_TIMEOUT:-7200}"
MODE="${MODE:-performance}"
OPENCODE_MODEL="${OPENCODE_MODEL:-}"
SKIP_STAGE7_PERF_TUNE="${SKIP_STAGE7_PERF_TUNE:-0}"
VERIFY_RTOL="${VERIFY_RTOL:-}"
VERIFY_ATOL="${VERIFY_ATOL:-}"
CONCURRENCY="${CONCURRENCY:-1}"

if [ "${VERIFIER_MODE}" = "opencode" ]; then
  require_opencode
fi
if [ "${FULL}" = "1" ]; then
  require_opencode
fi

REPORT_DIR="${BENCHMARK_LOG_DIR}/report"
LOG_DIR="${BENCHMARK_LOG_DIR}/logs"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

if [ "${FULL}" = "0" ]; then
  section "test-integration: ${CASES} concurrency=${CONCURRENCY} (cheap, --skip-pypto-gen, verifier=${VERIFIER_MODE})"
else
  section "test-integration: ${CASES} concurrency=${CONCURRENCY} (FULL, pypto 7-stage + verifier=${VERIFIER_MODE})"
  echo "  WARN: 单 case 约 50 min (含 Stage 7 性能调优 + LLM 调用)" >&2
fi

export CASES FULL VERIFIER_MODE SKILL_TIMEOUT PYPTO_TIMEOUT MODE OPENCODE_MODEL
export SKIP_STAGE7_PERF_TUNE VERIFY_RTOL VERIFY_ATOL CONCURRENCY
export BENCHMARK_MONITOR_DIR="${BENCHMARK_LOG_DIR}/monitor_state"
MONITOR_CMD="BENCHMARK_MONITOR_DIR=${BENCHMARK_MONITOR_DIR} python3 -m integration.benchmark.monitor monitor"

echo "  command:"
printf '    %s\n' python3 -m integration.benchmark.monitor
echo
echo "============================================================"
echo "  MONITOR COMMAND"
echo "============================================================"
echo "  ${MONITOR_CMD}"
echo "============================================================"
echo

# 真跑
python3 -m integration.benchmark.monitor 2>&1 | tee "${BENCHMARK_LOG_DIR}/batch.log"

# 验报告
section "解读 summary.json"
python3 - <<PY || fail "summary 不符 1/1 通过: ${REPORT_DIR}/summary.json"
import json
d = json.load(open("${REPORT_DIR}/summary.json"))
totals = d["totals"]
meta = d["meta"]
print(f"  verifier_mode    = {meta.get('verifier_mode')}")
print(f"  validator_agent  = {meta.get('validator_agent')}")
print(f"  mode             = {meta.get('mode')}")
print(f"  total            = {totals['total']}")
print(f"  success          = {totals['success']} ({totals['success_rate']*100:.0f}%)")
print(f"  correctness pass = {totals['correctness']['pass']}/{totals['total']}")
dur = totals["duration_sec"]
print(f"  pypto_total_sec  = {dur['pypto_total']:.1f}")
print(f"  verify_total_sec = {dur['verify_total']:.1f}")
print(f"  wall_total_sec   = {dur['wall_total']:.1f}")

assert totals["success"] == totals["total"], "存在失败 case"
assert totals["correctness"]["fail"] == 0, "精度失败"
PY
pass "summary 1/1 通过"

section "test-integration ALL PASSED"
echo "  summary    : ${REPORT_DIR}/summary.md"
echo "  json       : ${REPORT_DIR}/summary.json"
echo "  batch log  : ${BENCHMARK_LOG_DIR}/batch.log"
echo "  sessions   :"
find "${REPORT_DIR}" -maxdepth 2 -type f -name '*_session.md' -print | sed 's/^/    /'
