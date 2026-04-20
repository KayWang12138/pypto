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
# direct 模式 verify — 真跑 NPU, 不烧 LLM. 跑 ReLU 算子 (要求 custom/ReLU/ 已就绪),
# 验证两件事:
#   1. correctness: KernelVerifier 在 NPU 上跑通, 期望 verdict_machine=pass.
#   2. performance: pypto_adapter swimlane 走单 kernel 路径, 期望 swimlane n=1 +
#      cheat_multi_kernel=False + 给出真实 gen/base us 数.
#
# 这是给开发者快速验证 KernelVerifier + pypto_adapter 没改坏的回归测试.
# 算子产物缺失时给排错提示, 不会自动跑 pypto 工作流 (那是 test-integration.sh
# 的事).
#
# 用法:
#   bash integration/benchmark/scripts/local/test-direct.sh
#   OP_NAME=Softmax bash ...test-direct.sh        # 跑别的 op (须有 custom/<op>/)
#   BENCHMARK_LOG_DIR=/path bash ...              # 自定义 log 目录

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_npu

OP_NAME="${OP_NAME:-ReLU}"
OP_DIR="${PYPTO_ROOT}/custom/${OP_NAME}"
TASK_DESC="${OP_DIR}/task_desc.py"

section "test-direct: ${OP_NAME} (NPU device=${TILE_FWK_DEVICE_ID})"

# ---------- 0. 前置: 算子产物 + task_desc ----------
section "0. 前置检查"
if [ ! -f "${OP_DIR}/${OP_NAME}_impl.py" ]; then
  echo "[FAIL] ${OP_DIR}/${OP_NAME}_impl.py 不存在." >&2
  echo "       需要先有 PyPTO 算子产物. 可选方案:" >&2
  echo "       1) 若已有 KernelBench 风格的 task_desc, 手写一个 ${OP_NAME}_impl.py." >&2
  echo "       2) 跑 test-integration.sh 让 pypto-op-orchestrator 真生成 (耗时 ~50 min)." >&2
  exit 1
fi
if [ ! -f "${OP_DIR}/${OP_NAME}_pypto_impl.py" ]; then
  fail "${OP_DIR}/${OP_NAME}_pypto_impl.py 不存在 (KernelBench 桥接入口)"
fi
if [ ! -f "${TASK_DESC}" ]; then
  echo "[FAIL] ${TASK_DESC} 不存在." >&2
  echo "       用 case_loader 生成: " >&2
  echo "       python3 -m integration.benchmark.case_loader \\" >&2
  echo "         integration/benchmark/.cache/KernelBench/KernelBench/level1/19_ReLU.py \\" >&2
  echo "         --write custom" >&2
  exit 1
fi
pass "前置: ${OP_DIR} 含 _impl.py + _pypto_impl.py + task_desc.py"

# ---------- 1. correctness ----------
section "1. direct correctness"
COR_REPORT="${BENCHMARK_LOG_DIR}/${OP_NAME}_direct_correctness.json"
python3 -m integration.benchmark.verifier verify \
    "${OP_DIR}" \
    --op-name "${OP_NAME}" \
    --task-desc "${TASK_DESC}" \
    --mode correctness \
    --device-id "${TILE_FWK_DEVICE_ID}" \
    --json-out "${COR_REPORT}" >"${BENCHMARK_LOG_DIR}/${OP_NAME}_direct_correctness.log" 2>&1 \
  || true

python3 - <<PY || fail "correctness 验证不符: ${COR_REPORT}"
import json
d = json.load(open("${COR_REPORT}"))
assert d["verdict_machine"] == "pass", f"verdict_machine={d['verdict_machine']}"
assert d["cheat_check"]["verdict"] == "pass", f"cheat_check={d['cheat_check']['verdict']}"
assert d["correctness"]["status"] == "passed", f"correctness={d['correctness']['status']}"
PY
pass "correctness PASS, verdict_machine=pass"

# ---------- 2. performance ----------
section "2. direct performance (swimlane 单 kernel 路径)"
PERF_REPORT="${BENCHMARK_LOG_DIR}/${OP_NAME}_direct_performance.json"
python3 -m integration.benchmark.verifier verify \
    "${OP_DIR}" \
    --op-name "${OP_NAME}" \
    --task-desc "${TASK_DESC}" \
    --mode performance \
    --device-id "${TILE_FWK_DEVICE_ID}" \
    --json-out "${PERF_REPORT}" >"${BENCHMARK_LOG_DIR}/${OP_NAME}_direct_performance.log" 2>&1 \
  || true

python3 - <<PY || fail "performance 验证不符: ${PERF_REPORT}"
import json
d = json.load(open("${PERF_REPORT}"))
p = d["performance"]
assert p["status"] == "passed", f"perf.status={p['status']}"
assert p.get("cheat_multi_kernel") is False, f"cheat_multi_kernel={p.get('cheat_multi_kernel')}"
assert p.get("gen_time_us") and p["gen_time_us"] > 0, f"gen_time_us={p.get('gen_time_us')}"
assert p.get("base_time_us") and p["base_time_us"] > 0, f"base_time_us={p.get('base_time_us')}"
print(f"  gen={p['gen_time_us']:.2f}us base={p['base_time_us']:.2f}us speedup={p.get('speedup', 0):.2f}x")
PY
pass "performance PASS, swimlane 单 kernel 路径生效"

section "test-direct ALL PASSED"
echo "  详细报告: ${BENCHMARK_LOG_DIR}"
