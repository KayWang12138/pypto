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
# opencode skill 端到端 (合法 ReLU 算子) — 真烧 LLM, 真跑 NPU. 验证:
#   1. verifier_runner --verifier-mode opencode 能 spawn pypto-kernel-validator
#      agent + 加载 pypto-kernel-validate skill.
#   2. SKILL 4 步走完, skill_report.json 落 op_dir/.skill_validate/, final_verdict=PASS.
#   3. LLM 语义审阅 S1-S9 全 pass (合法 ReLU 没有任何作弊形态).
#
# 单次跑约 3-5 min (大头是 LLM 调用). 需要远端 / 本机配好 opencode + 大模型 token.
#
# 用法:
#   bash integration/benchmark/scripts/local/test-skill-legit.sh
#   OP_NAME=Softmax bash ...                  # 跑别的算子 (须有 custom/<op>/)
#   SKILL_TIMEOUT=900 bash ...                # 调超时 (默认 1500s)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_npu
require_opencode

OP_NAME="${OP_NAME:-ReLU}"
OP_DIR="${PYPTO_ROOT}/custom/${OP_NAME}"
TASK_DESC="${OP_DIR}/task_desc.py"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-1500}"

section "test-skill-legit: ${OP_NAME} (opencode + LLM, NPU device=${TILE_FWK_DEVICE_ID})"

if [ ! -f "${OP_DIR}/${OP_NAME}_pypto_impl.py" ]; then
  echo "[FAIL] ${OP_DIR}/${OP_NAME}_pypto_impl.py 不存在" >&2
  echo "       先跑 test-integration.sh 让 pypto-op-orchestrator 真生成" >&2
  exit 1
fi
if [ ! -f "${TASK_DESC}" ]; then
  fail "${TASK_DESC} 不存在"
fi

# 清理 skill 报告, 让结果可复现
rm -rf "${OP_DIR}/.skill_validate"

RUNNER_LOG="${BENCHMARK_LOG_DIR}/${OP_NAME}_skill_runner.log"
section "spawn opencode + pypto-kernel-validator (timeout=${SKILL_TIMEOUT}s)"
python3 -m integration.benchmark.verifier_runner "${OP_NAME}" \
    --op-dir "${OP_DIR}" \
    --task-desc-file "${TASK_DESC}" \
    --verifier-mode opencode \
    --mode correctness \
    --device "${TILE_FWK_DEVICE_ID}" \
    --log-file "${RUNNER_LOG}" \
    --skill-timeout "${SKILL_TIMEOUT}" >"${BENCHMARK_LOG_DIR}/${OP_NAME}_skill_stdout.log" 2>&1 \
  || true

REPORT="${OP_DIR}/.skill_validate/skill_report.json"
if [ ! -f "${REPORT}" ]; then
  echo "[FAIL] skill agent 没产出 ${REPORT}" >&2
  echo "  runner log: ${RUNNER_LOG}" >&2
  echo "  stdout log: ${BENCHMARK_LOG_DIR}/${OP_NAME}_skill_stdout.log" >&2
  exit 1
fi

section "解读 skill_report.json"
python3 - <<PY || fail "skill_report.json verdict 不符"
import json
d = json.load(open("${REPORT}"))
final = d.get("final_verdict")
script_v = d["cheat_check_script"]["verdict"]
sem_v = d["cheat_check_semantic"]["verdict"]
correctness = d["correctness"]["status"]

print(f"  final_verdict        = {final}")
print(f"  cheat_check_script   = {script_v}")
print(f"  cheat_check_semantic = {sem_v}")
print(f"  correctness          = {correctness}")
print()
print("  --- LLM 语义审阅 S1-S9 ---")
for it in d["cheat_check_semantic"]["items"]:
    sid, st, ev = it["id"], it["status"], it.get("evidence", "")[:100]
    print(f"    {sid:<3} {st:<10} {ev}")

assert final == "PASS", f"final_verdict={final}"
assert script_v == "pass", f"cheat_check_script={script_v}"
assert sem_v == "pass", f"cheat_check_semantic={sem_v}"
assert correctness == "passed", f"correctness={correctness}"
PY
pass "final_verdict=PASS, S1-S9 全 pass, correctness=passed"

section "test-skill-legit ALL PASSED"
echo "  skill_report : ${REPORT}"
echo "  runner log   : ${RUNNER_LOG}"
