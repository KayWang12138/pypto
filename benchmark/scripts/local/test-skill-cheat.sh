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
# opencode skill 端到端 (反作弊样例 relu_cheat) — 真烧 LLM. 验证:
#   1. cheat_check_script 不再检测 jit 数量, 不因此判 cheat.
#   2. performance profile 应通过 CHEAT_MULTI_KERNEL / cheat_multi_kernel=true 抓到多 kernel.
#   3. final_verdict=FAIL_CHEAT, correctness 可正常执行.
#
# 使用 fixtures/pypto_op_cheat/ 下预置的 relu_cheat 算子, 拷贝到 custom/ 并清理.
# 单次 ~3-5 min.
#
# 用法: bash benchmark/scripts/local/test-skill-cheat.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_npu
require_opencode

OP_NAME="relu_cheat"
SRC_FIX="${FIXTURES_DIR}/pypto_op_cheat"
OP_DIR="${PYPTO_ROOT}/custom/${OP_NAME}"
TASK_DESC="${OP_DIR}/task_desc.py"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-1500}"
SKILL_RETRY="${SKILL_RETRY:-2}"
SKILL_RETRY_INTERVAL="${SKILL_RETRY_INTERVAL:-600}"

section "test-skill-cheat: ${OP_NAME} (期望 final_verdict=FAIL_CHEAT)"

# ---------- 0. 预备 ----------
section "0. 部署 fixture 到 custom/${OP_NAME}/"
rm -rf "${OP_DIR}"
mkdir -p "${OP_DIR}"
cp "${SRC_FIX}"/* "${OP_DIR}/"
ls "${OP_DIR}"
pass "fixture deployed"

# ---------- 1. 跑 skill ----------
RUNNER_LOG="${BENCHMARK_LOG_DIR}/${OP_NAME}_skill_runner.log"
section "1. spawn opencode + pypto-kernel-validator (timeout=${SKILL_TIMEOUT}s)"
python3 -m benchmark.verifier_runner "${OP_NAME}" \
    --op-dir "${OP_DIR}" \
    --task-desc-file "${TASK_DESC}" \
    --verifier-mode opencode \
    --mode performance \
    --device "${TILE_FWK_DEVICE_ID}" \
    --log-file "${RUNNER_LOG}" \
    --skill-timeout "${SKILL_TIMEOUT}" \
    --skill-retry "${SKILL_RETRY}" \
    --skill-retry-interval "${SKILL_RETRY_INTERVAL}" >"${BENCHMARK_LOG_DIR}/${OP_NAME}_skill_stdout.log" 2>&1 \
  || true

REPORT="${OP_DIR}/.skill_validate/skill_report.json"
if [ ! -f "${REPORT}" ]; then
  echo "[FAIL] skill agent 没产出 ${REPORT}" >&2
  echo "  runner log: ${RUNNER_LOG}" >&2
  exit 1
fi

# ---------- 2. 验报告 ----------
section "2. 解读 skill_report.json (重点: runtime profile 抓 multi-kernel)"
python3 - <<PY || fail "skill_report.json verdict 不符: ${REPORT}"
import json
d = json.load(open("${REPORT}"))
final = d.get("final_verdict")
script_v = d["cheat_check_script"]["verdict"]
sem_v = d["cheat_check_semantic"]["verdict"]
correctness = d["correctness"]["status"]
performance = d["performance"]["status"]
cheat_multi = d["performance"].get("cheat_multi_kernel")

print(f"  final_verdict        = {final}")
print(f"  cheat_check_script   = {script_v}")
print(f"  cheat_check_semantic = {sem_v}")
print(f"  correctness          = {correctness}")
print(f"  performance          = {performance}")
print(f"  cheat_multi_kernel   = {cheat_multi}")
print()
print("  --- LLM 语义审阅 ---")

failed_ids = []
for it in d["cheat_check_semantic"]["items"]:
    sid, st, ev = it["id"], it["status"], it.get("evidence", "")[:120]
    mark = "[X]" if st in ("fail", "cheat") else "[ ]"
    print(f"    {mark} {sid:<3} {st:<10} {ev}")
    if st in ("fail", "cheat"):
        failed_ids.append(sid)

assert final == "FAIL_CHEAT", f"final_verdict={final}"
assert script_v == "pass", f"cheat_check_script={script_v}"
assert correctness == "passed", f"correctness={correctness}"
assert cheat_multi is True, f"cheat_multi_kernel={cheat_multi}"

print()
print(f"  LLM 自抓 fail/cheat 项: {failed_ids}")
PY
pass "final_verdict=FAIL_CHEAT, runtime profile 抓到 cheat_multi_kernel=true"

section "test-skill-cheat ALL PASSED"
echo "  skill_report : ${REPORT}"
echo "  runner log   : ${RUNNER_LOG}"
echo
echo "  fixture 留在 custom/${OP_NAME}/, 跑完后可删:"
echo "    rm -rf ${OP_DIR}"
