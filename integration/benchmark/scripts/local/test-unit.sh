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
# 单元测试 — 不需要 NPU, 不烧 LLM. 跑两件事:
#   1. cheat_detector 5 类 fixture (clean / multi_jit / no_pypto / no_jit / suspicious)
#      每类 verdict 必须与 fixtures/README.md 表格一致.
#   2. verifier verify 子命令的静态 cheat warning: 用 multi_jit 跑 verify,
#      必须保留 cheat_check=cheat 和 cheat_gate_warning, 后续 correctness 可继续裁定.
#
# 用法 (cwd 任意均可):
#   bash integration/benchmark/scripts/local/test-unit.sh
#
# 退出 0 表示全过, 任一失败立刻退出非 0.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

section "test-unit: cheat_detector + verify cheat-gate (无 NPU 无 LLM)"

# ---------- 1. AST + import 自检 ----------
section "1. AST + import 自检"
python3 -c "
import ast
for f in [
    'integration/benchmark/verifier/__main__.py',
    'integration/benchmark/verifier/cheat_detector.py',
    'integration/benchmark/verifier/pypto_adapter.py',
    'integration/benchmark/verifier_runner.py',
    'integration/benchmark/run_kernelbench.py',
    'integration/benchmark/case_loader.py',
]:
    ast.parse(open(f).read(), f)
print('AST OK')
from integration.benchmark.verifier import cheat_detector
from integration.benchmark.verifier import __main__ as vmain
print('import OK')
"
pass "AST + import"

# ---------- 2. 5 类 cheat_detector 分类 ----------
section "2. cheat_detector 5 类 fixture 分类"

EXPECT_CLEAN_VERDICT=pass
EXPECT_MULTI_JIT_VERDICT=cheat
EXPECT_NO_PYPTO_VERDICT=cheat
EXPECT_NO_JIT_VERDICT=cheat
EXPECT_SUSPICIOUS_VERDICT=suspicious

for case in clean multi_jit no_pypto no_jit suspicious; do
  fixture="${FIXTURES_DIR}/${case}"
  if [ ! -f "${fixture}/relu_impl.py" ]; then
    fail "fixture ${case} 缺 relu_impl.py"
  fi
  # cheat-check 对 cheat verdict 故意 exit 1, 用 || true 不杀脚本
  out=$(python3 -m integration.benchmark.verifier cheat-check "${fixture}" --op-name relu 2>/dev/null || true)
  verdict=$(echo "${out}" | python3 -c "import json, sys; print(json.load(sys.stdin)['verdict'])")
  expect_var="EXPECT_$(echo "${case}" | tr '[:lower:]' '[:upper:]')_VERDICT"
  expect="${!expect_var}"
  if [ "${verdict}" = "${expect}" ]; then
    pass "${case} -> ${verdict}"
  else
    echo "${out}" > "${BENCHMARK_LOG_DIR}/cheat_${case}.json"
    fail "${case}: 期望 verdict=${expect}, 实际 ${verdict}; 完整输出 ${BENCHMARK_LOG_DIR}/cheat_${case}.json"
  fi
done

# ---------- 3. verify 子命令静态 cheat warning ----------
section "3. verify 子命令静态 cheat warning (multi_jit 应保留 warning 并继续裁定)"

GATE_TASK="${BENCHMARK_LOG_DIR}/gate_task_desc.py"
cat > "${GATE_TASK}" <<'PY'
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
def get_inputs():
    return [torch.randn(16, 16384)]
def get_init_inputs():
    return []
PY

GATE_REPORT="${BENCHMARK_LOG_DIR}/gate_verify.json"
python3 -m integration.benchmark.verifier verify \
    "${FIXTURES_DIR}/multi_jit" \
    --op-name relu \
    --task-desc "${GATE_TASK}" \
    --mode correctness \
    --json-out "${GATE_REPORT}" >/dev/null 2>&1 || true

python3 - <<PY || fail "cheat warning 行为不符: 见 ${GATE_REPORT}"
import json
d = json.load(open("${GATE_REPORT}"))
assert d["cheat_check"]["verdict"] == "cheat", f"cheat_check={d['cheat_check']['verdict']}"
assert d["cheat_gate_warning"], "cheat_gate_warning missing"
assert d["correctness"]["status"] != "skipped", f"correctness={d['correctness']['status']}"
assert d["verdict_machine"] in {"failed_correctness", "pass"}, f"verdict_machine={d['verdict_machine']}"
print("cheat warning 行为符合预期")
PY
pass "cheat_check=cheat, warning 已保留, correctness 已继续执行"

section "test-unit ALL PASSED"
echo "  详细报告: ${BENCHMARK_LOG_DIR}"
