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
# 单元测试 — 不需要 NPU, 不烧 LLM. 跑三件事:
#   1. cheat_detector 5 类 fixture (clean / multi_jit / no_pypto / no_jit / suspicious)
#      每类 verdict 必须与 fixtures/README.md 表格一致.
#   2. verifier verify 子命令的静态 cheat warning: 用 multi_jit 跑 verify,
#      必须保留 cheat_check=cheat 和 cheat_gate_warning, 后续 correctness 可继续裁定.
#   3. opencode_exporter 必须容忍 opencode export stdout 中的非法 UTF-8 字节.
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
    'integration/benchmark/opencode_exporter.py',
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

# ---------- 4. opencode_exporter 非 UTF-8 输出容错 ----------
section "4. opencode_exporter 非 UTF-8 输出容错"

FAKE_OPENCODE="${BENCHMARK_LOG_DIR}/fake_opencode_invalid_utf8.py"
cat > "${FAKE_OPENCODE}" <<'PY'
#!/usr/bin/env python3
import sys

if sys.argv[1:3] == ["export", "ses_invalidutf8"]:
    payload = (
        b'{"info":{"id":"ses_invalidutf8","title":"invalid utf8"},'
        b'"messages":[{"info":{"role":"assistant","time":{}},'
        b'"parts":[{"type":"text","text":"bad byte: \xe2 end"}]}]}'
    )
    sys.stdout.buffer.write(payload)
    raise SystemExit(0)

if sys.argv[1:3] == ["export", "ses_truncatedutf8"]:
    payload = (
        b'{"info":{"id":"ses_truncatedutf8","title":"truncated utf8"},'
        b'"messages":[{"info":{"role":"assistant","time":{}},'
        b'"parts":[{"type":"text","text":"cut byte: \xe2'
    )
    sys.stdout.buffer.write(payload)
    raise SystemExit(0)

raise SystemExit(2)
PY
chmod +x "${FAKE_OPENCODE}"

EXPORT_MD="${BENCHMARK_LOG_DIR}/invalid_utf8_session.md"
TRUNCATED_MD="${BENCHMARK_LOG_DIR}/truncated_utf8_session.md"
python3 - <<PY || fail "opencode_exporter 非 UTF-8 输出容错失败"
from pathlib import Path

from integration.benchmark.opencode_exporter import (
    append_export_result_to_log,
    export_session_to_markdown,
)

result = export_session_to_markdown(
    session_id="ses_invalidutf8",
    output_file=Path("${EXPORT_MD}"),
    opencode_bin="${FAKE_OPENCODE}",
    cwd=Path("${PYPTO_ROOT}"),
)
assert result.ok, result.to_dict()
text = Path("${EXPORT_MD}").read_text(encoding="utf-8")
assert "bad byte:" in text, text
print(result.to_dict())
append_export_result_to_log(Path("${BENCHMARK_LOG_DIR}/verifier.log"), result, label="verifier")
status_file = Path("${BENCHMARK_LOG_DIR}/verifier_session_export.json")
assert status_file.exists(), "missing verifier_session_export.json"
assert "exported" in status_file.read_text(encoding="utf-8")

truncated = export_session_to_markdown(
    session_id="ses_truncatedutf8",
    output_file=Path("${TRUNCATED_MD}"),
    opencode_bin="${FAKE_OPENCODE}",
    cwd=Path("${PYPTO_ROOT}"),
)
assert truncated.status == "error", truncated.to_dict()
assert "JSON" in truncated.message, truncated.to_dict()
print(truncated.to_dict())
append_export_result_to_log(Path("${BENCHMARK_LOG_DIR}/pypto_run.log"), truncated, label="pypto")
error_file = Path("${BENCHMARK_LOG_DIR}/pypto_session_export.log")
assert error_file.exists(), "missing pypto_session_export.log"
assert "JSON" in error_file.read_text(encoding="utf-8")
PY
pass "opencode export stdout 非法 UTF-8 不再抛 UnicodeDecodeError"

section "test-unit ALL PASSED"
echo "  详细报告: ${BENCHMARK_LOG_DIR}"
