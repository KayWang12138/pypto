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
# 离线冒烟 — 不烧 LLM, 不占 NPU, 仅校验桥接代码 + 数据集就位.
#
# 用法:
#   bash benchmark/scripts/smoke_test.sh
#
# 退出 0 表示 4 项全过; 任一失败立刻退出非 0.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYPTO_ROOT="$(cd "${BRIDGE_ROOT}/.." && pwd)"
KB_ROOT="${BRIDGE_ROOT}/.cache/KernelBench/KernelBench"

cd "${PYPTO_ROOT}"

echo "==================================================="
echo "benchmark smoke test"
echo "  pypto root  : ${PYPTO_ROOT}"
echo "  bridge root : ${BRIDGE_ROOT}"
echo "  kernelbench : ${KB_ROOT}"
echo "==================================================="

if [ ! -d "${KB_ROOT}/level1" ]; then
  echo "FAIL: ${KB_ROOT}/level1 不存在; 先跑 download_kernelbench.sh" >&2
  exit 1
fi

# ------ 4.1 imports ------
python3 -c "
from benchmark import case_loader, pypto_runner, verifier_runner, run_kernelbench, report
print('[4.1] OK 5 modules importable')
"

# ------ 4.2 case_loader ------
# 上游 KernelBench 文件名混合大小写: 19_ReLU.py / 20_LeakyReLU.py / 23_Softmax.py / ...
# derive_op_name 去掉前缀数字保留剩余原样, 所以 '19_ReLU' -> op_name='ReLU'
rm -rf /tmp/benchmark_smoke
python3 -m benchmark.case_loader \
    "${KB_ROOT}/level1/19_ReLU.py" \
    --write /tmp/benchmark_smoke
echo "  produced files:"
ls /tmp/benchmark_smoke/ReLU/ | sed 's/^/    /'
echo "  SPEC.md head:"
head -8 /tmp/benchmark_smoke/ReLU/SPEC.md | sed 's/^/    /'
echo "[4.2] OK case_loader"

# ------ 4.3 discover_cases ------
# 测两种匹配: 纯数字前缀 ('19') + 完整 stem ('20_LeakyReLU')
export KB_ROOT
python3 - <<'PY'
from pathlib import Path
import os
from benchmark.run_kernelbench import discover_cases
root = Path(os.environ['KB_ROOT']) / 'level1'
cases = discover_cases(root, requested=['19', '20_LeakyReLU'])
names = [p.name for p in cases]
assert names == ['19_ReLU.py', '20_LeakyReLU.py'], names
print('[4.3] OK discover_cases ->', names)
PY

# ------ 4.4 CLI --help ------
python3 -m benchmark.run_kernelbench --help > /tmp/cli_help.txt 2>&1
echo "[4.4] OK --help (前 8 行):"
head -8 /tmp/cli_help.txt | sed 's/^/    /'

echo
echo "==================================================="
echo "[OK] smoke test passed (4.1-4.4)"
echo "==================================================="
