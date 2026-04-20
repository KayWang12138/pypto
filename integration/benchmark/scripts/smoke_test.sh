#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#
# 离线冒烟 — 不烧 LLM, 不占 NPU, 仅校验桥接代码 + 数据集就位.
#
# 用法:
#   cd <pypto repo root>
#   bash integration/benchmark/scripts/smoke_test.sh
#
# 退出 0 表示 5 项全过; 任一失败立刻退出非 0.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYPTO_ROOT="$(cd "${BRIDGE_ROOT}/../.." && pwd)"
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
from integration.benchmark import case_loader, pypto_runner, verifier_runner, run_kernelbench, report
print('[4.1] OK 5 modules importable')
"

# ------ 4.2 case_loader ------
# 上游 KernelBench 文件名混合大小写: 19_ReLU.py / 20_LeakyReLU.py / 23_Softmax.py / ...
# derive_op_name 去掉前缀数字保留剩余原样, 所以 '19_ReLU' -> op_name='ReLU'
rm -rf /tmp/benchmark_smoke
python3 -m integration.benchmark.case_loader \
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
from integration.benchmark.run_kernelbench import discover_cases
root = Path(os.environ['KB_ROOT']) / 'level1'
cases = discover_cases(root, requested=['19', '20_LeakyReLU'])
names = [p.name for p in cases]
assert names == ['19_ReLU.py', '20_LeakyReLU.py'], names
print('[4.3] OK discover_cases ->', names)
PY

# ------ 4.4 旧 numpy 子目录布局必须被拒绝 ------
rm -rf /tmp/fake_old_layout
mkdir -p /tmp/fake_old_layout/level1/19_relu
echo "import numpy" > /tmp/fake_old_layout/level1/19_relu/19_relu_numpy.py

python3 - <<'PY'
from pathlib import Path
from integration.benchmark.run_kernelbench import discover_cases
msg = None
try:
    discover_cases(Path('/tmp/fake_old_layout/level1'))
except ValueError as e:
    msg = str(e)
ok = msg is not None and '扁平' in msg
print('[4.4]', 'OK' if ok else 'FAIL', '旧子目录布局被拒绝 ->', (msg or 'no error').splitlines()[0])
if not ok:
    raise SystemExit(1)
PY

rm -rf /tmp/fake_old_layout

# ------ 4.5 CLI --help ------
python3 -m integration.benchmark.run_kernelbench --help > /tmp/cli_help.txt 2>&1
echo "[4.5] OK --help (前 8 行):"
head -8 /tmp/cli_help.txt | sed 's/^/    /'

echo
echo "==================================================="
echo "[OK] smoke test passed (4.1-4.5)"
echo "==================================================="
