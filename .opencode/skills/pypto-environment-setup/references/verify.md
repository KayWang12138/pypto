# 环境验证运行手册（最终版）

## 强制验收（置顶）

任何安装/修复后必须通过此验收。NPU 环境必须用 NPU 模式验证，非 NPU 环境使用 sim 模式。

标准变量约定（本手册统一使用）：

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
PYPTO_REPO=${PYPTO_REPO:-$PWD/pypto}
PTO_ISA_DIR=${PTO_ISA_DIR:-$PWD/pto-isa}
```

> 环境变量完整配置见 `SKILL.md` § "关键环境变量速查表"，详细说明见 `references/prepare_environment.md` § "环境变量配置"。

NPU 模式（必须用于 NPU 环境）：

```bash
# 加载环境（变量配置见 SKILL.md § "关键环境变量速查表"）
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || { echo "ERROR: CANN set_env.sh not found"; exit 1; }

export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-${PTO_ISA_DIR:-$PWD/pto-isa}}"
export PYTHONPATH="${PYPTO_REPO:-$PWD/pypto}/python:$PYTHONPATH"
export TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID:-0}

python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py"
```

SIM 模式（无需 NPU）：

```bash
# 加载环境（变量配置见 SKILL.md § "关键环境变量速查表"）
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || { echo "ERROR: CANN set_env.sh not found"; exit 1; }

export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-${PTO_ISA_DIR:-$PWD/pto-isa}}"
export PYTHONPATH="${PYPTO_REPO:-$PWD/pypto}/python:$PYTHONPATH"

python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py" --run_mode sim
```

验收通过标准：

- 退出码为 `0`
- 输出包含 `Softmax test passed`
- `Max difference` 在阈值内（通常 `3e-3`）

## 中间检查步骤

以下步骤用于定位问题，属于中间检查，不替代上方强制验收。

1) 运行诊断脚本（检查清单模式）

```bash
python3 "$SKILL_DIR/scripts/diagnose_env.py" --checklist
```

2) NPU 可见性检查（仅 NPU 环境）

```bash
npu-smi info
```

3) CANN 环境加载（自动发现 `CANN_ENV_SH`）

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH}/*/set_env.sh ${ASCEND_INSTALL_PATH}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || { echo "ERROR: CANN set_env.sh not found"; exit 1; }
```

4) Conda 激活（按需）

```bash
CONDA_HOME=${CONDA_HOME:-/opt/conda}
source "${CONDA_HOME}/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:?请设置 CONDA_ENV}"
```

5) Python 最小检查（`torch` + `torch_npu`）

```bash
python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu_available:", bool(getattr(torch, "npu").is_available()))
PY
```

## 仿真环境验证

仿真环境不需要 NPU，直接执行顶部“强制验收（置顶）”中的 SIM 模式命令即可。

## PyPTO 安装前提

若尚未安装 PyPTO，请先执行：

```bash
PYPTO_REPO=${PYPTO_REPO:-$PWD/pypto}
cd "${PYPTO_REPO}" && pip install -e .
```
