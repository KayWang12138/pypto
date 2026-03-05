---
name: pypto-environment-setup
description: "PyPTO 环境安装与环境问题修复，包括CANN、torch_npu、编译工具链、第三方依赖和PyPTO编译运行等。Triggers: PyPTO environment setup, CANN install, torch_npu, NPU environment, Ascend toolkit, compile PyPTO, build PyPTO, NPU driver, prepare_env, diagnose environment, fix import error, torch_npu import fail, DT_FP8E8M0, pto-isa, ASCEND_HOME_PATH, npu-smi, softmax verify, pip dependency conflict"
---

# PyPTO Environment Setup

## 约定

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
```

- `PYPTO_REPO`：由诊断脚本自动检测（`$HOME/pypto` → 当前目录 find）。未找到则尝试 GitCode 克隆；失败请用户手动提供路径或设置 `GITCODE_TOKEN`。Most importantly, verify current directory is a PyPTO git repository.
- `$SKILL_DIR`：由 agent 运行时自动注入的环境变量。指向当前 skill 的根目录（即本 `pypto-environment-setup/` 目录）。文档中所有 `$SKILL_DIR/scripts/...` 的引用均依赖此变量。手动执行时需自行设置，例如：`export SKILL_DIR=/path/to/pypto-environment-setup`。
- **默认版本**：CANN 8.5.0 + PyTorch 2.6.0 + torch_npu 2.6.0

## ⛔ 隐私保护

> ⚠️ **禁止在屏幕、日志、错误信息中打印 `GITCODE_TOKEN` 环境变量**
>
> 克隆私有仓库时使用 Token 认证，请确保 Token 仅存储在安全位置（环境变量或配置文件），不要在终端输出中暴露。

## 工作流程

> ⚠️ **环境保护**：任何安装前必须先检测当前状态，向用户展示已有组件及版本，获得确认后才执行变更。

### Step 1: 环境检测

```bash
# 快速诊断（推荐，跳过编译工具链等检查）
python3 scripts/diagnose_env.py --fast --checklist

# 深度诊断（完整检查，首次安装或编译问题时使用）
python3 scripts/diagnose_env.py --checklist
```

脚本会输出确认清单（每项标注 ✅ OK / ⚠️ 缺失 / ❌ 异常）。**将清单完整展示给用户，获得确认后继续。**

若 PyPTO 仓库未找到：
```bash
git clone https://gitcode.com/cann/pypto.git "${PYPTO_REPO:-$PWD/pypto}"
# 若需认证
git clone https://${GITCODE_TOKEN}@gitcode.com/cann/pypto.git "${PYPTO_REPO:-$PWD/pypto}"
```

### Step 2: 决策分支

- 0 issues → 跳至 Step 4 验证
- 有 issues → 向用户展示问题清单，获得确认后进入 Step 3
- 用户拒绝 → 停止，报告当前状态

### Step 3: 按类别修复

只修复缺失项，不改动已正常组件。

| 问题类别 | 修复操作 | 失败回滚 |
|---------|---------|---------|
| **NPU 环境 + CANN 缺失** | 分步执行：<br>1. `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2\|a3>`<br>2. `bash tools/prepare_env.sh --quiet --type=third_party`<br>3. `bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}` | 检查网络连通性和目录写入权限；见 `troubleshooting.md` |
| **编译工具链缺失** (cmake/gcc/make/g++/ninja/pip3) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps` | 回退到手动 `apt-get install`；检查 apt 源配置 |
| **第三方源码包缺失** (json/libboundscheck) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party` | 检查网络对 cann-src-third-party 的可达性 |
| **Python 依赖缺失/版本不足** | `pip3 install -r $PYPTO_REPO/python/requirements.txt` | 见 `troubleshooting.md` § "pip 依赖冲突" |
| **torch/torch_npu 导入失败** (NPU 环境) | 见 `references/prepare_environment.md` § torch_npu 安装 | 见 `troubleshooting.md` § "torch_npu 导入失败" |
| **pypto 未安装** | `cd $PYPTO_REPO && pip install -e .` | 见 `troubleshooting.md` § "DT_FP8E8M0" |
| **pto-isa 缺失/路径未设** | `git clone https://gitcode.com/cann/pto-isa.git $PTO_TILE_LIB_CODE_PATH` + 设置环境变量 | 检查 GITCODE_TOKEN；见 `troubleshooting.md` § "pto-isa 版本不匹配" |

> `--device-type` 由 Step 1 检测自动确定（910B→a2，910C→a3）。
> 手动安装/编译细节见 [📋 prepare_environment.md](references/prepare_environment.md)。
> 遇到报错见 [🔧 troubleshooting.md](references/troubleshooting.md)。

### Step 4: 验证（必须执行）

任何安装/修复后必须通过 softmax 验证。NPU 环境必须用 NPU 模式。

```bash
# 加载环境
CANN_ENV_SH=$(ls -1 ${ASCEND_INSTALL_PATH}/*/set_env.sh ${ASCEND_INSTALL_PATH}/*/*/set_env.sh 2>/dev/null | head -1)
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || { echo "ERROR: CANN set_env.sh not found"; return 1; }

# 设置 pto-isa 路径
PTO_CANN_DIR="${ASCEND_HOME_PATH:-}/aarch64-linux"
if [ -d "$PTO_CANN_DIR/include/pto" ]; then
  export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-$PTO_CANN_DIR}"
else
  export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-${PTO_ISA_DIR:-$PWD/pto-isa}}"
fi
export PYTHONPATH="${PYPTO_REPO:-$PWD/pypto}/python:$PYTHONPATH"
export TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID:-0}

# NPU 模式
python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py"
# SIM 模式（非 NPU 环境）
python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py" --run_mode sim
```

通过标准：退出码 `0`，输出 `Softmax test passed`，`Max difference` ≤ `3e-3`。

失败时：重新运行 Step 1 诊断 → 对照 [🔧 troubleshooting.md](references/troubleshooting.md) 排查。

## 📚 参考文件

| File | Contents |
|------|----------|
| [📋 prepare_environment.md](references/prepare_environment.md) | 安装 CANN/torch_npu/pto-isa、编译 PyPTO |
| [🔧 troubleshooting.md](references/troubleshooting.md) | 安装/导入/运行报错 |
## 外部参考

- PyPTO: https://gitcode.com/cann/pypto
- CANN 安装指南: https://www.hiascend.com/document/redirect/CannCommunityInstSoftware
- Ascend Extension for PyTorch: https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html
