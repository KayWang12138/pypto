---
name: pypto-environment-setup
description: "PyPTO 环境安装与环境问题修复，包括CANN、torch_npu、编译工具链、第三方依赖和PyPTO编译运行等。Triggers: PyPTO environment setup, CANN install, torch_npu, NPU environment, Ascend toolkit, compile PyPTO, build PyPTO, NPU driver, prepare_env, diagnose environment, fix import error, torch_npu import fail, DT_FP8E8M0, pto-isa, ASCEND_HOME_PATH, npu-smi, softmax verify, pip dependency conflict"
---

# PyPTO Environment Setup

## 约定

```bash
# Ascend 安装根路径（可由用户覆盖）
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
```

- `PYPTO_REPO`：由诊断脚本自动检测（依次尝试 `$HOME/pypto` → 当前目录下 find）。若未找到，尝试从 GitCode 克隆；克隆失败则请用户手动提供路径或设置 `GITCODE_TOKEN`。
- CANN 路径检测优先级：`ASCEND_HOME_PATH` → `ASCEND_TOOLKIT_HOME` → `ASCEND_OPP_PATH` 反推 → Fallback 扫描 `cann-*`（详见 `references/prepare_environment.md`）。
- `$SKILL_DIR`：由 agent 运行时自动注入的环境变量。指向当前 skill 的根目录（即本 `pypto-environment-setup/` 目录）。文档中所有 `$SKILL_DIR/scripts/...` 的引用均依赖此变量。手动执行时需自行设置，例如：`export SKILL_DIR=/path/to/pypto-environment-setup`。

## ⛔ 隐私保护

> ⚠️ **禁止在屏幕、日志、错误信息中打印 `GITCODE_TOKEN` 环境变量**

克隆私有仓库时使用 Token 认证，请确保 Token 仅存储在安全位置（环境变量或配置文件），不要在终端输出中暴露。
## 工作流程

> 💡 **网络前提**：只要网络连接正常，本技能即具备完整安装 PyPTO 环境的能力（包括 CANN、torch_npu、编译工具链、第三方依赖等）。
>
> ⚠️ **环境保护**：本技能在任何安装操作前，必须先检测当前环境状态，向用户展示已有组件及版本，获得明确确认后才执行变更。绝不覆盖或破坏已正常工作的环境。

### Step 1: 环境检测

运行诊断脚本获取环境状态：

```bash
# 快速诊断（推荐，跳过编译工具链/第三方依赖/Python依赖检查）
python3 "$SKILL_DIR/scripts/diagnose_env.py" --fast --checklist

# 深度诊断（完整检查所有项目）
python3 "$SKILL_DIR/scripts/diagnose_env.py" --checklist
```

> 💡 **快速 vs 深度**：`--fast` 模式跳过 cmake/gcc/json/libboundscheck/requirements.txt 等编译相关检查，适合已安装 PyPTO 的用户快速验证环境。深度诊断用于首次安装或编译问题排查。

脚本会自动检测以下项目并输出确认清单（每项标注 ✅ OK / ⚠️ 缺失 / ❌ 异常）：

脚本会自动检测以下所有项目并输出确认清单（每项标注 ✅ OK / ⚠️ 缺失 / ❌ 异常）：

| 检测项 | 检测方式 |
|--------|----------|
| cmake | `cmake --version`，需 >= 3.16.3 |
| gcc | `gcc --version`，需 >= 7.3.1 |
| make | `make --version`，存在即可 |
| g++ | `g++ --version`，需 >= 7.3.1 |
| ninja | `ninja --version`，存在即可 |
| pip3 | `pip3 --version`，存在即可 |
| python3 | `platform.python_version()`，需 >= 3.9.5 |
| JSON for Modern C++ | 检查 `third_party/` 或 `pypto_download/` 下是否有源码包 (v3.11.3) |
| libboundscheck | 检查同上 (v1.1.16) |
| NPU 环境 | 五级瀑布检测：PCI sysfs → /dev/davinci* → npu-smi → ctypes ACL → torch_npu/python acl |
| NPU 芯片型号 | 自动识别芯片名、家族、代际（A1/A2/A3/A3+）、SoC 版本号，映射为 `device_type`（a2/a3 等） |
| NPU 设备数量 | 多级置信度：PCI < davinci < npu-smi < ACL < torch_npu |
| PyPTO 仓库路径 | 自动探测常用路径，或由 `$PYPTO_REPO` 指定 |
| CANN 安装路径 | 优先读取 `ASCEND_HOME_PATH` → `ASCEND_TOOLKIT_HOME` → `ASCEND_OPP_PATH` 反推；Fallback 扫描 `cann-*` 目录 |
| CANN 版本 | 读取 `compiler/version.info` 或从目录名提取 |
| CANN 组件 | 检查 toolkit、ops（opp 目录）是否存在 |
| Python 版本 | `sys.executable` + `platform.python_version()` |
| torch | `import torch` + 版本号 |
| torch_npu | `import torch_npu` + 版本号 |
| pypto | `import pypto` |
| pto-isa | `$PTO_TILE_LIB_CODE_PATH` 及 `include/pto` 目录 |
| Python 依赖 | 读取 `$PYPTO_REPO/python/requirements.txt`，逐包检查是否已安装及版本是否满足 |
**将清单完整展示给用户，明确询问是否按该检测结果继续，若需要执行安装则必须包含验证流程。**

若 PyPTO 仓库未找到：
```bash
# 尝试克隆
git clone https://gitcode.com/cann/pypto.git "${PYPTO_REPO:-$PWD/pypto}"
# 若需认证
git clone https://${GITCODE_TOKEN}@gitcode.com/cann/pypto.git "${PYPTO_REPO:-$PWD/pypto}"
```
克隆失败时，告知用户需要设置 `GITCODE_TOKEN` 或手动提供仓库路径。

### Step 2: 决策分支

```
检测结果
├─ 0 issues（全部 ✅）
│  └─ 🎯 Happy Path → 跳至 Step 4 验证
│
├─ 有 issues → 向用户展示问题清单，获得确认后进入 Step 3
│
└─ 用户拒绝 → 停止，报告当前状态
```

### Step 3: 按类别修复（获得用户确认后执行）

只安装或配置缺失项，不改动已正常组件。按下表匹配 issue 类别执行对应操作：

| 问题类别 | 修复操作 | 失败回滚 |
|---------|---------|---------|
| **NPU 环境 + CANN 缺失** | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=all --device-type=<a2\|a3> --install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}` | 检查网络连通性和目录写入权限；见 `troubleshooting.md` |
| **编译工具链缺失** (cmake/gcc/make/g++/ninja/pip3) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps` | 回退到手动 `apt-get install`；检查 apt 源配置 |
| **第三方源码包缺失** (json/libboundscheck) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party` | 检查网络对 cann-src-third-party 的可达性 |
| **Python 依赖缺失/版本不足** | `pip3 install -r $PYPTO_REPO/python/requirements.txt` | 见 `troubleshooting.md` § "pip 依赖冲突" |
| **torch/torch_npu 导入失败** (NPU 环境) | 见 `references/prepare_environment.md` § torch_npu 安装 | 见 `troubleshooting.md` § "torch_npu 导入失败" |
| **pypto 未安装** | `cd $PYPTO_REPO && pip install -e .` | 见 `troubleshooting.md` § "DT_FP8E8M0" |
| **pto-isa 缺失/路径未设** | `git clone https://gitcode.com/cann/pto-isa.git $PTO_TILE_LIB_CODE_PATH` + 设置环境变量 | 检查 GITCODE_TOKEN；见 `troubleshooting.md` § "pto-isa 版本不匹配" |

> `--device-type` 的值由 Step 1 检测的 NPU 芯片型号自动确定（910B→a2，910C→a3）。
> `--install-path` 支持用户自定义 CANN 安装路径（默认 `/usr/local/Ascend`）。
> 手动安装细节参考 `references/prepare_environment.md`。

### Step 4: 验证（必须执行）

完成任何安装/修复变更后（或 Happy Path 下直接到达），**必须**执行 softmax 示例验证：

```
Step 3 完成 / Happy Path
  └─ 运行 softmax 验证
      ├─ PASS ✅ → 交付成功
      └─ FAIL ❌ → 重新运行 Step 1 诊断 → 对照 troubleshooting.md 排查
```

- NPU 环境：
  ```bash
  python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py"
  ```
- sim 模式（非 NPU 环境）：
  ```bash
  python3 "${PYPTO_REPO:-$PWD/pypto}/examples/02_intermediate/operators/softmax/softmax.py" --run_mode sim
  ```
- NPU 环境必须用 NPU 模式验证成功才算通过。
- 需要完整验证链路时读取：`references/verify.md`。

## prepare_env.sh

`prepare_env.sh` 位于 PyPTO 仓库的 `tools/prepare_env.sh`，不在本技能内硬编码。
参数说明执行 `cd "$PYPTO_REPO" && bash tools/prepare_env.sh --help`，手动安装细节参考 `references/prepare_environment.md`。

## PyPTO 安装

- 源码安装：`cd "$PYPTO_REPO" && pip install -e .`（开发）或 `pip install .`（生产）
- PyPI 安装：`pip install pypto`
- Docker：查看 PyPTO 仓库 `docker/README.md`
- 详细编译选项参考：`references/build.md`

> ⚠️ **遇到问题时**重点参考 PyPTO 仓库内的官方文档：
> - `$PYPTO_REPO/docs/install/prepare_environment.md` — 环境准备
> - `$PYPTO_REPO/docs/install/build_and_install.md` — 编译与安装

## ⚠️ 关键环境变量速查表

> **缺少任何一项都会导致 import 失败或运行异常。这是最高频的环境问题根因。**
> 完整配置说明及变量列表见 `references/prepare_environment.md` § "环境变量配置"。

| 变量 / 操作 | 设置方式 | 必须性 |
|------------|----------|--------|
| CANN 环境加载 | `source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/latest/set_env.sh"` | 每次新 shell 必须 |
| `PTO_TILE_LIB_CODE_PATH` | `export PTO_TILE_LIB_CODE_PATH="${PTO_ISA_DIR:-$PWD/pto-isa}"` | 编译/运行必须 |
| `TILE_FWK_DEVICE_ID` | `export TILE_FWK_DEVICE_ID=1` | NPU 模式必须 |
| `PYTHONPATH` | `export PYTHONPATH="${PYPTO_REPO}/python:$PYTHONPATH"` | 开发模式推荐 |

### 完整设置模板（推荐写入 ~/.bashrc）

```bash
source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/latest/set_env.sh"
export PYTHONPATH="${PYPTO_REPO:-$PWD/pypto}/python:$PYTHONPATH"
export PTO_TILE_LIB_CODE_PATH="${PTO_ISA_DIR:-$PWD/pto-isa}"
export TILE_FWK_DEVICE_ID=1  # 可选，NPU 模式需要
```

> 💡 最小验证方法同 Step 4，请参考上方验证流程。

## 诊断工具

```bash
# 快速检查（推荐，~15s）
python3 "$SKILL_DIR/scripts/diagnose_env.py" --fast --checklist

# 完整检查（~50s）
python3 "$SKILL_DIR/scripts/diagnose_env.py" --checklist
```

> 更多输出格式和选项请运行 `python3 "$SKILL_DIR/scripts/diagnose_env.py" --help` 查看。

> `--fast` 模式适用于已安装 PyPTO 的环境快速验证；首次安装或遇到编译问题时使用不带 `--fast` 的完整模式。

## 参考文件导航

| 文件 | 何时读取 |
|------|----------|
| `references/prepare_environment.md` | 安装/配置 CANN、torch_npu、pto-isa |
| `references/verify.md` | 环境验证（含 softmax 示例） |
| `references/troubleshooting.md` | 安装/导入/运行错误 |
| `references/build.md` | 从源码编译 PyPTO（含高级编译选项） |

## 外部参考

- PyPTO: https://gitcode.com/cann/pypto
- CANN 安装指南: https://www.hiascend.com/document/redirect/CannCommunityInstSoftware
- Ascend Extension for PyTorch: https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html
