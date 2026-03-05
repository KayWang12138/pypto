---
name: pypto-environment-setup
description: "PyPTO 环境安装与环境问题修复，包括CANN、torch_npu、编译工具链、第三方依赖和PyPTO编译运行等。Triggers: PyPTO environment setup, CANN install, torch_npu, NPU environment, Ascend toolkit, compile PyPTO, build PyPTO, NPU driver, prepare_env, diagnose environment, fix import error, torch_npu import fail, DT_FP8E8M0, pto-isa, ASCEND_HOME_PATH, npu-smi, softmax verify, pip dependency conflict"
---

# PyPTO Environment Setup

## 约定

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
```


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
# Step 1.1: 检查 PYPTO_REPO 环境变量
echo "PYPTO_REPO: ${PYPTO_REPO:-未设置}"

# Step 1.2: 检查当前目录是否为 PyPTO 仓库
[ -f "pyproject.toml" ] && [ -d "framework" ] && echo "✓ 当前目录是 PyPTO 仓库"
```

**PyPTO 仓库查找逻辑**（按优先级自动执行）：
1. 环境变量 `$PYPTO_REPO`
2. 当前目录（含 `pyproject.toml` + `framework/`）
3. `$PWD` 下搜索（最多 3 层），找到多个时提示用户选择
4. 未找到 → 自动克隆：

```bash
git clone https://gitcode.com/cann/pypto.git "$PWD/pypto"
export PYPTO_REPO="$PWD/pypto"
```

**运行环境诊断**：
```bash
# Step 1.3: 进入 skill 目录运行诊断脚本
cd ${SKILL_DIR:-.opencode/skills/pypto-environment-setup}
python3 scripts/diagnose_env.py --checklist
```

**通过标准**：清单所有项 ✅ OK。⚠️/❌ 项需修复后再继续。

### Step 2: 决策分支

- 0 issues → 跳至 Step 4 验证
- 有 issues → 向用户展示问题清单，获得确认后进入 Step 3
- 用户拒绝 → 停止，报告当前状态

### Step 3: 按类别修复

只修复缺失项，不改动已正常组件。

| 问题类别 | 修复操作 | 失败回滚 |
|---------|---------|---------|
| **NPU 环境 + CANN 缺失** | 分步执行：<br>1. `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2\|a3>`<br>2. `bash tools/prepare_env.sh --quiet --type=third_party`<br>3. `bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend} 2>&1 \| tee prepare_env.cann.log` | 检查网络连通性和目录写入权限；见 `troubleshooting.md` |
| **编译工具链缺失** (cmake/gcc/make/g++/ninja/pip3) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps` | 回退到手动 `apt-get install`；检查 apt 源配置 |
| **第三方源码包缺失** (json/libboundscheck) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party` | 检查网络对 cann-src-third-party 的可达性 |

> `--device-type` 由 Step 1 检测自动确定（910B→a2，910C→a3）。
> 手动安装/编译细节见 [📋 prepare_environment.md](references/prepare_environment.md)。
> 遇到报错见 [🔧 troubleshooting.md](references/troubleshooting.md)。

### Step 4: 验证（必须执行）

任何安装/修复后必须通过 softmax 验证。

```bash
# Step 4.1: 加载 CANN 环境
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh

# Step 4.2: 检查可用的 NPU 卡
npu-smi info

# Step 4.3: 设置 NPU 设备 ID（根据 Step 4.2 选择空闲卡）
export TILE_FWK_DEVICE_ID=0

# Step 4.4: 设置 PTO-ISA 路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux

# Step 4.5: 进入 PyPTO 仓库
cd ${PYPTO_REPO:-$PWD}
```

**编译安装**：
```bash
python3 build_ci.py -f python3 --clean --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall -q
```

**运行测试**：
```bash
# NPU 模式
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu

# SIM 模式（非 NPU 环境）
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode sim
```

⚠️ **注意**：NPU 环境必须使用 NPU 模式通过验证。

**通过标准**：退出码 `0`，输出 `Softmax test passed`。

**失败时**：重新运行 Step 1 诊断 → 对照 [🔧 troubleshooting.md](references/troubleshooting.md) 排查。

### Step 5: 完成报告

环境配置完成后，输出以下报告供用户确认。

**配置摘要**：
```bash
echo "=== PyPTO 环境配置报告 ==="
echo "CANN 版本: $(cat \"${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/version.info\" 2>/dev/null || echo '未知')"
echo "NPU 设备: $(npu-smi info 2>/dev/null | grep 'Chip Name' | head -1 || echo '未检测到')"
echo "TILE_FWK_DEVICE_ID: ${TILE_FWK_DEVICE_ID}"
echo "PTO_TILE_LIB_CODE_PATH: ${PTO_TILE_LIB_CODE_PATH}"
echo "PYPTO_REPO: ${PYPTO_REPO:-$PWD}"
```

**验证结果**：
- Softmax 测试（NPU 模式）：✅ 通过

**持久化配置（可选）**：
```bash
cat >> ~/.bashrc << 'EOF'
# PyPTO 环境配置
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
EOF
source ~/.bashrc
```

⚠️ `TILE_FWK_DEVICE_ID` 需根据 `npu-smi info` 输出修改。
## 📚 参考文件

| File | Contents |
|------|----------|
| [📋 prepare_environment.md](references/prepare_environment.md) | 安装 CANN/torch_npu/pto-isa、编译 PyPTO |
| [🔧 troubleshooting.md](references/troubleshooting.md) | 安装/导入/运行报错 |
## 外部参考

- PyPTO: https://gitcode.com/cann/pypto
- CANN 安装指南: https://www.hiascend.com/document/redirect/CannCommunityInstSoftware
- Ascend Extension for PyTorch: https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html
