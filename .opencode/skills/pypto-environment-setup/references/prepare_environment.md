# 环境准备（CANN + pto-isa + 第三方依赖）

## 安装顺序

官方顺序（来源：`prepare_environment.md`）：
1. 驱动/固件（见官方 CANN 安装指南，本技能不执行）
2. CANN toolkit 包
3. CANN ops 包（按芯片型号选 A2/A3）
4. pto-isa 源码
5. 加载环境变量
6. PyTorch + Ascend Extension for PyTorch

⚠️ 本技能不执行驱动/固件操作，仅提供官方指南链接：https://www.hiascend.com/document/redirect/CannCommunityInstSoftware

## 使用 prepare_env.sh 完整安装（推荐）

> CANN 包安装推荐使用 `prepare_env.sh` 进行完整安装，它会自动处理 toolkit、ops、第三方依赖等全部组件。

```bash
cd "$PYPTO_REPO"

# 完整安装（推荐，包含 CANN + 第三方依赖）
bash tools/prepare_env.sh --quiet --type=all --device-type=a2 --install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}  # 或 a3

# 仅安装 CANN（toolkit + ops）
bash tools/prepare_env.sh --quiet --type=cann --device-type=a2 --install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}  # 或 a3
```

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| --type | str | 是 | deps, cann, third_party, all |
| --device-type | str | 是 | a2, a3 |
| --install-path | str | 否 | CANN 安装路径 |
| --download-path | str | 否 | 下载路径 |
| --with-install-driver | bool | 否 | 是否下载驱动固件，默认 false |
| --quiet | bool | 否 | 静默模式，减少输出（执行prepare_env.sh脚本时必须加上此参数） |

## 手动安装 CANN

> 手动安装步骤（toolkit 包和 ops 包的下载 URL、安装命令、芯片选型）详见：
> `$PYPTO_REPO/docs/install/prepare_environment.md` § "手动安装"
>
> 版本：CANN 8.5.0，需按芯片型号选择 A2(910B) 或 A3(910C) 的 ops 包。
> 安装路径统一使用 `--install-path=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}`。

## pto-isa 获取

### 方法一：git clone 源码（推荐，优先）

原因：PyPTO 编译/代码生成优先使用环境变量 `PTO_TILE_LIB_CODE_PATH` 指定的 pto-isa；当 pto-isa 与 PyPTO 头文件版本不匹配时，常见现象是出现 `pto::TROWEXPANDADD`/`pto::TROWEXPANDMAX` 等符号缺失的编译错误（见 `references/troubleshooting.md`）。

```bash
PTO_ISA_DIR=${PTO_ISA_DIR:-$PWD/pto-isa}
mkdir -p "$PTO_ISA_DIR"

# 建议使用源码方式，便于和 PyPTO 分支同步升级
git clone https://gitcode.com/cann/pto-isa.git "$PTO_ISA_DIR"

export PTO_TILE_LIB_CODE_PATH="$PTO_ISA_DIR"
test -d "$PTO_TILE_LIB_CODE_PATH/include/pto" && echo OK
```

### 方法二：安装 .run 包（备用）

> .run 包下载地址和安装命令详见：
> `$PYPTO_REPO/docs/install/prepare_environment.md` § "获取pto-isa源码"
>
> 注：使用 .run 包安装时，头文件位置可能不在一个固定路径；如遇到头文件版本不匹配，优先切换到"源码方式"。

## 环境变量配置

### set_env.sh 导出的变量

CANN 安装完成后，`source set_env.sh` 会导出以下关键变量：

| 变量 | 示例值 | 说明 |
|------|--------|------|
| `ASCEND_HOME_PATH` | `${ASCEND_INSTALL_PATH}/cann-<version>` | CANN 版本根目录（PyPTO 项目主要依赖此变量） |
| `ASCEND_TOOLKIT_HOME` | `${ASCEND_INSTALL_PATH}/cann-<version>` | 与 HOME_PATH 相同 |
| `ASCEND_OPP_PATH` | `${ASCEND_INSTALL_PATH}/cann-<version>/opp` | OPP 算子包路径 |
| `ASCEND_AICPU_PATH` | `${ASCEND_INSTALL_PATH}/cann-<version>` | AICPU 路径 |
| `TOOLCHAIN_HOME` | `${ASCEND_INSTALL_PATH}/cann-<version>/toolkit` | 工具链路径 |

> **PyPTO 约定**：项目代码（Python/C++/CMake）统一通过 `ASCEND_HOME_PATH` 判断 CANN 是否可用并定位安装路径。

### 诊断脚本路径探测优先级

`diagnose_env.py` 按以下顺序定位 CANN 安装路径：

1. `ASCEND_HOME_PATH` — 最高优先级，PyPTO 项目标准
2. `ASCEND_TOOLKIT_HOME` — 备选，与 HOME_PATH 通常相同
3. `ASCEND_OPP_PATH` — 取其父目录反推版本根
4. Fallback — 扫描 `${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/cann-*` 目录

### source 命令

```bash
# 默认路径（root 用户）
source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh"

# 指定路径
source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/cann/set_env.sh"
```

```bash
CANN_ENV_SH=${CANN_ENV_SH:-$(ls -1 ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/set_env.sh ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/*/*/set_env.sh 2>/dev/null | head -1)}
test -n "$CANN_ENV_SH" && source "$CANN_ENV_SH" || echo "CANN set_env.sh not found"
```

## 版本兼容

<!-- last_verified: 2026-02-28 -->
<!-- 权威源: $PYPTO_REPO/docs/install/prepare_environment.md § "安装驱动与固件" + https://github.com/Ascend/pytorch -->

### 快照：已验证兼容组合

| CANN 版本 | torch | torch_npu | Python | 芯片 |
|-----------|-------|-----------|--------|------|
| 8.5.0 | 2.5.1 | 2.5.1.post1 | 3.10 | A2(910B) / A3(910C) |
| 8.5.0 | 2.4.0 | 2.4.0.post2 | 3.10 | A2(910B) / A3(910C) |
| 8.5.0 | 2.3.1 | 2.3.1.post1 | 3.9/3.10 | A2(910B) / A3(910C) |

> ⚠️ 此快照可能滞后于上游。**安装前务必核对权威源**：
> - PyPTO 官方矩阵：`$PYPTO_REPO/docs/install/prepare_environment.md` § "安装驱动与固件"
> - torch/torch_npu 配对：https://github.com/Ascend/pytorch

### 验证当前版本

```bash
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import torch_npu; print('torch_npu', getattr(torch_npu, '__version__', 'unknown'))"
python3 -c "import os; print('ASCEND_HOME_PATH', os.environ.get('ASCEND_HOME_PATH'))"
```
