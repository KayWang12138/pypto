# PyPTO 环境准备与安装

> **目标读者：** 想把 PyPTO 在本机/容器里跑起来，并且能稳定复现与调试的开发者  
> **覆盖范围：** 真实 NPU 环境 + 仅 CPU 仿真环境；从依赖到安装到常见坑位

---

## 🚀 按人群快速跳转

| 人群 | 快速入口 |
|------|---------|
| **第一次安装 PyPTO** | [2. 必备依赖](#2-必备依赖所有环境都需要) → [4. 安装 PyPTO](#4-安装-pyto) |
| **真实 NPU 环境** | [1. 你现在是哪种运行环境？](#1-你现在是哪种运行环境) → [3. CANN 安装（真实 NPU 环境）](#3-cann-安装真实-npu-环境) → [4. 安装 PyPTO](#4-安装-pyto) |
| **仿真环境（仅 CPU）** | [1. 你现在是哪种运行环境？](#1-你现在是哪种运行环境) → [2. 必备依赖](#2-必备依赖所有环境都需要) → [4. 安装 PyPTO](#4-安装-pyto) |
| **使用 Docker** | [5. Docker 环境（可选）](#5-docker-环境可选) |
| **从源码构建** | [4. 安装 PyPTO](#4-安装-pyto) → [从源码构建与跑测试](00-getting-started/02-build-and-test.md) |
| **遇到安装问题** | [6. 常见问题](#6-常见问题) → [问题库](05-debugging/03-troubleshooting-and-known-issues.md) |

---

## 1. 你现在是哪种运行环境？

| 环境类型 | 硬件要求 | 你能做什么 |
|---|---|---|
| **真实环境** | CPU + Ascend NPU（常见 A2/A3） | NPU 上执行计算；也可跑 SIM 做预估 |
| **仿真环境** | 仅 CPU | 仅支持 SIM（用于功能验证/预估性能） |

补充说明：
- **支持系统**：常见为 OpenEuler、Ubuntu 等主流 Linux 发行版
- **关键结论**：如果你只想先把算子逻辑跑通，仿真环境即可；如果你要验证真实性能/上板问题，必须真实环境

---

## 2. 必备依赖（所有环境都需要）

### 2.1 Python 运行时

- **Python**：版本 `>= 3.9`
- **PyTorch**：需要安装（真实环境与仿真环境都需要）
- **torch_npu（真实环境）**：真实 NPU 环境下需要安装

**最容易踩的坑：**
- `PyTorch`、`torch_npu`（Ascend Extension for PyTorch）与 `PyPTO` 三者的 **Python 版本必须一致**（否则经常表现为导入失败、运行期崩溃、ABI 不兼容）

### 2.2 编译依赖（只在你需要从源码编译时）

- `cmake >= 3.16.3`
- `make`
- `gcc >= 7.3.1`

Python 侧依赖包位置通常在 `python/requirements.txt`，可按如下安装：

```bash
cd /path/to/pypto
python3 -m pip install -r python/requirements.txt
```

---

## 3. 第三方源码包准备（无法访问 cann-src-third-party 时）

PyPTO 编译过程依赖部分第三方源码包；如果你的环境可以访问 `cann-src-third-party`，它们通常会在编译时自动下载与编译。否则请提前准备并设置：

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-third-party>
```

### 3.1 何时必须手工准备？

**决策树：**
- ✅ **可以访问 `cann-src-third-party`** → 编译时自动下载（无需手工）
- ❌ **无法访问 `cann-src-third-party`** → 需要手工准备或使用脚本下载
- ❌ **网络受限/内网环境** → 必须手工准备（下载后离线拷贝）

### 3.2 手工准备（推荐：显式可控）

```bash
mkdir -p <path-to-your-thirdparty>
# 将第三方源码压缩包下载到本地后，上传/拷贝到该目录
```

### 3.3 脚本下载（推荐：批量自动化）

```bash
mkdir -p <path-to-your-thirdparty>

# 如果不指定 --download-path：默认下载到与 pypto 同级目录的 pypto_download/third_party_packages
# 如果指定 --download-path：下载到 <path-to-your-thirdparty>/third_party_packages
bash tools/prepare_env.sh --type=third_party --download-path=<path-to-your-thirdparty>
```

---

## 4. 真实 NPU 环境：CANN/驱动/固件安装（只在真实环境需要）

### 4.1 驱动与固件

建议先完成 NPU 驱动与固件安装（这一步在容器外的宿主机更常见）。安装完成后再处理 CANN 软件栈。

### 4.2 CANN toolkit / ops 安装

**重要提示：**
- **脚本优先**：建议优先使用 `tools/prepare_env.sh` 脚本进行安装（见[4.4 一键脚本安装](#44-一键脚本安装prepare_envsh)），避免手动操作错误
- **不要盲目复制 run 包名**：包名中的版本号（如 `8.5.0`）和架构需要与你的实际环境匹配

**手动安装（示意，不推荐）：**

以 `.run` 安装包为例：

```bash
# 1) 确保安装包可执行
chmod +x Ascend-cann-toolkit_8.5.0_linux-${arch}.run

# 2) 安装（install_path 常见为 /usr/local/Ascend）
./Ascend-cann-toolkit_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
```

Ops 包安装示意：

```bash
chmod +x Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run
./Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
```

变量说明（合法值列表）：
- `${arch}`：`x86_64`（Intel/AMD 64位）或 `aarch64`（ARM 64位）
- `${device_type}`：`a2`（Ascend 910A）或 `a3`（Ascend 910B）
- `${install_path}`：自定义安装路径；默认常见为 `/usr/local/Ascend`

**重要提示：**
- **不要盲目复制 run 包名**：包名中的版本号（如 `8.5.0`）和架构需要与你的实际环境匹配
- **脚本优先**：建议优先使用 `tools/prepare_env.sh` 脚本进行安装，避免手动操作错误

### 4.3 环境变量生效

```bash
# 默认路径安装（root 用户示例；非 root 用户常用 ${HOME}）
source /usr/local/Ascend/cann/set_env.sh

# 指定路径安装
source ${install_path}/cann/set_env.sh
```

### 4.4 一键脚本安装（prepare_env.sh）

项目通常提供一键脚本用于下载与安装（如 toolkit/ops/inst 等），常见用法：

```bash
bash tools/prepare_env.sh --type=cann --device-type=a2
```

常见参数含义（以脚本约定为准）：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `--type` | str | 是 | 可选 `deps/cann/third_party/all` |
| `--device-type` | str | 是 | `a2` / `a3` |
| `--install-path` | str | 否 | CANN 安装路径 |
| `--download-path` | str | 否 | 下载路径 |
| `--with-install-driver` | bool | 否 | 是否下载驱动/固件（默认 false） |

---

## 5. 安装 PyPTO（从源码编译）

本节把“常规安装/可编辑安装/Debug 编译/生成器选择”一次讲清楚。

### 5.1 常规安装（适合稳定使用）

```bash
cd /path/to/pypto

# (可选) 无法访问 cann-src-third-party 时需要
export PYPTO_THIRD_PARTY_PATH=<path-to-third-party>

python3 -m pip install . --verbose
```

### 5.2 可编辑安装（适合开发调试）

```bash
cd /path/to/pypto
export PYPTO_THIRD_PARTY_PATH=<path-to-third-party>
python3 -m pip install -e . --verbose
```

### 5.3 通过 pip `--config-setting` 指定 Debug/verbose/generator

前提：`pip >= 22.1`。

**如何确认 pip 版本 >= 22.1：**
```bash
python3 -m pip --version
# 应显示类似：pip 22.1.0 from ...
```

**Shell 注意事项（单引号/双引号）：**
- 外层使用单引号，内层使用双引号，避免引号嵌套错误
- 如果使用双引号，需要对内层引号进行转义

```bash
# Debug 构建（推荐：使用单引号）
python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug'

# Debug + 详细编译输出
python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug --cmake-verbose'

# 指定 CMake Generator（示例：Ninja）
python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-generator=Ninja'
```

### 5.4 通过 `PYPTO_BUILD_EXT_ARGS` 注入 build_ext 参数（更稳定）

一些场景下 `setuptools` 对 `--config-setting` 的直接透传不够理想，项目通常约定用环境变量注入：

```bash
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose'
python3 -m pip install -e . --verbose
```

---

## 6. 运行样例的最小前置（真实 NPU）

```bash
# 配置 CANN 环境变量（NPU 模式）
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0
```

---

## 7. 产物可视化：PyPTO Toolkit（VSCode 插件）

如果你希望查看 **计算图** 与 **泳道图**，通常需要安装 VSCode 插件（`.vsix` 文件）：

1. 下载 `.vsix` 插件文件（以项目提供的下载入口为准）
2. 打开 VSCode → 扩展 → 右上角 “...” → “从 VSIX 安装...”
3. 选择已下载的 `.vsix` 完成安装

安装完成后，结合运行产物目录即可查看计算图/泳道图（产物含义见 `docs/note/TERMINOLOGY.md`）。

---

## 8. Docker 环境（可选）

如果你希望在容器里跑 PyPTO（团队统一环境/CI 常用），通常会基于 Ascend 社区提供的 CANN 基础镜像制作开发镜像。

关键点：
- 建议 Docker 版本 `>= 27.2.1`
- **必须先在宿主机完成 NPU 驱动/固件安装**（容器主要解决软件栈）
- 常见分两类 Dockerfile：**带 CANN** / **不带 CANN（仿真）**

带 CANN 的基座示例（节选）：

```dockerfile
ARG CANN_VERSION=8.5.0.alpha001-a3-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:$CANN_VERSION

WORKDIR /tmp
RUN pip install --no-cache-dir \
    wheel tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0
```

容器里跑 NPU 时仍需要：
- 正确的设备映射/权限
- `TILE_FWK_DEVICE_ID` 与 CANN 环境变量可用


