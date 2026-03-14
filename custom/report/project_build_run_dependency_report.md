# PyPTO 项目编译、运行与依赖报告

## 1. 目的

本文档汇总当前仓库的编译入口、运行方式和依赖要求，并补充当前机器上已经实际验证过的命令，便于后续环境准备、源码构建和基础运行排查。

## 2. 项目构建特征

PyPTO 当前有三类相关能力：

- 基础 PyPTO Python + C++ 扩展构建
- CANN/NPU 路径
- Vulkan GPU 路径

当前仓库中的两个关键构建开关如下：

- `BUILD_WITH_CANN`
  - 默认打开
  - 当 `ASCEND_HOME_PATH` 未设置或对应目录不存在时，会自动关闭
- `BUILD_WITH_VULKAN`
  - 默认打开
  - 当系统找不到 Vulkan 头文件或库时，会自动关闭

这意味着：

- 没有 CANN 环境时，项目仍可构建，但会退化为非 CANN 路径
- 没有 Vulkan 开发环境时，项目仍可构建，但 Vulkan GPU backend 会被自动关闭

## 3. 依赖清单

### 3.1 基础系统与工具链依赖

以下依赖是源码编译的基础条件：

| 类别 | 要求 |
| --- | --- |
| 操作系统 | Linux，官方文档说明支持 Ubuntu、OpenEuler 等主流发行版 |
| Python | `>= 3.9` |
| Python 开发头文件 | 若源码编译，建议安装 `python3-dev` |
| CMake | `>= 3.16.3` |
| C/C++ 编译器 | `g++ >= 7.3.1`，同时需要 `gcc` |
| 构建工具 | `make` 或 `ninja` |
| pip | 推荐 `>= 22.1` |

当前机器上检测到的版本如下：

| 项目 | 当前版本 |
| --- | --- |
| Python | `3.12.7` |
| gcc | `13.3.0` |
| g++ | `13.3.0` |
| cmake | `4.2.3` |
| ninja | `1.13.0.git.kitware.jobserver-pipe-1` |

### 3.2 Python 依赖

`pyproject.toml` 的 `project.dependencies` 当前为空，因此这个仓库并不是只靠 `pip install pypto` 就能自动补齐所有开发依赖；源码构建和测试时，应以 `python/requirements.txt` 为准。

推荐先安装：

```bash
python3 -m pip install -r python/requirements.txt
```

`python/requirements.txt` 中当前包含的主要依赖如下：

- 基础编译
  - `setuptools>=77.0.3`
  - `pybind11>=2.13.6`
- 构建脚本
  - `pip>=22.1`
  - `build>=1.0.3`
  - `packaging`
- 测试
  - `pytest`
  - `pytest-forked`
  - `pytest-xdist`
  - `setproctitle`
  - `psutil`
- PyPTO Python 侧依赖
  - `sympy`
  - `PyYAML`
- 绘图与分析
  - `matplotlib`
  - `pandas`
  - `plotly`
  - `tabulate`

当前机器上检测到的关键 Python 包版本如下：

| 包 | 当前版本 |
| --- | --- |
| torch | `2.6.0+cpu` |
| pybind11 | `3.0.2` |
| pytest | `7.4.4` |
| setuptools | `82.0.1` |
| wheel | `0.44.0` |

### 3.3 编译期第三方源码包

PyPTO 编译时依赖以下第三方源码包：

| 软件包 | 版本 |
| --- | --- |
| JSON for Modern C++ | `3.11.3` |
| libboundscheck | `1.1.16` |

若开发环境可访问 `cann-src-third-party`，构建过程中会自动下载并编译；否则需要提前准备源码包，并设置：

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

这一步非常重要。若没有网络，且又没有提前准备 `PYPTO_THIRD_PARTY_PATH`，构建会在第三方依赖阶段失败。

### 3.4 NPU/CANN 相关依赖

若需要 NPU 真机运行，或者进行依赖 CANN 的精度仿真，需要额外准备：

- Ascend CANN 8.5.0 相关组件
- 对应设备的 ops 包
- `pto-isa`
- `torch_npu`
- 可用 NPU 设备和驱动/固件

常见关键环境变量如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

根据仓库文档和环境技能说明，CANN 8.5.0 常用组合为：

- `torch==2.6.0`
- `torch_npu==2.6.0.post3`

### 3.5 Vulkan GPU 相关依赖

若需要启用当前仓库中的 Vulkan GPU backend，需要额外准备：

- Vulkan 头文件与运行库
- `glslangValidator`

在 Ubuntu 上，一般对应：

```bash
sudo apt-get install -y libvulkan-dev glslang-tools
```

当前机器上检测结果：

| 项目 | 当前状态 |
| --- | --- |
| Vulkan 头文件/库 | 已可用 |
| `glslangValidator` | 已可用，版本 `15.1.0` |

### 3.6 可选 MPI 依赖

若要运行分布式用例，还需额外安装 MPI。官方文档中推荐版本不低于 `3.2.1`。

## 4. 官方推荐编译方式

### 4.1 方式一：直接通过 pip 源码安装

适合常规安装：

```bash
# 可选：若不能访问 cann-src-third-party，需要事先准备三方源码包
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

python3 -m pip install . --verbose
```

若需要指定构建类型或 CMake Generator，可使用 `--config-setting` 传参，例如：

```bash
python3 -m pip install . --verbose \
  --config-setting=--build-option='build_ext --cmake-build-type=Debug --cmake-generator=Ninja'
```

### 4.2 方式二：可编辑安装

适合开发调试：

```bash
export PYPTO_BUILD_EXT_ARGS='--cmake-generator=Ninja'
python3 -m pip install -e . --verbose
```

如果需要调试版和更详细的 C++ 输出：

```bash
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose --cmake-generator=Ninja'
python3 -m pip install -e . --verbose
```

### 4.3 方式三：通过 `build_ci.py`

仓库自带 `build_ci.py` 作为统一构建入口，适合 CI 或标准化构建流程：

```bash
python3 build_ci.py
```

典型命令：

```bash
# 指定 Python 前端和 NPU 后端
python3 build_ci.py -f python3 -b npu

# 清理后重新构建
python3 build_ci.py -c --build_type Debug

# 生成 wheel，但不自动执行后续动作
python3 build_ci.py -f python3 --clean --disable_auto_execute
```

然后安装产物：

```bash
pip install build_out/pypto-*.whl --force-reinstall -q
```

需要注意的是，`build_ci.py` 内部如果检测不到 `ASCEND_HOME_PATH`，会把 `npu` 后端自动回退为 `cost_model`。

## 5. 当前机器上已验证的编译方式

当前机器上已经实际通过的源码构建命令如下：

```bash
PATH='/home/anfield/anaconda3/lib/python3.12/site-packages/cmake/data/bin:/home/anfield/anaconda3/bin:'"$PATH" \
PYPTO_BUILD_EXT_ARGS='--cmake-generator=Ninja' \
pip3 install -e . --no-build-isolation --verbose
```

说明：

- 该命令已经在当前仓库成功执行
- 成功生成并安装了 `pypto` 的 editable wheel
- 构建过程完成了 CMake configure、build 和 install

之所以显式设置 `PATH`，是为了确保当前环境优先使用已安装的 `cmake` 和 `ninja`。

## 6. 运行方式

### 6.1 运行官方 Hello World 示例

无 NPU 真机时，可先跑仿真模式：

```bash
cd examples/00_hello_world
python3 hello_world.py --run_mode=sim
```

有 NPU 真机时，可运行：

```bash
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

更多样例可参考 `examples/` 目录。

### 6.2 NPU 路径建议运行方式

若使用 NPU 真机，建议按如下顺序执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu
```

若是纯仿真环境，则可运行：

```bash
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode sim
```

### 6.3 当前机器上已验证的导入命令

当前代码在直接 `import pypto` 时，会尝试创建默认日志目录和运行目录；在这台机器上，如果不提前设置环境变量，会触发目录创建失败。

当前机器上已经验证通过的导入命令如下：

```bash
PYPTO_HOME=/tmp/pypto_home \
ASCEND_PROCESS_LOG_PATH=/tmp/ascend_process_log \
PYTHONPATH=/home/anfield/project/pypto/python \
python3 -c 'import pypto; print(pypto.__file__)'
```

成功输出：

```text
/home/anfield/project/pypto/python/pypto/__init__.py
```

### 6.4 当前机器上已验证的 Vulkan GPU 测试命令

当前仓库的 Vulkan GPU UT 已在当前机器上通过：

```bash
PYPTO_HOME=/tmp/pypto_home \
ASCEND_PROCESS_LOG_PATH=/tmp/ascend_process_log \
PYTHONPATH=/home/anfield/project/pypto/python \
pytest -q python/tests/ut/gpu_vk
```

验证结果：

```text
16 passed
```

### 6.5 直接运行单个 Vulkan GPU 用例

若只想运行 `python/tests/ut/gpu_vk/test_runtime_execute.py` 中的
`test_runtime_execute_mul_matches_torch`，推荐优先使用 `python -m pytest`
精确执行单个用例：

```bash
cd /home/anfield/project/pypto

export PYTHONPATH=$PWD/python
export PYPTO_HOME=$PWD/.pypto_tmp
export ASCEND_PROCESS_LOG_PATH=$PWD/.ascend_log

python3 -m pytest python/tests/ut/gpu_vk/test_runtime_execute.py \
  -k test_runtime_execute_mul_matches_torch -q -s -rs
```

如果需要完全不走 `pytest`，也可以直接通过 Python 导入并调用该测试函数：

```bash
cd /home/anfield/project/pypto

export PYTHONPATH=$PWD/python
export PYPTO_HOME=$PWD/.pypto_tmp
export ASCEND_PROCESS_LOG_PATH=$PWD/.ascend_log

python3 -c "from tests.ut.gpu_vk.test_runtime_execute import test_runtime_execute_mul_matches_torch; test_runtime_execute_mul_matches_torch()"
```

说明：

- `PYTHONPATH=$PWD/python` 必须设置，否则无法导入 `pypto` 与 `pypto_gpu_vk`
- `PYPTO_HOME` 和 `ASCEND_PROCESS_LOG_PATH` 建议显式设置到可写目录，否则可能因为默认目录创建失败而中断
- 若当前机器不可用真实 Vulkan runtime，该用例会被 `skip`，而不是 `fail`

## 7. 建议的最小落地流程

如果目标是“先把项目编译并跑起来”，建议使用下面的最小流程：

### 场景 A：只验证源码构建与 Python 导入

```bash
python3 -m pip install -r python/requirements.txt
export PYPTO_BUILD_EXT_ARGS='--cmake-generator=Ninja'
python3 -m pip install -e . --verbose
PYPTO_HOME=/tmp/pypto_home ASCEND_PROCESS_LOG_PATH=/tmp/ascend_process_log PYTHONPATH=$(pwd)/python \
python3 -c 'import pypto; print(pypto.__file__)'
```

### 场景 B：验证 Vulkan GPU 路径

```bash
python3 -m pip install -r python/requirements.txt
sudo apt-get install -y libvulkan-dev glslang-tools
export PYPTO_BUILD_EXT_ARGS='--cmake-generator=Ninja'
python3 -m pip install -e . --verbose
PYPTO_HOME=/tmp/pypto_home ASCEND_PROCESS_LOG_PATH=/tmp/ascend_process_log PYTHONPATH=$(pwd)/python \
pytest -q python/tests/ut/gpu_vk
```

### 场景 C：验证 NPU 路径

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
python3 -m pip install -r python/requirements.txt
python3 build_ci.py -f python3 --clean --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall -q
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu
```

## 8. 当前已知注意事项

- `pyproject.toml` 的运行时依赖列表当前为空，实际开发/测试时应先安装 `python/requirements.txt`
- 若无法访问 `cann-src-third-party`，请务必提前准备第三方源码包并设置 `PYPTO_THIRD_PARTY_PATH`
- 若未设置 `ASCEND_HOME_PATH`，CANN 相关构建能力会自动关闭
- 若缺少 Vulkan 头文件或库，`BUILD_WITH_VULKAN` 会自动关闭
- 当前机器上直接 `import pypto` 会因为默认日志目录/运行目录创建失败而中断，建议显式设置：
  - `PYPTO_HOME`
  - `ASCEND_PROCESS_LOG_PATH`
- 本报告只对当前机器上的源码构建、Python 导入和 Vulkan GPU UT 做了实际验证；NPU 真机执行路径没有在本轮报告编写过程中重新跑一次完整验证

## 9. 结论

当前项目的源码构建入口是清晰的，主路径有两条：

- `python3 -m pip install .` / `python3 -m pip install -e .`
- `python3 build_ci.py ...`

实际依赖可以分为三层：

- 基础编译层：Python、gcc/g++、cmake、make/ninja、setuptools、pybind11
- 仓库编译层：`nlohmann_json`、`libboundscheck`
- 可选后端层：CANN/NPU 相关组件，或 Vulkan 相关组件

对当前机器而言，源码可编辑安装、`pypto` 导入和 Vulkan GPU 单元测试已经可以跑通；若下一步目标是 NPU 真机运行，还需要按 CANN 环境要求补齐 `ASCEND_HOME_PATH`、`torch_npu`、`PTO_TILE_LIB_CODE_PATH` 和 NPU 设备配置。
