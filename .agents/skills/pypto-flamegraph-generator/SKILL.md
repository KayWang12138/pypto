---
name: pypto-flamegraph-generator
description: PyPTO 算子火焰图生成技能。用于采集 PyPTO 运行时的 perf 数据，并生成火焰图，帮助开发者可视化分析算子的 CPU 性能瓶颈。触发词：火焰图、flamegraph、perf 数据采集、CPU 性能分析、生成火焰图。
---

# PyPTO 算子火焰图生成

## 概述

此技能提供 PyPTO 算子火焰图生成的完整工作流程，通过采集 perf 数据并转换为可视化火焰图，帮助开发者分析 CPU 侧的性能瓶颈。

## 适用场景

- 分析 PyPTO 算子 CPU 侧的性能瓶颈
- 可视化展示函数调用栈的耗时分布
- 定位热点函数和性能问题

## 核心原则

1. **环境准备优先**：确保 perf 工具可用，否则无法进行数据采集
2. **代码修改必要**：需要修改 runtime.cpp 以隔离 emulation 模式
3. **debug_options 配置**：必须启用 `runtime_debug_mode=1` 才能正确采集 perf 数据
4. **循环执行策略**：通过循环执行 500 次获取稳定的性能采样数据
5. **数据筛选**：使用 grep 筛选特定进程（aicput0-4）的调用栈

---

## 步骤 1：环境检查与准备

### 1.1 检查 perf 工具是否存在

检查当前 Linux 环境中是否安装了 perf 工具：

```bash
which perf
```

或者检查 perf 是否可执行：

```bash
perf --version 2>&1 || echo "perf not found"
```

**判断逻辑**：
- 如果 `which perf` 返回路径（如 `/usr/bin/perf`），则 perf 工具已安装
- 如果返回空或 `perf not found`，则 perf 工具未安装或不在 PATH 中

### 1.2 处理 perf 工具未找到的情况

如果 perf 工具不存在或不在 PATH 中，需要询问用户：

**询问用户 perf 工具路径**：
- 使用 question 工具询问用户 perf 工具的路径
- 用户可能已经安装了 perf 但不在标准 PATH 中
- 用户需要提供 perf 工具的完整路径（如 `/home/user/tools/perf`）

**用户选择选项**：
1. **输入 perf 工具路径**：用户提供已安装 perf 工具的完整路径
2. **安装 perf 工具**：用户选择安装 perf 到系统

### 1.3 安装 perf 工具（如果用户选择安装）

如果用户选择安装 perf：

**Ubuntu 系统**：
```bash
sudo apt-get update
sudo apt-get install -y linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

**CentOS/RHEL 系统**：
```bash
sudo yum install -y perf
```

**注意事项**：
- 安装 perf 需要 sudo 权限
- 不同 Linux 发行版安装命令不同
- 安装后再次执行 `which perf` 确认安装成功

### 1.4 记录 perf 工具路径

无论 perf 是通过系统 PATH 还是用户指定路径获得，都需要记录实际使用的 perf 路径：

- 如果 `which perf` 成功，记录系统路径
- 如果用户指定路径，验证路径有效性并记录

**验证用户指定的 perf 路径**：
```bash
{用户指定的perf路径} --version
```

---

## 步骤 2：修改 PyPTO 源码

### 2.1 修改 runtime.cpp 文件

**目标文件**：`python/src/bindings/runtime.cpp`

**修改位置**：`DoLaunch` 函数

**修改内容**：将 `kmodule->EmulationLaunch(kbinary, tensors);` 之外的其他部分注释掉

**修改前**：
```cpp
void DoLaunch(KernelBinary* kbinary)
{
    if (config::GetSimConfig(KEY_ACCURACY_LEVEL, 2) == 2) {
        kmodule->EslModelLaunch(kbinary, tensors);
        return;
    }
    kmodule->EmulationLaunch(kbinary, tensors);

    int64_t* wsAddr = nullptr;
    int64_t wsSize = kmodule->GetWorkspaceSize(kbinary, tensors);
    if (wsSize) {
        auto pyalloc = py::getattr(module, "alloc");
        wsAddr = (int64_t*)pyalloc(wsSize).cast<int64_t>();
    }
    // ... 其他代码
    kmodule->Launch(kbinary, aicoreStream, tensors, ctrlFlowCache, wsAddr);
}
```

**修改后**：
```cpp
void DoLaunch(KernelBinary* kbinary)
{
    // if (config::GetSimConfig(KEY_ACCURACY_LEVEL, 2) == 2) {
    //     kmodule->EslModelLaunch(kbinary, tensors);
    //     return;
    // }
    kmodule->EmulationLaunch(kbinary, tensors);

    // 注释掉其他部分，用于火焰图生成
    // int64_t* wsAddr = nullptr;
    // int64_t wsSize = kmodule->GetWorkspaceSize(kbinary, tensors);
    // ... 其他代码注释掉
}
```

**修改目的**：
- 仅保留 EmulationLaunch 调用，隔离 CPU 侧的模拟执行
- 注释掉 NPU 设备相关代码，使 perf 能采集到 CPU 侧完整调用栈

---

## 步骤 3：检查 CANN 环境变量

### 3.1 检查 ASCEND_HOME_PATH 环境变量

编译 PyPTO 需要依赖 CANN 环境。首先检查环境变量中是否已配置 CANN：

```bash
echo $ASCEND_HOME_PATH
```

**判断逻辑**：
- 如果返回非空路径（如 `/usr/local/Ascend/ascend-toolkit`），则 CANN 环境已配置
- 如果返回空，则 CANN 环境未配置，需要用户指定 setenv.sh 脚本路径

### 3.2 处理 CANN 环境未配置的情况

如果 ASCEND_HOME_PATH 环境变量不存在，需要询问用户：

**询问用户 CANN setenv.sh 脚本路径**：
- 使用 question 工具询问用户 CANN 的 setenv.sh 脚本路径
- setenv.sh 通常位于 CANN 安装目录的根目录下，例如：
  - `/usr/local/Ascend/ascend-toolkit/setenv.sh`
  - `/home/user/Ascend/ascend-toolkit/setenv.sh`

**用户需要提供的路径示例**：
- `/usr/local/Ascend/ascend-toolkit/setenv.sh`
- `/home/developer/Ascend/ascend-toolkit/setenv.sh`

### 3.3 验证并执行 setenv.sh 脚本

验证用户提供的 setenv.sh 脚本路径是否有效：

```bash
# 检查文件是否存在
ls {用户指定的setenv.sh路径}

# 检查文件是否可执行
test -f {用户指定的setenv.sh路径} && echo "文件存在" || echo "文件不存在"
```

**执行 setenv.sh 脚本**：

由于 source 命令在 bash 子进程中执行后环境变量不会传递到父进程，需要在后续编译命令中串联执行：

```bash
source {用户指定的setenv.sh路径} && python3 -m pip install . --verbose
```

**注意事项**：
- source 命令只能在 bash 环境中执行
- 每次执行编译或运行命令时都需要先 source setenv.sh
- setenv.sh 会设置 ASCEND_HOME_PATH 及其他必要环境变量

---

## 步骤 4：重新编译 PyPTO

### 4.1 编译安装

**如果 CANN 环境已配置（ASCEND_HOME_PATH 存在）**：

在 PyPTO 根目录执行编译：

```bash
python3 -m pip install . --verbose
```

**如果 CANN 环境未配置（需要先 source setenv.sh）**：

```bash
source {用户指定的setenv.sh路径} && python3 -m pip install . --verbose
```

**编译说明**：
- 必须重新编译才能使修改生效
- `--verbose` 参数用于查看详细编译日志
- 编译时间可能较长，请耐心等待
- 确保 CANN 环境变量在编译过程中可用

### 4.2 验证编译成功

检查编译输出，确认以下内容：
- 所有模块编译成功（无 ERROR）
- pypto_impl.so 文件生成
- 安装到 Python 环境成功

---

## 步骤 5：准备算子测试脚本

### 5.1 询问用户算子脚本路径

询问用户需要执行的算子脚本路径：

**示例输入**：
- `models/deepseek_v32_exp/deepseekv32_mla_prolog_quant.py`
- `custom/softmax/test_softmax.py`
- 其他算子测试脚本路径

### 5.2 检查并添加 debug_options 配置

**⚠️ 重要：必须启用 runtime_debug_mode 才能正确采集 perf 数据！**

检查脚本中的 `@pypto.frontend.jit` 装饰器配置：

#### 5.2.1 检查当前脚本

搜索脚本中是否存在 `@pypto.frontend.jit` 装饰器：

```bash
grep -n "@pypto.frontend.jit" <脚本路径>
```

#### 5.2.2 场景 1：脚本中有 @pypto.frontend.jit

如果找到装饰器，检查是否已配置 `debug_options={"runtime_debug_mode": 1}`：

**情况 A：装饰器没有 debug_options**

**修改前**：
```python
@pypto.frontend.jit(
    runtime_options={...},
    pass_options={...}
)
def kernel_function(...):
    pass
```

**修改后**：
```python
@pypto.frontend.jit(
    runtime_options={...},
    pass_options={...},
    debug_options={"runtime_debug_mode": 1}
)
def kernel_function(...):
    pass
```

**情况 B：装饰器有 debug_options 但没有 runtime_debug_mode**

**修改前**：
```python
@pypto.frontend.jit(
    debug_options={"other_option": value}
)
def kernel_function(...):
    pass
```

**修改后**：
```python
@pypto.frontend.jit(
    debug_options={"other_option": value, "runtime_debug_mode": 1}
)
def kernel_function(...):
    pass
```

**情况 C：装饰器已有 runtime_debug_mode=1**

无需修改，继续执行后续步骤。

#### 5.2.3 场景 2：脚本中没有 @pypto.frontend.jit

如果当前脚本中没有找到装饰器，需要从 import 的位置查找：

**步骤**：
1. 查看脚本中的 import 语句
2. 找到导入的模块文件路径
3. 在导入模块中搜索 `@pypto.frontend.jit` 装饰器
4. 在找到的装饰器中添加 `debug_options={"runtime_debug_mode": 1}`

**示例流程**：

如果脚本中有：
```python
from mla_prolog_quant_impl import mla_prolog_quant_p, mla_prolog_quant_d
```

则需要：
1. 查找 `mla_prolog_quant_impl.py` 文件位置
2. 在该文件中搜索 `@pypto.frontend.jit`
3. 在装饰器中添加 `debug_options={"runtime_debug_mode": 1}`

**查找 import 文件位置**：
```bash
# 在脚本目录中搜索
find . -name "mla_prolog_quant_impl.py"
```

**修改 import 文件中的装饰器**（同场景 1 的修改方法）。

**⚠️ 注意**：
- 如果 import 链路有多层，需要逐层查找直到找到 @pypto.frontend.jit
- 如果 import 模块是系统模块（非自定义），不需要修改

### 5.3 注释精度比对代码

**⚠️ 重要：火焰图采集模式下不需要进行精度比对，避免因精度问题中断 perf 采集！**

在算子执行完成后，通常会有精度比对代码（如 `compare()`、`check()`、`assert` 等），需要将这些代码注释掉。

#### 5.3.1 搜索精度比对代码

在算子脚本中搜索常见的精度比对函数：

```bash
grep -n "compare\|check\|assert" <脚本路径>
```

**常见的精度比对函数名**：
- `compare()` - 比较输出与 golden 数据
- `check()` - 检查输出正确性
- `check_is_nan_inf()` - 检查 NaN/Inf 值
- `assert` - 断言检查

#### 5.3.2 注释精度比对代码

找到精度比对代码后，将其注释掉：

**修改示例**：

**修改前**：
```python
########### compare #######
print("qNope =======")
compare(output_q_nope_data.cpu(), golden1.cpu(), "qNope", 0.005, 0.0078125, 0.005)
print("qRope =======")
compare(output_q_rope_data.cpu(), golden2.cpu(), "qRope", 0.005, 0.0078125, 0.005)
print("kv =======")
compare(output_kv_cache_data.cpu(), golden3.cpu(), "kv", 0.0001, 0.0078125, 0)
```

**修改后**：
```python
########### compare #######
# 注释掉精度比对代码，用于火焰图生成
# print("qNope =======")
# compare(output_q_nope_data.cpu(), golden1.cpu(), "qNope", 0.005, 0.0078125, 0.005)
# print("qRope =======")
# compare(output_q_rope_data.cpu(), golden2.cpu(), "qRope", 0.005, 0.0078125, 0.005)
# print("kv =======")
# compare(output_kv_cache_data.cpu(), golden3.cpu(), "kv", 0.0001, 0.0078125, 0)
```

**修改目的**：
- 火焰图采集模式下，EmulationLaunch 仅模拟 CPU 侧执行，输出结果可能包含 NaN/Inf
- 避免精度比对失败导致脚本中断，影响 perf 数据采集
- 火焰图分析完成后，建议回退此修改恢复精度比对

#### 5.3.3 特殊情况处理

**情况 A：精度比对代码在独立函数中**

如果精度比对代码封装在独立函数中（如 `compare_result()`），可以直接注释该函数调用：

```python
# 注释掉精度比对函数调用，用于火焰图生成
# compare_result(outputs, goldens)
```

**情况 B：精度比对代码与计算代码混合**

如果精度比对代码与算子计算代码在同一函数中，需要仔细区分：
- **保留**：算子计算代码（`mla_prolog_quant_d()` 等调用）
- **注释**：精度比对代码（`compare()`、`check()` 等）

```python
# 执行算子（保留）
mla_prolog_quant_d(*input_data, *output_data, ...)

# 精度比对（注释）
# compare(output_data, golden_data)
```

### 5.4 修改测试脚本 main 函数

找到脚本中的 `main` 函数（或 `if __name__ == "__main__":` 部分），将测试函数改为循环执行 500 次。

**修改示例**：

**修改前**：
```python
if __name__ == "__main__":
    logging.basicConfig(...)
    test_b4_s64k2_pa_nd_bf16_quantb_d()
```

**修改后**：
```python
if __name__ == "__main__":
    logging.basicConfig(...)
    for i in range(500):
        test_b4_s64k2_pa_nd_bf16_quantb_d()
```

**修改目的**：
- 循环执行 500 次，确保 perf 采集到足够多的样本数据
- 提高火焰图的准确性和稳定性

---

## 步骤 6：获取 FlameGraph 工具

### 6.1 询问用户 FlameGraph 工具位置

询问用户 FlameGraph 工具的获取方式：
- **选项 1**：git clone 获取
- **选项 2**：输入已有路径

### 6.2 git clone 获取 FlameGraph（如果用户选择）

如果用户选择 git clone 方式：

```bash
git clone https://github.com/brendangregg/FlameGraph.git
```

**克隆位置**：默认克隆到当前工作目录

### 6.3 验证 FlameGraph 工具

检查 FlameGraph 目录中的关键工具：
```bash
ls FlameGraph/
```

**必需工具**：
- `stackcollapse-perf.pl` - perf 数据折叠工具
- `flamegraph.pl` - 火焰图生成工具

---

## 步骤 7：采集 perf 数据

### 7.1 执行 perf record

使用 perf record 采集性能数据：

**如果 perf 在系统 PATH 中**：
```bash
perf record -F 997 -g python <用户指定的算子脚本路径>
```

**如果使用用户指定的 perf 路径**：
```bash
{用户指定的perf路径} record -F 997 -g python <用户指定的算子脚本路径>
```

**参数说明**：
- `-F 997`：采样频率 997 Hz（推荐值，避免与系统时钟同步）
- `-g`：记录调用栈信息
- `-o perf.data`：输出文件（默认为 perf.data）
- `{用户指定的perf路径}`：步骤 1.4 中记录的 perf 工具路径

**执行说明**：
- 执行过程中会运行算子脚本 500 次
- perf 会在后台采集 CPU 性能数据
- 执行完成后生成 `perf.data` 文件

**注意事项**：
- 需要 perf 工具已安装
- 需要 root 权限或适当配置（`/proc/sys/kernel/perf_event_paranoid`）
- 采集时间取决于算子执行时长

---

## 步骤 8：转换 perf 数据

### 8.1 将 perf.data 转换为可读格式

使用 perf script 解析 perf.data：

**如果 perf 在系统 PATH 中**：
```bash
perf script -i perf.data > {脚本名称}.perf
```

**如果使用用户指定的 perf 路径**：
```bash
{用户指定的perf路径} script -i perf.data > {脚本名称}.perf
```

**参数说明**：
- `-i perf.data`：指定输入文件
- `> {脚本名称}.perf`：输出到文本文件

**输出文件**：`{脚本名称}.perf`
- 包含完整的调用栈信息
- 文本格式，可直接查看

**脚本名称示例**：
- 如果原脚本为 `deepseekv32_mla_prolog_quant.py`
- 输出文件为 `deepseekv32_mla_prolog_quant.perf`

---

## 步骤 9：生成火焰图

### 9.1 使用 FlameGraph 工具生成火焰图

执行以下命令生成火焰图：

```bash
{FlameGraph-path}/stackcollapse-perf.pl {脚本名称}.perf | \
    grep -E '^aicput[0-4]' | \
    {FlameGraph-path}/flamegraph.pl > {脚本名称}.svg
```

**参数说明**：
- `{FlameGraph-path}`：FlameGraph 工具目录路径
- `{脚本名称}.perf`：上一步生成的 perf 数据文件
- `grep -E '^aicput[0-4]'`：筛选特定进程（aicput0-4）的调用栈
- `> {脚本名称}.svg`：输出火焰图 SVG 文件

**命令分解**：
1. `stackcollapse-perf.pl`：将 perf 数据转换为折叠格式（每行一个调用栈及其采样次数）
2. `grep -E '^aicput[0-4]'`：筛选包含 aicput 进程的调用栈（PyPTO 模拟执行进程）
3. `flamegraph.pl`：将折叠数据转换为 SVG 火焰图

### 9.2 输出文件

**输出文件**：`{脚本名称}.svg`
- SVG 格式的火焰图
- 可在浏览器中打开查看
- 支持交互式查看（点击放大）

---

## 步骤 10：查看和分析火焰图

### 10.1 打开火焰图

火焰图生成后，可以：
- 在浏览器中打开 SVG 文件
- 使用图形查看工具打开
- 分享给其他开发者分析

### 10.2 火焰图解读

**火焰图结构**：
- **横轴**：采样占比（宽度越大，耗时越多）
- **纵轴**：调用栈深度（从下往上是函数调用关系）
- **颜色**：随机分配，无特殊含义

**分析方法**：
1. 查看最宽的部分（耗时最多的函数）
2. 从底部向上追踪调用栈
3. 找到热点函数进行优化
4. 分析 CPU 侧性能瓶颈

---

## 完整执行流程总结

```
步骤 1: 环境检查
    ├─ 1.1 检查 perf 工具 (which perf 或 perf --version)
    ├─ 1.2 如果未找到，询问用户 perf 工具路径
    ├─ 1.3 如果用户选择安装，执行安装命令
    └─ 1.4 记录并验证 perf 工具路径

步骤 2: 修改源码
    └─ 2.1 修改 runtime.cpp DoLaunch 函数
        └─ 注释掉除 EmulationLaunch 外的其他代码

步骤 3: 检查 CANN 环境
    ├─ 3.1 检查 ASCEND_HOME_PATH 环境变量
    ├─ 3.2 如果未配置，询问用户 setenv.sh 脚本路径
    └─ 3.3 验证并执行 setenv.sh 脚本

步骤 4: 重新编译
    ├─ 4.1 编译安装（必要时先 source setenv.sh）
    └─ 4.2 验证编译成功

步骤 5: 准备算子脚本
    ├─ 5.1 询问用户算子脚本路径
    ├─ 5.2 检查并添加 debug_options={"runtime_debug_mode": 1}
    │   ├─ 搜索当前脚本中的 @pypto.frontend.jit
    │   ├─ 如果没有，从 import 位置查找
    │   └─ 在装饰器中添加 debug_options 配置
    ├─ 5.3 注释精度比对代码（compare、check 等）
    └─ 5.4 修改 main 函数，循环执行 500 次

步骤 6: 获取 FlameGraph
    ├─ 6.1 询问用户获取方式
    ├─ 6.2 git clone（如果选择）
    └─ 6.3 验证工具完整性

步骤 7: 采集 perf 数据
    └─ perf record -F 997 -g python <脚本路径>

步骤 8: 转换 perf 数据
    └─ perf script -i perf.data > {脚本名称}.perf

步骤 9: 生成火焰图
    └─ stackcollapse-perf.pl + grep + flamegraph.pl
        └─ 输出 {脚本名称}.svg

步骤 10: 查看和分析
    └─ 在浏览器中打开 SVG 文件
```

---

## 常见问题

### 问题 1：perf 工具无法使用

**可能原因**：
- perf 未安装
- perf 不在系统 PATH 中
- 权限不足（perf_event_paranoid 配置）
- 用户指定的 perf 路径无效

**解决方案**：
1. **如果 perf 未安装**：
```bash
# 安装 perf
sudo apt-get install -y linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

2. **如果 perf 不在 PATH 中**：
- 提供用户指定的 perf 工具完整路径
- 验证路径有效性：`{用户路径}/perf --version`

3. **如果权限不足**：
```bash
# 调整权限配置
sudo sysctl kernel.perf_event_paranoid=1
sudo sysctl kernel.kptr_restrict=0
```

### 问题 2：火焰图无数据或数据很少

**可能原因**：
- 循环次数太少（建议 500 次）
- 执行时间太短
- grep 筛选条件不匹配

**解决方案**：
- 增加循环次数
- 检查进程名称是否为 aicput
- 调整 grep 筛选条件

### 问题 3：编译失败

**可能原因**：
- runtime.cpp 语法错误
- 环境依赖缺失
- CANN 环境未配置（ASCEND_HOME_PATH 未设置）

**解决方案**：
- 检查代码修改是否正确
- 确保所有依赖已安装
- 查看编译日志定位具体错误
- 如果 CANN 环境未配置：
  ```bash
  # 检查 ASCEND_HOME_PATH
  echo $ASCEND_HOME_PATH
  
  # 如果为空，source setenv.sh
  source {CANN安装路径}/setenv.sh && python3 -m pip install . --verbose
  ```

### 问题 4：CANN 环境变量未配置

**可能原因**：
- CANN 未安装
- setenv.sh 未被执行
- setenv.sh 路径错误

**解决方案**：
1. **检查 CANN 是否安装**：
   - 通常安装在 `/usr/local/Ascend/ascend-toolkit` 或用户自定义目录
   - 确认安装目录下存在 `setenv.sh` 文件

2. **验证 setenv.sh 路径**：
   ```bash
   ls {用户提供的路径}/setenv.sh
   ```

3. **正确执行 setenv.sh**：
   ```bash
   source {CANN路径}/setenv.sh
   # 验证环境变量
   echo $ASCEND_HOME_PATH
   ```

4. **注意事项**：
   - source 命令只在当前 shell 会话中生效
   - 需要在每次编译/运行前执行 source
   - 或将 source 命令添加到 `~/.bashrc` 中永久生效

### 问题 5：未找到 @pypto.frontend.jit 装饰器

**可能原因**：
- 装饰器在 import 的模块中
- import 链路有多层

**解决方案**：
- 查看脚本的 import 语句
- 逐层查找 import 模块文件
- 在最终找到的模块文件中修改装饰器

### 问题 6：debug_options 配置未生效

**可能原因**：
- 配置项名称错误
- 配置值错误

**解决方案**：
- 确保配置项为 `debug_options={"runtime_debug_mode": 1}`
- 检查 JSON 格式是否正确（注意引号和逗号）
- 重新编译并运行验证

### 问题 7：精度比对代码导致 perf 采集中断

**可能原因**：
- 未注释掉精度比对代码（compare、check 等）
- EmulationLaunch 模式下输出包含 NaN/Inf 值
- 精度比对失败触发 assert 或 raise

**解决方案**：
1. **搜索精度比对代码**：
   ```bash
   grep -n "compare\|check\|assert" <脚本路径>
   ```

2. **注释精度比对代码**：
   - 注释掉所有 `compare()` 函数调用
   - 注释掉所有 `check()` 函数调用
   - 注释掉与精度相关的 `assert` 语句

3. **示例修改**：
   ```python
   # 注释掉精度比对代码，用于火焰图生成
   # compare(output_data, golden_data, "test", 0.005, 0.0078125, 0.005)
   ```

4. **注意事项**：
   - 火焰图生成完成后，务必回退此修改
   - 恢复精度比对代码以进行正常测试

---

## 注意事项

1. **修改回退**：火焰图生成完成后，建议将以下修改回退：
   - runtime.cpp 中的 DoLaunch 函数修改
   - 测试脚本中的循环修改
   - @pypto.frontend.jit 中的 debug_options 配置
   - 精度比对代码的注释修改（恢复 compare、check 等函数调用）

2. **数据清理**：perf.data 和 .perf 文件较大，分析完成后可删除以节省空间

3. **多次采集**：如果数据不稳定，可多次采集生成多个火焰图对比

4. **进程筛选**：如果火焰图中包含其他进程数据，可调整 grep 筛选条件
   - 默认筛选 aicput 进程（模拟执行进程）
   - 如果进程名为 python3，需要调整 grep 条件

5. **配置验证**：确保 debug_options={"runtime_debug_mode": 1} 配置正确添加，否则可能无法采集到有效数据

6. **CANN 环境管理**：
   - source setenv.sh 只在当前 shell 会话中生效
   - 如需永久配置，可将 source 命令添加到 `~/.bashrc` 或 `~/.profile`
   - 每次编译或运行 PyPTO 时都需要确保 CANN 环境变量已加载

7. **精度比对代码**：
   - 火焰图采集模式下，EmulationLaunch 仅模拟 CPU 侧执行，输出可能包含 NaN/Inf
   - 必须注释掉精度比对代码，否则会导致脚本中断，影响 perf 数据采集
   - 火焰图生成完成后，务必回退此修改以恢复正常测试功能

---

## 参考资料

- [FlameGraph 官方仓库](https://github.com/brendangregg/FlameGraph)
- [perf 工具文档](https://perf.wiki.kernel.org/index.php/Tutorial)
- [火焰图原理](http://www.brendangregg.com/flamegraphs.html)