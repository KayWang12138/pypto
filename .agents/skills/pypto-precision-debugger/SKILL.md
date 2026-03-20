---
name: pypto-precision-debugger
description: |
  PyPTO 算子精度问题排查技能。通过系统化排查流程定位精度问题根因，包括 workspace、unroll、合轴、内存重叠等常见问题。当算子精度验证失败、输出结果异常或用户请求精度问题排查时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题排查技能

此技能提供系统化的精度问题排查流程，帮助定位 PyPTO 算子精度问题的根本原因。

## 核心原则

### ⭐ 重要提示：使用新前端写法

**强烈建议使用 `pypto.frontend.jit` 而非 `pypto.jit`！**

新前端（`pypto.frontend.jit`）是 PyPTO 推荐的写法，旧前端（`pypto.jit`）已不再维护。使用新前端可避免许多已知问题。

**推荐写法**：

```python
# ✅ 推荐：新前端写法
@pypto.frontend.jit
def my_kernel(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):

# ❌ 不推荐：旧前端写法（已废弃）
@pypto.jit
def my_kernel(input_tensor, output_tensor):
    ...
```

**代码参考位置**：

| 示例类型 | 文件路径 |
|---------|---------|
| Hello World | `examples/00_hello_world/hello_world.py` |
| 基础操作 | `examples/01_beginner/basic/basic_ops.py` |
| Attention 算子 | `examples/03_advanced/advanced_nn/attention/attention.py` |
| 循环控制 | `examples/02_intermediate/controlflow/loop/loop.py` |
| 动态 shape | `examples/02_intermediate/controlflow/others/dynamic.py` |

### 排查原则

1. **先易后难**：从简单检查开始，逐步深入
2. **隔离变量**：每次只修改一个配置，确认效果
3. **记录过程**：记录每步排查结果，便于回溯
4. **定位根因**：找到问题后确认根因，而非绕过问题

### 常见精度问题类型

> 基于 34 个已关闭精度 Issue 的统计分析

| 问题类型 | 占比 | 典型现象 | 排查方法 |
|---------|------|---------|---------|
| Pass 相关问题 | 32% | reshape/view 后精度异常、unroll 后精度失败 | 检查 valid_shape 推导、关闭 unroll 测试 |
| Frontend/Parser 问题 | 21% | 循环展开写法精度差异、动态轴处理错误 | 对比不同写法、检查动态轴缓存 |
| Operation 问题 | 26% | 特定算子精度失败、buffer 越界 | 检查临时 buffer 大小、参数合法性 |
| Operator 问题 | 9% | 算子计算结果错误 | 检查算子实现逻辑 |
| Interpreter/Config 问题 | 12% | 多 kernel 配置干扰、inplace 问题 | 检查配置作用域、内存管理 |

**详细问题分类**：

| 问题类型 | 典型现象 | 排查方法 |
|---------|---------|---------|
| workspace 不足 | 输出随机异常 | 扩大 workspace 测试 |
| 循环展开问题 | 循环场景精度异常 | 设置 unroll_list=[1] |
| 合轴问题 | 特定 shape 精度异常 | 检查尾轴为1的tensor |
| 并行执行问题 | 嵌套循环精度异常 | 配置 submit_before_loop=True |
| 内存重叠 | 输出数据错乱 | 内存重叠检测 |
| valid_shape 错误 | view/reshape 异常 | 检查 valid_shape 配置 |
| 编译器优化问题 | 特定 op 结果异常 | 尝试 +0.0 |
| 临时 buffer 越界 | 特定 shape 报错或静默错误 | 检查算子内部 buffer 限制 |
| 动态轴推导错误 | 静态/动态轴结果不一致 | 检查 outcast 动态轴表达式 |
| inplace 操作问题 | view+reshape 后精度大偏差 | 避免 inplace 组合使用 |
| 配置作用域泄漏 | 多 kernel 配置干扰 | 检查配置栈清理 |
| GC 内存回收问题 | 精度工具粗检失败 | 检查 tensor 生命周期 |

## 完整工作流程

### 步骤 0：检查前端写法（最高优先级）

**⚠️ 最高优先级：首先检查用户是否使用了旧前端写法！**

**检查内容**：查看用户代码中的装饰器是 `pypto.jit` 还是 `pypto.frontend.jit`

**如果用户使用 `pypto.jit`**：

```markdown
## ⚠️ 强烈建议：切换到新前端写法

检测到您使用的是旧前端写法 `@pypto.jit`，这是已废弃的写法。

**请立即尝试将 `@pypto.jit` 改为 `@pypto.frontend.jit`**：

```python
# 原写法（已废弃）
@pypto.jit
def my_kernel(...):
    ...

# 改为新前端写法
@pypto.frontend.jit
def my_kernel(...):
    ...
```

**切换后请重新测试精度**，许多精度问题在新前端中已不再出现。

如果切换后精度问题仍然存在，请继续后续排查步骤。
```

**判断标准**：
- 切换后精度正确 → 问题已解决，无需继续排查
- 切换后精度仍错误 → 继续步骤 1 收集更多信息

### 步骤 1：收集必要信息

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **operator_path**: 算子代码路径
- **test_path**: 测试用例路径
- **error_description**: 精度误差描述（误差大小、异常元素分布等）
- **reproduce_steps**: 问题复现步骤

### 步骤 1.5：问题匹配与参考

**⚠️ 重要：必须根据用户描述的问题现象，提供与历史 Issue 的相似性分析和参考依据。**

**匹配流程**：

1. **提取关键词**：从用户描述中提取关键信息
   - 涉及的 API（reshape、view、loop、assemble 等）
   - 触发条件（特定 shape、非整除、unroll 等）
   - 错误现象（精度偏差、aicore error、nan 等）

2. **匹配相似 Issue**：参考本文档"问题匹配指南"表格，找到相关 Issue

3. **提供参考依据**：向用户说明
   - 相似 Issue 编号及链接
   - 该 Issue 的问题原因
   - 该 Issue 的修复方法

**强制输出格式**：

```markdown
## 问题匹配结果

根据您描述的问题现象，匹配到以下相似 Issue：

### 相关 Issue
- **Issue #XXX**：[Issue 标题]
  - 链接：https://gitcode.com/cann/pypto/issues/XXX
  - 相似点：[具体说明与用户问题的相似之处]

### 可能原因
[参考 Issue 的问题原因，结合用户情况说明]

### 推荐排查方法
1. [具体排查步骤]
2. [具体排查步骤]
3. [具体排查步骤]

**注意**：如果使用最新版本代码，该 Issue 可能已修复。建议先确认代码版本。
```

**示例输出**：

```markdown
## 问题匹配结果

根据您描述的"reshape 后精度异常"问题，匹配到以下相似 Issue：

### 相关 Issue
- **Issue #498**：pypto.reshape 丢失 valid_shape
  - 链接：https://gitcode.com/cann/pypto/issues/498
  - 相似点：reshape + matmul 场景，b 轴不能被 tile_b 整除时精度失败

### 可能原因
reshape 操作未正确传播 valid_shape，导致后续 MTE 越界

### 推荐排查方法
1. 检查 reshape 前后 valid_shape 的值
2. 对比整除与非整除场景的差异
3. 检查 MTE 是否越界

**注意**：该 Issue 已修复 (PR !1385)，如果使用最新版本代码，该问题不应再出现。
```

**要求**：
- 如果存在相似 Issue，必须提供至少一个
- 必须说明相似点
- 必须提供参考链接
- 必须给出具体排查方法
- 如果 Issue 已修复，必须说明修复状态和 PR 号
- 如果没有相似 Issue，可以直接进入排查步骤

### 步骤 2：基础检查

#### 2.1 检查 Tensor 初始化

**⚠️ 重要：区分输入参数 tensor 和局部临时 tensor，它们的初始化方式存在差异。**

**Tensor 类型分类**：

| 类型 | 定义方式 | 内存管理 | 常见问题 |
|-----|---------|---------|---------|
| **输入参数 tensor** | 作为函数参数传入 | 由外部管理 | 未正确传入、数据类型不匹配 |
| **局部临时 tensor** | 函数内部创建（`pypto.Tensor`、`pypto.zeros` 等） | 由框架管理 | 未初始化就读取、作用域问题 |

**建议**：确保输入参数 tensor 和输出参数 tensor 已正确传入，局部临时 tensor 使用 `pypto.zeros`、`pypto.full` 等接口显式初始化，避免使用 `pypto.Tensor` 声明后直接读取。

**排查技巧：将局部临时 tensor 改为输入参数**

当怀疑局部临时 tensor 存在问题时，可尝试将其改为输入参数，观察精度是否正确：

```python
# 原写法：局部临时 tensor
@pypto.frontend.jit
def my_kernel(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):
    temp = pypto.zeros(shape, dtype=pypto.DT_FP32)  # 局部临时 tensor
    temp[:] = input_tensor * 2.0
    output_tensor[:] = temp + 1.0

# 排查写法：将临时 tensor 改为输入参数
@pypto.frontend.jit
def my_kernel_debug(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    temp: pypto.Tensor(shape, pypto.DT_FP32),  # 改为输入参数
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):
    temp[:] = input_tensor * 2.0
    output_tensor[:] = temp + 1.0

# 调用时从外部传入
temp = torch.zeros(shape, dtype=torch.float32, device=device)
my_kernel_debug(input_tensor, temp, output_tensor)
```

**判断标准**：
- 改为输入参数后精度正确 → 局部 tensor 的内存管理/作用域存在问题
- 改为输入参数后精度仍错误 → 问题不在 tensor 初始化，继续其他排查

**相关 Issue**：
- #340：pypto.full 在 loop 外创建导致精度问题
- #626：pypto.full 在 loop 外创建 tensor 精度问题

#### 2.2 检查数据类型

确认数据类型转换是否正确：

```python
# 检查数据类型
print(f"Input dtype: {input_tensor.dtype}")
print(f"Output dtype: {output_tensor.dtype}")
```

### 步骤 3：内存相关排查

#### 3.1 扩大 workspace 测试

**目的**：排查 workspace 计算是否正确

**操作步骤**：

1. 找到 workspace 创建代码：`python/pypto/frontend/parser/entry.py`

2. 修改 workspace 大小（扩大 10 倍）：

```python
# 原代码
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)

# 修改为
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```

3. 重新编译并测试：

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd - && python3 test_operator.py
```

**判断标准**：
- 问题不复现 → workspace 计算问题，需检查 workspace 大小计算逻辑
- 问题仍存在 → 继续其他排查

#### 3.2 Workspace 管理方式切换

**目的**：排查 workspace 使用是否存在内存踩踏

**操作步骤**：

1. 找到 workspace 管理代码：`framework/src/machine/runtime/device_launcher.cpp`

2. 修改为内部自管理：

```cpp
// 原代码
if (config.workspaceAddr) {
    kArgs.workspace = (int64_t *)config.workspaceAddr;
} else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
    kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, ...);
}

// 修改为
if (0) {  // 强制使用内部管理
    kArgs.workspace = (int64_t *)config.workspaceAddr;
} else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
    kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, ...);
}
```

3. 重新编译并测试

**判断标准**：
- 问题不复现 → workspace 使用问题，存在内存踩踏
- 问题仍存在 → 继续其他排查

#### 3.3 内存重叠检测

**目的**：检测是否存在内存踩踏

**前置条件**：

1. 开启 VERBOSE 日志：`framework/src/machine/utils/device_switch.h`

```cpp
#define ENABLE_COMPILE_VERBOSE_LOG 1
```

2. 开启 DEBUG 日志并指定落盘路径：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./device_log
```

3. 重新编译 pypto whl 包并安装

4. 启用性能数据采集：

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}
)
```

**执行检测**：

```bash
python3 tools/schema/schema_memory_check.py \
    -d /path/to/device_log/debug/device-0/ \
    -t /path/to/output/output_xxx/dyn_topo.txt
```

**结果判断**：
- 无异常提示 → 无内存重叠
- 提示内存重叠 → 记录问题 device task 和 leaf function

### 步骤 4：特性排除

#### 4.1 关闭 unroll_list

**适用场景**：循环展开导致的精度问题

**问题原因**：`unroll_list` 用于循环展开，产生更大的 loop body。展开次数为 n 时，循环步长会变成 step*n，每次迭代会执行 n 次循环体。某些情况下展开可能导致精度问题。

**配置方法**：

设置 `unroll_list=[1]` 或不传递该参数（默认值为 `[1]`），即不进行展开：

```python
# 不展开（默认行为）
for idx in pypto.loop(b_loop):
    ...

# 或者显式设置 unroll_list=[1]
for idx in pypto.loop(b_loop, unroll_list=[1]):
    ...
```

**判断标准**：
- 问题不复现 → 循环展开导致的精度问题
- 问题仍存在 → 继续其他排查

#### 4.2 合轴问题排查

**适用场景**：特定 shape 精度异常，可能与内存布局相关

**问题原因**：合轴（axis_combine）是框架内部的优化特性，用于优化内存访问。当尾轴为 1 时，框架可能会进行合轴优化。某些情况下合轴可能导致精度问题。

**说明**：合轴是框架层面的优化，用户无法直接配置开关。如果怀疑是合轴问题：

1. 检查算子中是否存在尾轴为 1 的 tensor
2. 检查是否存在 view/assemble 操作（尾轴有 assemble 时不支持合轴）
3. 尝试调整 tensor shape，避免尾轴为 1

**相关代码位置**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.cpp`

**判断标准**：
- 调整 shape 后问题消失 → 可能与合轴相关
- 问题仍存在 → 继续其他排查

#### 4.3 配置 submit_before_loop

**适用场景**：父循环内跨多个子循环的 Tensor 内存问题、需要 loop 串行执行

**问题原因**：父循环多次迭代分配的内存地址相同，不同迭代并行执行时内存覆盖

**配置方法**：

在子循环中添加 `submit_before_loop=True`，强制多次迭代串行运行：

```python
for outer in pypto.loop(...):  # 父循环，执行至少两次
    t = pypto.Tensor(...)      # 定义一个临时 tensor
    for inner0 in pypto.loop(...):  # 第一个子循环，对临时 tensor t 赋值
        ...
        t[...] = ... 
    # 添加 submit_before_loop，确保父循环多次迭代不在同一个并行执行块中
    for inner1 in pypto.loop(..., submit_before_loop=True):  # 第二个子循环，使用了临时 tensor t
        x[:] = t[:] + t[:]
```

**判断标准**：
- 问题不复现 → 并行执行导致的内存覆盖问题
- 问题仍存在 → 继续其他排查

#### 4.4 检查 valid_shape 配置

**适用场景**：使用 `pypto.view` 或 `pypto.reshape` 时特定维度异常

**问题原因**：`valid_shape` 是 `pypto.view` 和 `pypto.reshape` 的参数，用于指定 tensor 的有效形状。如果配置不正确，可能导致数据访问错误。

**检查项**：

- [ ] 检查 `valid_shape` 是否与实际数据范围匹配
- [ ] 检查 `valid_shape` 各维度是否正确计算
- [ ] 检查动态 shape 场景下 `valid_shape` 是否正确传递

**配置示例**：

```python
# pypto.view 使用 valid_shape
x_valid_shape = x.shape
x_valid_shape[2] = x_valid_shape[2] // 2  # 修改某个维度
x1 = pypto.view(input_tensor, shape, offset, valid_shape=x_valid_shape)

# pypto.reshape 使用 valid_shape
x_re = pypto.reshape(x_trans, x.shape, valid_shape=x_valid_shape)
```

**判断标准**：
- 修正 `valid_shape` 后问题消失 → valid_shape 配置问题
- 问题仍存在 → 继续其他排查

#### 4.5 尝试 +0.0 技巧

**适用场景**：某些计算精度异常，可能与编译器优化相关

**使用方法**：

在计算中添加 `+ 0.0`，可能影响编译器的优化路径：

```python
# 原代码
result = compute_op(input)

# 尝试添加 +0.0
result = compute_op(input) + 0.0
```

**说明**：此方法是一种经验性调试技巧，通过添加 `+0.0` 可能改变计算图的优化方式。如果有效，说明问题可能与编译器优化相关，建议记录并反馈给框架团队。

**判断标准**：
- 问题不复现 → 可能与编译器优化相关，建议反馈
- 问题仍存在 → 继续其他排查

### 步骤 5：二分定位

如果以上排查未定位问题，使用二分查找定位具体问题 op：

```bash
# 调用 pypto-binary-search-verify skill
# 或使用其脚本
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

详见 [pypto-binary-search-verify](../pypto-binary-search-verify/SKILL.md)。

### 步骤 6：深入分析（常规方法无效时）

**⚠️ 重要：当常规排查方法无法定位问题时，应主动深入分析问题代码。**

**触发条件**：
- 问题匹配指南中无相似 Issue
- 常规排查步骤均无效
- 问题现象特殊，无法直接归类

**分析内容**：
1. **代码结构分析**：理解算子计算逻辑、绘制数据流图、识别风险点
2. **运行流程分析**：分析编译阶段、内存分配、并行执行、数据搬运
3. **精度问题推测**：基于分析结果给出推测和验证方法

**详细分析流程**：参见本文档"深入分析：代码与运行流程分析"章节。

### 步骤 7：输出结果

排查完成后，输出以下信息：

```markdown
## 排查结果

### 问题类型
[workspace/unroll_list/合轴/内存重叠/valid_shape/计算精度/其他]

### 问题位置
- 文件：xxx.py
- 函数：xxx
- 行号：xxx

### 问题原因
[详细描述问题原因]

### 修复建议
[具体的修复方案]

### 验证方法
[如何验证修复有效]
```

## 排查决策树

```
精度问题
    │
    ├─ 步骤 0：前端写法检查（最高优先级）
    │   └─ 使用 pypto.jit？ ──是──▶ 切换到 pypto.frontend.jit 重试
    │
    ├─ 基础检查
    │   ├─ 输入初始化？ ──否──▶ 初始化输入
    │   └─ 数据类型正确？ ──否──▶ 修正数据类型
    │
    ├─ 内存排查
    │   ├─ 扩大 workspace 正常？ ──是──▶ workspace 计算问题
    │   ├─ 内部管理正常？ ──是──▶ workspace 使用问题
    │   └─ 内存重叠？ ──是──▶ 修复内存重叠
    │
    ├─ 特性排除
    │   ├─ unroll_list=[1] 正常？ ──是──▶ 循环展开问题
    │   ├─ 调整 shape 正常？ ──是──▶ 可能合轴问题
    │   ├─ submit_before_loop 正常？ ──是──▶ 并行执行问题
    │   ├─ valid_shape 正确？ ──否──▶ 修正 valid_shape
    │   └─ +0.0 正常？ ──是──▶ 编译器优化问题
    │
    └─ 二分定位
        └─ 定位到具体 op ──▶ 分析该 op 实现问题
```

## 常见问题

### Q1: 所有排查都无效怎么办？

1. 检查 golden 实现是否正确
2. 使用最小用例（8-16 元素）验证
3. 分段验证中间结果
4. 检查是否有特殊边界条件

### Q2: 如何确认是框架问题还是算子实现问题？

1. 使用相同逻辑的 CPU 实现对比
2. 检查是否使用了不支持的 API
3. 查阅 `docs/api/` 确认 API 用法

## 检查清单

使用此 skill 时，确保：

- [ ] **步骤 0**：检查前端写法（最高优先级）
  - [ ] 检查是否使用 `pypto.jit`
  - [ ] 如使用旧写法，强烈建议切换到 `pypto.frontend.jit`

- [ ] **步骤 1**：收集必要信息
  - [ ] 算子代码路径
  - [ ] 测试用例路径
  - [ ] 精度误差描述
  - [ ] 问题复现步骤

- [ ] **步骤 1.5**：问题匹配与参考
  - [ ] 提取问题关键词
  - [ ] 匹配相似 Issue
  - [ ] 提供参考依据和链接

- [ ] **步骤 2**：基础检查
  - [ ] 输入初始化
  - [ ] 数据类型

- [ ] **步骤 3**：内存相关排查
  - [ ] 扩大 workspace 测试
  - [ ] Workspace 管理方式切换
  - [ ] 内存重叠检测

- [ ] **步骤 4**：特性排除
  - [ ] 设置 unroll_list=[1]
  - [ ] 检查合轴相关问题
  - [ ] 配置 submit_before_loop
  - [ ] 检查 valid_shape
  - [ ] 尝试 +0.0

- [ ] **步骤 5**：二分定位（如需要）

- [ ] **步骤 6**：深入分析（常规方法无效时）
  - [ ] 代码结构分析（数据流、风险点）
  - [ ] 运行流程分析（编译、内存、并行）
  - [ ] 基于实际情况给出精度问题推测

- [ ] **步骤 7**：输出结果
  - [ ] 问题类型
  - [ ] 问题位置
  - [ ] 问题原因
  - [ ] 修复建议

## 参考资料

### 参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| Issue 案例 | [references/issue-cases.md](references/issue-cases.md) | GitCode 已关闭精度 Issue 案例 |
| 定位方法 | [references/debugging-methods.md](references/debugging-methods.md) | 通用定位方法建议 |
| 深入分析 | [references/deep-analysis.md](references/deep-analysis.md) | 代码与运行流程分析方法 |

### 文档资料

| 文档 | 路径 | 说明 |
|------|------|------|
| Machine 错误排查 | `docs/trouble_shooting/machine.md` | 内存相关错误排查指南 |
| 精度验证技能 | `.agents/skills/pypto-operator-accuracy-verify/SKILL.md` | 精度验证方法 |
| 二分查找调试 | `.agents/skills/pypto-binary-search-verify/SKILL.md` | 二分定位精度问题 |
| 已知问题文档 | `docs/tutorials/appendix/issue.md` | 常见问题及解决方案 |
| 循环开发指南 | `docs/tutorials/development/loops.md` | loop 使用方法 |

### API 文档

| API | 文档路径 | 说明 |
|-----|---------|------|
| pypto.loop | `docs/api/controlflow/pypto-loop.md` | 循环接口，含 submit_before_loop 参数 |
| pypto.loop_unroll | `docs/api/controlflow/pypto-loop_unroll.md` | 循环展开接口，含 unroll_list 参数 |
| pypto.tensor | `docs/tutorials/development/tensor_creation.md` | Tensor 创建方法 |

### 代码参考

**submit_before_loop 用法**：
- `python/tests/st/operator/deepseek_v32/test_gather_after_prolog.py`
- `python/tests/ut/interface/test_pto_loop.py`

**unroll_list 用法**：
- `python/tests/ut/ds_v32/test_lightning_indexer_prolog.py`
- `python/tests/ut/ds_v32/test_lightning_indexer_topk.py`

**valid_shape 用法**：
- `python/tests/ut/ds_v32/test_lightning_indexer_prolog.py` (第 167-217 行)

**内存相关代码**：
- 内存重叠检测脚本: `tools/schema/schema_memory_check.py`
- workspace 管理: `framework/src/machine/runtime/device_launcher.cpp`
- 合轴标记逻辑: `framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.cpp`

**loop 控制器**：
- `python/pypto/_controller.py` (第 500-550 行)

### 工具脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 内存重叠检测 | `tools/schema/schema_memory_check.py` | 检测 device task 内存重叠 |
| CCE 变量解析 | `tools/scripts/auto_parse_cce_var.py` | 解析动态 CCE 中的 validShape |
| Tensor 解析 | `tools/verifier/parse_dump_tensors.py` | 解析 dump 的 tensor 信息 |

