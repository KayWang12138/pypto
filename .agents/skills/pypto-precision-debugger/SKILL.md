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

### Q3: BF16 精度损失如何判断是否正常？

BF16 正常精度损失范围：
- 相对误差：< 1% (rtol=0.01)
- 绝对误差：< 0.01 (atol=0.01)

如果超出此范围，需进一步排查。

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

---

## 典型案例

以下案例来自实际开发中的问题总结，可作为排查参考。

### 案例 1：使用未初始化的 Tensor

**问题现象**：使用 `pypto.tensor` 声明 Tensor 后直接读取，导致框架校验错误或精度问题。

**问题原因**：`pypto.tensor` 不包含初始化行为，未初始化的 Tensor 不申请内存。PyPTO 要求每个 Tensor 必须先写后读，即必须先有 producer，然后才能有 consumer。

**解决措施**：避免使用未经初始化的 Tensor，使用 `pypto.full`、`pypto.zeros` 等显式初始化接口。

```python
# 错误写法
t = pypto.tensor([32, 32])  # 未初始化
result = t.exp()  # 直接读取，可能出错

# 正确写法
t = pypto.zeros([32, 32])  # 显式初始化
result = t.exp()
```

---

### 案例 2：同一个算子多次执行时，静态轴传入不同的运行时值

**问题现象**：精度错误，或者 AI CPU/AI Core 异常。

**问题原因**：编译时针对静态轴进行固定大小切分，切分数量固定。如果传入的静态轴与首次编译不一致，切分后访问的内存地址可能超出实际 Tensor 大小，导致内存访问错误。

**解决措施**：

**方案 1**：针对不同静态值定义不同算子

```python
def handler(in_tensor):  # 定义公共处理函数
    return pypto.add(in_tensor, in_tensor)

@pypto.jit
def adder_256(in_shape_256):  # 处理 in 轴大小是 256 的场景
    return handler(in_shape_256)

@pypto.jit
def adder_1024(in_shape_1024):  # 处理 in 轴大小是 1024 的场景
    return handler(in_shape_1024)

adder_256(in_256)
adder_1024(in_1024)
```

**方案 2**：定义为动态轴

```python
@pypto.jit
def adder(in_shape):
    out = Tensor(in_shape.shape[0])
    for k in pypto.loop(in_shape.shape[0] / 256):
        out[k * 256: k * 256 + 256] = pypto.add(
            in[k * 256: k * 256 + 256],
            in[k * 256: k * 256 + 256])
    return out
```

---

### 案例 3：父循环内跨多个子循环的 Tensor 内存问题

**问题现象**：两层以上循环嵌套，父循环定义 tensor，一个子循环写入，另一个子循环使用时存在精度错误。

**问题原因**：父循环多次迭代分配的内存地址相同，不同迭代并行执行时内存覆盖。后执行的迭代会覆盖前一次迭代的临时内存。

**解决措施**：在后一个子循环中添加 `submit_before_loop=True`，强制多次迭代串行运行。

```python
for outer in pypto.loop(...):  # 父循环，执行至少两次
    t = pypto.Tensor(...)      # 定义一个临时 tensor
    for inner0 in pypto.loop(...):  # 第一个子循环，对临时 tensor t 赋值
        ...
        t[...] = ... 
    # 添加 submit_before_loop，确保父循环多次迭代不在同一个并行执行块中
    for inner1 in pypto.loop(..., submit_before_loop=True):  # 第二个子循环
        x[:] = t[:] + t[:]
```

---

## Issue 典型案例

> 以下案例来自 GitCode cann/pypto 仓库已关闭的精度 Issue（2026-02-01 至 2026-03-23）

### 问题匹配指南

根据用户描述的问题现象，快速匹配相似的 Issue 案例：

> **注意**：以下 Issue 大部分已修复，如使用最新版本仍遇到类似问题，可能是新 bug 或其他原因。

| 用户问题关键词 | 可能的问题类型 | 相关 Issue | 修复状态 | 推荐排查方法 |
|--------------|--------------|-----------|---------|-------------|
| reshape 后精度异常 | valid_shape 推导错误 | #498, #292, #813 | ✅ 已修复 | 检查 reshape 前后 valid_shape |
| view + reshape 组合 | inplace 元数据问题 | #343 | - | 避免 inplace=True |
| unroll 后精度失败 | RegisterCopy pass 错误 | #223, #341, #653 | ✅ 已修复 | 设置 unroll_list=[1] 测试 |
| 非整除场景精度失败 | valid_shape 推导 | #498, #341 | ✅ 已修复 | 检查整除/非整除差异 |
| loop 外 tensor 问题 | tensor 作用域问题 | #340, #626 | ✅ 已修复 | 在 loop 内创建 tensor |
| 动态轴结果不一致 | JIT 缓存问题 | #273, #529, #533 | ✅ 已修复 | 检查动态轴缓存 |
| 嵌套循环精度异常 | 并行执行内存覆盖 | #4 (案例4) | - | 添加 submit_before_loop=True |
| 特定 shape 精度失败 | buffer 限制/对齐问题 | #787, #724, #355 | ✅ 已修复 | 检查 buffer 大小限制 |
| mix 场景精度失败 | UB2L1 脏数据 | #799 | ✅ 已修复 | 检查 viewShape/validShape |
| 多 kernel 配置干扰 | 配置作用域泄漏 | #712 | ✅ 已修复 | 检查配置栈清理 |
| 精度工具通过但上板失败 | 仿真不覆盖 | #539 | - | 二分定位问题 op |
| assemble 相关问题 | Memtype 约束缺失 | #260 | ✅ 已修复 | 检查 Assemble Memtype |
| bf16 精度失败 | 舍入/对齐问题 | #355, #354 | ✅ 已修复 | 检查尾轴对齐 |
| 多核切 K 精度不确定 | GM 原子累加 | #795 | ✅ 已修复 | 检查多核累加逻辑 |
| inplace tensor 问题 | 内存处理错误 | #481, #468 | ✅ 已修复 | 检查 inplace 内存管理 |
| 精度工具粗检失败 | GC 内存回收 | #680 | ✅ 已修复 | 保存 tensor 引用 |

**使用建议**：
- 如使用最新版本代码，上述已修复问题不应再出现
- 如仍遇到类似问题，建议先确认代码版本，或考虑是新 bug
- 案例中的排查方法仍可作为定位思路参考

### Issue #799：UB2L1 脏数据污染

**修复状态**：✅ 已修复 (PR !1775, !1781)

**问题现象**：DeepSeek V3 SparseAttention antiquant 算子，打开 mix 场景后精度失败。

**问题原因**：UB2L1 搬运在 mix 场景下，当 `viewShape > align(validShape)` 时，搬运了多余的脏数据区域（UB 中未初始化的区域）混入 L1，导致后续计算结果错误。

**定位方法**：
1. 使用精度工具对比仿真与上板结果
2. 检查 viewShape 与 validShape 的关系
3. 定位到 UB2L1 操作的边界判断逻辑

**修复方法**：修复 UB2L1 操作的边界判断逻辑，正确处理 `viewShape > align(validShape)` 情况。

---

### Issue #787：Log1p/PReLU 超 UB 精度错误

**修复状态**：✅ 已修复 (PR !1716, !1747)

**问题现象**：Log1p/PReLU 算子运行报错 `exception aicore error`，或精度静默失败。

**问题原因**：Compare/TCMP 操作的临时缓冲区存在 **4096 字节硬限制**，当实际 shape 所需缓冲区超过 4096 字节时，发生 UB 越界。

**定位方法**：
1. 检查算子内部临时 buffer 大小计算
2. 对比 shape 与 4096 字节限制
3. 使用精度工具检测静默越界

**修复方法**：移除 4096 字节硬限制，改为根据实际 shape 动态计算 buffer 大小。

---

### Issue #498：pypto.reshape 丢失 valid_shape

**修复状态**：✅ 已修复 (PR !1385)

**问题现象**：reshape + matmul 场景，b 轴不能被 tile_b 整除时精度失败。

**问题原因**：`pypto.reshape` 在进行形状变换时，**未将上游 tensor 的 valid_shape 正确传播**到输出 tensor，导致 valid_shape 从动态符号退化为静态常量，后续 MTE 指令使用错误的 valid_shape 访问越界。

**定位方法**：
1. 对比整除与非整除场景的精度差异
2. 检查 reshape 前后 valid_shape 的值
3. 定位 MTE 越界问题

**修复方法**：修复 reshape pass 中的 valid_shape 推导逻辑，确保 reshape 操作正确传播上游 valid_shape 的动态符号信息。

---

### Issue #343：Reshape inplace=True 大偏差

**注意**：这是前端语法问题。

**问题现象**：`pypto.view` 后再做 `pypto.reshape(inplace=True)`，精度严重偏差（96%+ 元素超容差）。

**问题原因**：inplace=True 的 reshape 试图在原 tensor 上就地修改，但 view 的元数据（offset、valid_shape）未被正确传播，导致 reshape 结果指向错误的内存区域。

**定位方法**：
1. 对比 `inplace=False` 与 `inplace=True` 的精度差异
2. 检查 view + reshape 组合的元数据传播
3. 使用精度工具定位偏差来源

**修复方法**：避免 view 后使用 `reshape(inplace=True)`，或修复 inplace reshape 的元数据传播逻辑。

---

### Issue #340：pypto.full 在 loop 外创建导致精度问题

**修复状态**：✅ 已修复 (PR !1069)

**问题现象**：`pypto.full` 在循环外创建的 tensor 放在 assemble 中使用时精度失败，循环内创建则精度正确。

**问题原因**：`pypto.full` 在循环外创建的 tensor 在编译器的 RemoveRedundantOp/validShape 处理中，tensor 的 view attr 未被正确维护，在 unroll 场景下 tensor 被错误复用或 view 映射错误。

**定位方法**：
1. 对比 loop 内/外创建 tensor 的精度差异
2. 检查 unroll 场景下的 tensor 复用情况
3. 使用精度工具验证 view attr

**修复方法**：在 loop 内创建 full tensor，或修复 RemoveRedundantOp pass 对 validShape 的重用逻辑。

---

### Issue #273：动态轴第二次 call 按第一次 shape 执行

**修复状态**：✅ 已修复 (PR !1135)

**问题现象**：创建一次 kernel，两次 call，第二次 call 传入不同的动态轴 shape，但仍按第一次 compile 的 shape 执行，精度错误。

**问题原因**：新前端对动态轴的 JIT 缓存逻辑存在 bug：第二次调用时未重新编译，复用了第一次的编译产物（不含新动态轴的正确推导）。

**定位方法**：
1. 对比两次 call 的 shape 和输出
2. 检查 JIT 缓存是否正确失效
3. 打印编译日志确认是否重新编译

**修复方法**：修复 Parser 对动态 dimension 的缓存与重编译逻辑，确保动态轴变化时触发重新编译。

---

## 通用定位方法建议

### 精度工具使用

**1. 开启精度工具验证**

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}
)
```

**2. 使用 set_verify_golden_data**

```python
# 注意：必须保存 golden tensor 的引用，避免 GC 回收
golden_data = torch_output  # 保存引用
pypto.set_verify_golden_data(output=torch_output, golden=golden_data)
```

**3. 精度工具校验通过但上板失败的处理**

当精度工具校验通过但上板结果不符时：
- 检查仿真路径是否覆盖实际运行路径
- 使用二分定位法定位问题 op
- 检查 Pass 变换是否在仿真与上板间存在差异

### 二分定位法

使用 `pypto-binary-search-verify` skill 进行二分定位：

```bash
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

**步骤**：
1. 在计算图中添加中间输出检查点
2. 对比每个检查点的 tensor 与 golden
3. 定位精度偏差首次出现的 op

### 常见定位技巧

**1. 对比不同写法**

```python
# 写法1：简洁写法
for i in pypto.loop(N):
    result = compute(a, i)

# 写法2：手动展开写法
for i in pypto.loop(N, unroll_list=[1]):  # 关闭展开
    result = compute(a, i)
```

**2. 检查动态轴处理**

```python
# 手动取出动态轴（规避写法）
c_shape_0 = a_tensor.shape[0]  # 手动取出
c_tensor = pypto.Tensor((c_shape_0, z), dtype)
```

**3. 检查 valid_shape 传播**

```python
# 打印 reshape 前后的 valid_shape
print(f"Before reshape: valid_shape = {input_tensor.valid_shape}")
output = pypto.reshape(input_tensor, new_shape)
print(f"After reshape: valid_shape = {output.valid_shape}")
```

**4. 检查配置作用域**

```python
# 多 kernel 场景，检查配置是否泄漏
@pypto.jit(combine_axis=...)
def kernel1(...): ...

@pypto.jit()  # 不应继承 kernel1 的 combine_axis
def kernel2(...): ...
```

### 问题定位决策树

```
精度问题
    │
    ├─ 是否与 shape 相关？
    │   ├─ 整除/非整除差异 → 检查 valid_shape 推导
    │   ├─ 静态/动态轴差异 → 检查动态轴缓存和 outcast
    │   └─ 特定 shape 失败 → 检查 buffer 限制、对齐要求
    │
    ├─ 是否与循环相关？
    │   ├─ unroll 后失败 → 设置 unroll_list=[1] 测试
    │   ├─ 嵌套循环失败 → 添加 submit_before_loop=True
    │   └─ loop 外 tensor 问题 → 检查 tensor 作用域
    │
    ├─ 是否与内存操作相关？
    │   ├─ view/reshape 组合 → 检查 inplace 参数
    │   ├─ assemble 问题 → 检查 Memtype 约束
    │   └─ 多 kernel 干扰 → 检查配置作用域
    │
    └─ 是否与精度工具相关？
        ├─ 仿真通过上板失败 → 二分定位问题 op
        ├─ 粗检失败 → 检查 tensor 生命周期
        └─ precheck/postcheck 误判 → 检查 Pass 校验逻辑
```

---

## 深入分析：代码与运行流程分析

**⚠️ 重要：当常规排查方法无法定位问题时，agent 应主动深入分析问题代码，从运行流程角度给出精度问题推测。**

### 触发条件

当出现以下情况时，应启动深入分析：
- 问题匹配指南中无相似 Issue
- 常规排查步骤均无效
- 问题现象特殊，无法直接归类

### 分析流程

#### 1. 代码结构分析

**分析目标**：理解算子的计算逻辑和数据流

**分析步骤**：

1. **阅读算子代码**
   - 理解算子的数学公式
   - 识别关键计算步骤
   - 标注每个 tensor 的 shape 和 dtype

2. **绘制数据流图**
   ```
   输入 tensor → 步骤1 → 中间 tensor → 步骤2 → ... → 输出 tensor
   ```

3. **识别潜在风险点**
   - 循环结构（是否有 unroll、嵌套）
   - 内存操作（view、reshape、assemble）
   - 动态轴使用
   - 特殊数据类型（bf16、fp16）

#### 2. 运行流程分析

**分析目标**：理解代码在 NPU 上的实际执行过程

**分析维度**：

| 维度 | 分析内容 | 可能的问题 |
|-----|---------|-----------|
| **编译阶段** | JIT 编译、Pass 变换 | Pass 优化可能引入问题 |
| **内存分配** | tensor 内存布局、workspace | 内存重叠、越界访问 |
| **并行执行** | 多核切分、循环展开 | 数据竞争、内存覆盖 |
| **数据搬运** | GM↔UB↔L1 数据流 | 搬运边界、脏数据 |

**分析方法**：

1. **检查编译产物**
   ```bash
   # 查看生成的 CCE 代码
   export PTO_DUMP_CCE=1
   # 运行算子后查看生成的 .cce 文件
   ```

2. **分析 Pass 变换**
   - 检查是否触发了有问题的 Pass（如 RegisterCopy、RemoveRedundantOp）
   - 对比开启/关闭特定 Pass 的结果

3. **追踪数据流**
   - 在关键位置添加中间输出
   - 对比每一步的 tensor 值与 golden

#### 3. 基于实际情况的推测

**推测方法**：

1. **对比分析法**
   - 对比正常 case 和异常 case 的差异
   - 找出导致差异的关键因素

2. **最小复现法**
   - 逐步简化算子代码
   - 找到最小复现用例

3. **排除法**
   - 逐个排除可能的原因
   - 确认每个排除步骤的结果

**输出格式**：

```markdown
## 深入分析报告

### 代码结构分析
- **算子功能**：[描述算子的数学公式/功能]
- **数据流**：[描述输入→计算→输出的流程]
- **关键步骤**：
  1. 步骤1：[描述]
  2. 步骤2：[描述]
  ...

### 潜在风险点
- [ ] 循环展开（unroll_list 配置）
- [ ] 内存操作（view/reshape/assemble 组合）
- [ ] 动态轴处理
- [ ] 数据类型转换
- [ ] 其他：[具体描述]

### 运行流程分析
- **编译阶段**：[描述可能的问题]
- **内存分配**：[描述可能的问题]
- **并行执行**：[描述可能的问题]

### 精度问题推测
基于以上分析，推测可能的原因为：

1. **主要推测**：[描述最可能的原因]
   - 依据：[说明推测的依据]
   - 验证方法：[如何验证这个推测]

2. **次要推测**：[描述其他可能的原因]
   - 依据：[说明推测的依据]
   - 验证方法：[如何验证这个推测]

### 建议的排查步骤
1. [具体步骤1]
2. [具体步骤2]
...
```

### 分析示例

**用户问题**：某算子在 shape=(2, 1024, 64) 时精度正常，shape=(2, 1000, 64) 时精度失败。

**深入分析过程**：

1. **代码结构分析**
   - 算子包含 reshape 操作：`x.reshape([b, s, h])`
   - 数据流：input → reshape → compute → output
   - 风险点：reshape 操作

2. **运行流程分析**
   - 1024 可被 tile_size 整除，1000 不可整除
   - 非整除场景可能触发 valid_shape 推导问题

3. **精度问题推测**
   - 主要推测：reshape 在非整除场景下 valid_shape 推导错误
   - 依据：Issue #498 有相似现象
   - 验证方法：检查 reshape 前后 valid_shape 值

4. **建议排查步骤**
   - 打印 reshape 前后的 valid_shape
   - 对比 1024 和 1000 场景的 valid_shape 差异
   - 尝试手动设置 valid_shape 参数
