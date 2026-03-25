---
name: pypto-precision-debugger
description: |
  PyPTO 算子精度问题排查技能。通过系统化排查流程定位精度问题根因，包括 workspace、unroll、合轴、内存重叠等常见问题。当算子精度验证失败、输出结果异常或用户请求精度问题排查时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题排查技能

此技能提供系统化的精度问题排查流程，帮助定位 PyPTO 算子精度问题的根本原因。

## 核心原则

### 排查原则

1. **先易后难**：从简单检查开始，逐步深入
2. **隔离变量**：每次只修改一个配置，确认效果
3. **记录过程**：记录每步排查结果，便于回溯
4. **定位根因**：找到问题后确认根因，而非绕过问题

### 常见精度问题类型

| 问题类型 | 典型现象 | 排查方法 |
|---------|---------|---------|
| workspace 不足 | 输出随机异常 | 扩大 workspace 测试 |
| 循环展开问题 | 循环场景精度异常 | 设置 unroll_list=[1] |
| 合轴问题 | 特定 shape 精度异常 | 检查尾轴为1的tensor |
| 并行执行问题 | 嵌套循环精度异常 | 配置 submit_before_loop=True |
| 内存重叠 | 输出数据错乱 | 内存重叠检测 |
| valid_shape 错误 | view/reshape 异常 | 检查 valid_shape 配置 |
| 编译器优化问题 | 特定 op 结果异常 | 尝试 +0.0 |

## 完整工作流程

### 步骤 1：收集必要信息

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **operator_path**: 算子代码路径
- **test_path**: 测试用例路径
- **error_description**: 精度误差描述（误差大小、异常元素分布等）
- **reproduce_steps**: 问题复现步骤

### 步骤 2：基础检查

#### 2.1 检查输入初始化

确保输入/输出 tensor 已正确初始化：

```python
# 检查输入是否已初始化
assert input_tensor is not None, "Input tensor is None"
assert not torch.isnan(input_tensor).any(), "Input contains NaN"

# 检查输出是否已初始化
assert output_tensor is not None, "Output tensor is None"
```

#### 2.2 检查 Tensor 连续性

部分算子要求输入/输出在指定内存布局下连续：

```python
# 检查 tensor 连续性
print(f"Input is_contiguous: {input_tensor.is_contiguous()}")
print(f"Output is_contiguous: {output_tensor.is_contiguous()}")

# 如果不连续，尝试转为连续
if not input_tensor.is_contiguous():
    input_tensor = input_tensor.contiguous()
```

#### 2.3 检查数据类型

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

### 步骤 6：输出结果

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
    ├─ 基础检查
    │   ├─ 输入初始化？ ──否──▶ 初始化输入
    │   ├─ Tensor 连续？ ──否──▶ 转为连续
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

- [ ] **步骤 1**：收集必要信息
  - [ ] 算子代码路径
  - [ ] 测试用例路径
  - [ ] 精度误差描述
  - [ ] 问题复现步骤

- [ ] **步骤 2**：基础检查
  - [ ] 输入初始化
  - [ ] Tensor 连续性
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

- [ ] **步骤 6**：输出结果
  - [ ] 问题类型
  - [ ] 问题位置
  - [ ] 问题原因
  - [ ] 修复建议

## 参考资料

当 workflow 或实现阶段怀疑问题与 `valid_shape`、`unroll_list`、workspace、合轴或内存重叠有关时，优先转入本 skill。

### 文档资料

| 文档 | 路径 | 说明 |
|------|------|------|
| Machine 错误排查 | `docs/trouble_shooting/machine.md` | 内存相关错误排查指南 |
| 精度验证入口 | `test_{op}.py` 与 `{op}_golden.py` | 基于 golden 的精度验证方法 |
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

### 案例 2：同一个 Tensor 进行 View 和 Assemble 导致图成环报错

**问题现象**：从 Tensor a 中 view 获取数据，计算后 assemble 写回 a，报错：

```
ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
```

**问题代码**：

```python
@pypto.jit
def foo_kernel(x, y):
    pypto.set_vec_tile_shapes(16, 16)
    a = pypto.zeros([32, 32])
    b = a[:16, :16]           # 从 a 中 view 获取数据
    a[16:, 16:] = b.exp()     # 计算后 assemble 写回 a
    y[:] = x + a
```

**问题原因**：数据从 a 中读取又写回 a，形成环路 `a→b→b.exp()→a`。PyPTO 不允许图内存在环路，必须为 DAG（有向无环图）。

**解决措施**：将读取和写入 a 的逻辑拆分成两个图去定义。

```python
@pypto.jit
def foo_kernel(x, y):
    pypto.set_vec_tile_shapes(16, 16)
    a = pypto.zeros([32, 32])
    b = pypto.zeros([16, 16])  # 使用独立的 tensor
    b[:] = a[:16, :16]         # 从 a 中读取
    a[16:, 16:] = b.exp()      # 写回 a
    y[:] = x + a
```

---

### 案例 3：同一个算子多次执行时，静态轴传入不同的运行时值

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

### 案例 4：父循环内跨多个子循环的 Tensor 内存问题

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

### 案例 5：SymbolicScalar 不支持循环内自增

**问题现象**：循环内 `count = count + 1` 不会按预期递增。

```python
@pypto.jit
def add_kernel_1(a, b, c):
    count = 0
    for i in pypto.loop(20):
        count = count + 1  # 不会按预期递增
```

**问题原因**：PyPTO 框架仅捕获 Tensor 操作，未捕获 scalar 操作，因此不会将 count 处理为变量。只有循环变量能够实现自增。

**解决措施**：使用循环变量来表达自增逻辑。

```python
@pypto.jit
def add_kernel_1(a, b, c):
    for i in pypto.loop(20):
        # 使用循环变量 i 代替 count
        result = compute_with_index(a, i)
```

---

### 案例 6：已安装 torch_npu 但未安装 CANN 时执行仿真异常

**问题现象**：报错 `ImportError: libhccl.so: cannot open shared object file: No such file or directory`

**问题原因**：torch（版本 > 2.5）会自动加载所有名为 "torch.backends" 的扩展。如果环境中已安装 torch_npu 但未安装 CANN，找不到依赖项会引发异常。

**解决措施**：设置环境变量避免自动加载：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

---

## 独立使用

当用户直接调用本 Skill 时：
1. 从用户输入提取算子代码路径、测试路径、精度误差描述等必要信息
2. 如果信息不足，向用户逐步提问补充
3. 按排查决策树系统化执行精度问题排查
4. 输出排查结果报告（问题类型、位置、原因、修复建议）
