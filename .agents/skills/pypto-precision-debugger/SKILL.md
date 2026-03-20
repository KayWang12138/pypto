---
name: pypto-precision-debugger
description: |
  PyPTO 算子精度问题排查技能。专注于用户代码层面的语法逻辑检查和规避方法尝试。当算子精度验证失败、输出结果异常或用户请求精度问题排查时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题排查技能

此技能专注于**用户代码层面**的精度问题排查，提供语法逻辑检查和规避方法。

## 核心定位

### ⭐ 职责范围

**本技能负责**：
- 用户代码语法逻辑检查
- 提供规避方法和工具
- 尝试各种可能的解决方案

**本技能不负责**：
- 底层框架代码分析
- 深层根因定位
- 框架 Bug 确认

**结论判断**：
- 如果规避方法有效 → 提供解决方案
- 如果规避方法无效 → 报告可能是底层框架层面问题，建议提 Issue

---

## 常见规避方法速查表

**⚠️ 遇到精度问题时，优先尝试以下规避方法：**

| 优先级 | 问题现象 | 规避方法 | 代码示例 | 依据参考 |
|-------|---------|---------|---------|---------|
| 1 | 使用旧前端写法 | 切换到 `pypto.frontend.jit` | `@pypto.frontend.jit` | 新前端是 PyPTO 推荐写法，旧前端已不再维护 |
| 2 | view + reshape 精度异常 | 避免 `inplace=True` | `pypto.reshape(tensor, shape, inplace=False)` | Issue #343 |
| 3 | 循环展开后精度异常 | `unroll_list=[1]` | `pypto.loop(range(n), unroll_list=[1])` | Issue #223, #341 |
| 4 | 嵌套循环精度异常 | `submit_before_loop=True` | `pypto.loop(range(m), submit_before_loop=True)` | `docs/api/controlflow/pypto-loop.md` |
| 5 | 特定 shape 精度异常 | 调整 shape | 避免尾轴为 1，避免非整除 | Issue #498, #787 |
| 6 | 编译器优化异常 | `+0.0` 技巧 | `result = compute(...) + 0.0` | 阻止过度优化 |
| 7 | workspace 不足 | 扩大 workspace | `pypto.set_workspace_size(larger_size)` | `docs/trouble_shooting/machine.md` |
| 8 | 内存重叠 | 调整分配顺序 | 避免原地修改 | `tools/schema/schema_memory_check.py` |

---

## ⭐ 重要提示：使用新前端写法

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
    ...

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

---

## 排查决策树

```
精度问题
    │
    ├─ 步骤 0：前端写法检查（最高优先级）
    │   └─ 使用 pypto.jit？ ──是──▶ 切换到 pypto.frontend.jit 重试
    │
    ├─ 步骤 1：快速规避方法尝试
    │   ├─ 避免 view + reshape inplace=True
    │   ├─ unroll_list=[1]
    │   ├─ submit_before_loop=True
    │   └─ +0.0 技巧
    │
    ├─ 步骤 2：用户代码语法检查
    │   ├─ 输入初始化正确？
    │   ├─ 数据类型正确？
    │   ├─ shape 定义正确？
    │   └─ valid_shape 配置正确？
    │
    ├─ 步骤 3：内存相关检查
    │   ├─ workspace 大小足够？
    │   ├─ 内存重叠？
    │   └─ buffer 使用正确？
    │
    ├─ 步骤 4：二分定位（如需要）
    │
    └─ 步骤 5：结论判断
        ├─ 规避方法有效 ──▶ 提供解决方案
        └─ 规避方法无效 ──▶ 报告可能是框架问题
```

---

## 完整工作流程

### 步骤 0：检查前端写法（最高优先级）

**⚠️ 最高优先级：首先检查用户是否使用了旧前端写法！**

**检查内容**：查看用户代码中的装饰器是 `pypto.jit` 还是 `pypto.frontend.jit`

**如果用户使用 `pypto.jit`**：

1. **立即建议切换到新前端**
2. 提供修改示例：

```python
# 修改前
@pypto.jit
def my_kernel(input_tensor, output_tensor):
    ...

# 修改后
@pypto.frontend.jit
def my_kernel(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):
    ...
```

3. 重新运行测试验证精度

**如果问题解决**：结束排查，建议用户使用新前端写法

**如果问题未解决**：继续下一步

---

### 步骤 1：快速规避方法尝试

**⚠️ 重要：以下规避方法按优先级逐一尝试，记录每种方法的效果。**

#### 1.1 避免 view + reshape inplace=True

**适用场景**：`pypto.view` 后再做 `pypto.reshape(inplace=True)` 导致精度严重偏差

**问题现象**：精度严重偏差（96%+ 元素超容差）

**操作**：

```python
# ❌ 避免写法
tensor_view = pypto.view(tensor, new_shape, ...)
result = pypto.reshape(tensor_view, final_shape, inplace=True)  # 精度严重偏差

# ✅ 推荐写法
tensor_view = pypto.view(tensor, new_shape, ...)
result = pypto.reshape(tensor_view, final_shape, inplace=False)  # 精度正确
```

**判断**：问题是否解决

**依据参考**：Issue #343 - `pypto.view` 后再做 `pypto.reshape(inplace=True)` 时，内存地址或元数据处理错误，inplace=True 的 reshape 试图在原 tensor 上就地修改，但 view 的元数据（offset、valid_shape）未被正确传播，导致 reshape 结果指向错误的内存区域。

#### 1.2 尝试 unroll_list=[1]

**适用场景**：循环展开后精度异常

**操作**：

```python
# 在循环中添加 unroll_list=[1]
for i in pypto.loop(range(n), unroll_list=[1]):
    ...
```

**判断**：问题是否解决

**依据参考**：Issue #223, #341 - RegisterCopy pass 在处理 unroll 场景时，对 tensor 的寄存器拷贝逻辑存在 bug，某些 unroll 配置下 tensor 的 Copy 路径选择错误，导致数据写入位置偏移。设置 `unroll_list=[1]` 可关闭循环展开，规避此问题。

#### 1.3 尝试 submit_before_loop=True

**适用场景**：嵌套循环精度异常

**操作**：

```python
# 在子循环中添加 submit_before_loop=True
for i in pypto.loop(range(n)):
    for j in pypto.loop(range(m), submit_before_loop=True):
        ...
```

**判断**：问题是否解决

**依据参考**：`docs/api/controlflow/pypto-loop.md` - `submit_before_loop` 参数用于控制嵌套循环的调度行为，确保子循环在父循环迭代前正确提交，避免并行执行时的内存覆盖问题。

#### 1.4 尝试 +0.0 技巧

**适用场景**：编译器优化导致精度异常

**操作**：

```python
# 在计算结果上添加 + 0.0
result = compute(...) + 0.0
```

**判断**：问题是否解决

**依据参考**：编译器优化 pass 在某些场景下会错误地消除或重排序操作，添加 `+0.0` 可以阻止过度优化，保留计算操作的完整性。

#### 1.5 尝试调整 shape

**适用场景**：特定 shape 精度异常

**操作**：

- 调整 tensor shape，避免尾轴为 1
- 避免非整除情况
- 尝试不同的 shape 组合

**判断**：问题是否解决

**依据参考**：Issue #498, #787 - 特定 shape（如尾轴为 1、非整除）可能触发 Pass 推导的边界情况，导致 valid_shape 传播错误或 buffer 越界。调整 shape 可规避这些边界场景。

---

### 步骤 2：用户代码语法检查

#### 2.1 输入初始化检查

**检查项**：

- [ ] 输入 tensor 是否正确初始化
- [ ] 输出 tensor 是否正确分配
- [ ] 中间 tensor 是否需要初始化

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| 输入未初始化 | 随机值导致精度异常 | 确保输入数据正确初始化 |
| 输出未清零 | 残留数据干扰 | 使用 `pypto.zeros` 初始化输出 |

**示例**：

```python
# ✅ 正确：输入初始化
input_tensor = pypto.Tensor(shape, pypto.DT_FP32)
input_tensor.from_numpy(np_input)  # 确保输入数据正确

# ✅ 正确：输出初始化
output_tensor = pypto.zeros(shape, pypto.DT_FP32)
```

#### 2.2 数据类型检查

**检查项**：

- [ ] 输入输出数据类型是否一致
- [ ] 中间计算是否需要更高精度
- [ ] 是否存在隐式类型转换

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| FP16 精度损失 | 大数值计算误差 | 使用 FP32 或调整计算顺序 |
| 类型不匹配 | 计算结果异常 | 确保类型一致 |

#### 2.3 Shape 定义检查

**检查项**：

- [ ] shape 是否正确定义
- [ ] 动态轴是否正确标记
- [ ] valid_shape 是否与 shape 一致

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| shape 与实际数据不匹配 | 越界或数据截断 | 检查 shape 定义 |
| 动态轴标记错误 | 编译或运行时错误 | 检查动态轴定义 |

#### 2.4 valid_shape 配置检查

**检查项**：

- [ ] view/reshape 后 valid_shape 是否正确
- [ ] 动态 shape 场景下 valid_shape 表达式是否正确
- [ ] assemble 后 valid_shape 是否更新

**示例**：

```python
# ✅ 正确：view 后设置 valid_shape
tensor_view = tensor.view(new_shape)
tensor_view.set_valid_shape(valid_shape_expr)
```

---

### 步骤 3：内存相关检查

#### 3.1 Workspace 大小检查

**检查项**：

- [ ] workspace 是否足够大
- [ ] 是否存在 workspace 越界

**规避方法**：

```python
# 方法 1：扩大 workspace
pypto.set_workspace_size(larger_size)

# 方法 2：切换 workspace 管理方式
pypto.set_workspace_manager("internal")  # 或 "external"
```

**判断**：扩大 workspace 后问题是否解决

#### 3.2 内存重叠检测

**检查项**：

- [ ] 是否存在 tensor 内存重叠
- [ ] 并行执行时是否有数据竞争

**检测工具**：

```bash
# 使用内存重叠检测脚本
python tools/schema/schema_memory_check.py --input your_kernel.cce
```

**规避方法**：

- 调整 tensor 分配顺序
- 避免原地修改操作

#### 3.3 Buffer 使用检查

**检查项**：

- [ ] buffer 大小是否足够
- [ ] buffer 访问是否越界

---

### 步骤 4：二分定位（如需要）

如果上述方法无法定位问题，使用二分查找定位具体问题 op：

```bash
# 调用 pypto-binary-search-verify skill
# 或使用其脚本
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

详见 [pypto-binary-search-verify](../pypto-binary-search-verify/SKILL.md)。

---

### 步骤 5：结论判断

#### 情况 A：规避方法有效

**输出**：

```markdown
## 排查结论

### 问题类型
[描述问题类型]

### 有效的规避方法
[描述有效的规避方法]

### 解决方案
[提供具体的代码修改建议]

### 注意事项
[使用规避方法时的注意事项]
```

#### 情况 B：规避方法无效

**输出**：

```markdown
## 排查结论

### 问题现象
[描述问题现象]

### 已尝试的规避方法
1. [方法1] - 无效
2. [方法2] - 无效
3. [方法3] - 无效
...

### 结论
经过多种规避方法尝试，问题仍未解决。**可能是底层框架层面的问题**。

### 建议
1. 在 GitCode 上提交 Issue，附上复现步骤和最小用例
2. 提供算子代码、测试用例、环境信息
3. 说明已尝试的规避方法
```

---

## 检查清单

使用此 skill 时，确保：

- [ ] **步骤 0**：检查前端写法（最高优先级）
  - [ ] 检查是否使用 `pypto.jit`
  - [ ] 如使用旧写法，建议切换到 `pypto.frontend.jit`

- [ ] **步骤 1**：快速规避方法尝试
  - [ ] 避免 view + reshape inplace=True
  - [ ] unroll_list=[1]
  - [ ] submit_before_loop=True
  - [ ] +0.0 技巧
  - [ ] 调整 shape

- [ ] **步骤 2**：用户代码语法检查
  - [ ] 输入初始化
  - [ ] 数据类型
  - [ ] Shape 定义
  - [ ] valid_shape 配置

- [ ] **步骤 3**：内存相关检查
  - [ ] workspace 大小
  - [ ] 内存重叠
  - [ ] buffer 使用

- [ ] **步骤 4**：二分定位（如需要）

- [ ] **步骤 5**：结论判断
  - [ ] 规避方法有效 → 提供解决方案
  - [ ] 规避方法无效 → 报告可能是框架问题

---

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

### 工具脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 内存重叠检测 | `tools/schema/schema_memory_check.py` | 检测 device task 内存重叠 |
| CCE 变量解析 | `tools/scripts/auto_parse_cce_var.py` | 解析动态 CCE 中的 validShape |
| Tensor 解析 | `tools/verifier/parse_dump_tensors.py` | 解析 dump 的 tensor 信息 |
