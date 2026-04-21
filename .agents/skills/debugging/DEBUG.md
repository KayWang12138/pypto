# Debug 手册 —— 遇到晦涩错误码时不要停下来

**受众：** 在本 skill 集下开发复杂 PyPTO kernel 的 Agent。

**不要跳过：** 当遇到问题时，本文件是必读路径的一部分 —— 参见 **`skills/lead-orchestrator/references/rules.md` → Zero tolerance**。

**策略：** **`Errcode: FFFFF!`**、**`UNKNOWN`**、**`0x3FFFF`** 或其他含义不明/看起来可怕的错误码**不是**结束 session、报告失败或"放弃"的理由，除非 `skills/lead-orchestrator/references/rules.md` 中的**停止条件**确实生效。**Token 和轮次预算不是此工作流的约束**：持续迭代，收集证据，尝试下一个结构化策略，直到问题解决或确认存在真正的阻塞项。

**什么是"放弃"（此处禁止）：** 仅打印错误码后就停止、未查阅文档就宣布任务不可能、跳过日志、或拒绝尝试其他角度。

---

## 1. 首要操作（始终执行）

1. **捕获完整消息** —— stderr、Python traceback，以及任何 **`pypto-log*.log`** / 设备日志行。搜索 `Errcode`、`ErrCode`、`F` + 数字、`aicore`。
2. **路由错误码** —— 阅读 `docs/trouble_shooting/README.md` 并打开对应前缀的组件文档（例如 FUNCTION / `docs/trouble_shooting/function.md` 用于许多 `F2xxxx` 格式的错误码）。对于 **`F21004`** / **`REGISTER_COPY`** / 无效 vector tile，先参见下方 **§4**。
3. **追加到 `custom/plan/<operator_name>.md` → Development & debug log** —— 记录失败内容、命令、假设、下一步。不要留空的"已停止"结尾。

---

## 2. UNKNOWN / FFFFF / 细节不足

- 按相关 `docs/trouble_shooting/*.md` 中的说明启用 **verbose logging**（例如 `ASCEND_GLOBAL_LOG_LEVEL`、日志路径 —— FUNCTION 相关错误参见 `function.md`）。
- 如果怀疑计算图有问题：遵循故障排除文档中链接的 **computation-graph / program-dump** 指南（上游文档可能称之为"view computation graph"）。
- **不要**将"unknown"视为终止状态；应将其视为**需要更多信号**（日志、更小的复现用例、更早的检查点）。

---

## 3. Kernel 专属：缩小影响范围

1. 确认当前 **staged file** 是哪个（`<op>_module12.py` 等 —— 参见 **`skills/lead-orchestrator/references/rules.md`** 规则 14）。在**该**文件（或最终 kernel）上运行 **`extract_pypto_calls.py`**。
2. 运行 **`python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py <kernel.py>`**（本仓库中的规范路径；某些检出版本记录为 `.agents/skills/ci-and-layout-check/scripts/` —— 使用实际存在的路径），并遵循 `skills/debugging/SKILL.md` 中的 **op-by-op check protocol**。
3. 每次只处理**一个活跃模块**；如有需要，用 golden 喂入的 tensor 填充下游。
4. 使用 **`detailed_tensor_compare`** 重新运行 **模块边界**检查，并在 **Per-module verification log** 中记录结果。
5. 修复后，重新运行 **当前 staged file** 和/或 **`test_<operator_name>.py`**，并确认**每一个** kernel 输出（参见 **`skills/lead-orchestrator/references/rules.md`** 规则 10）—— 不仅仅是第一个 tensor。

### 3a. Layout CI 和 Pass 回归（快速指引）

- **Layout / `pypto_function` 循环：** 在 **`custom/`** 下进行有意义的编辑后，使用 **`skills/ci-and-layout-check/CI.md`** 中的 **`bash`** 命令从仓库根目录运行 **`skills/ci-and-layout-check/run_validate_layout.sh`**（也在 **`skills/lead-orchestrator/references/rules.md`** 中引用）。**Exit 1** 意味着需要修复 plan / `test_<op>.py` / staged 命名问题，**或者**移除 `pypto_function` 内部的 **`for ... in range(...)`** —— 在 **`_your_op_kernel_impl`** / JIT kernel 中使用 **`pypto.loop`** 来表达该迭代，遵循 **`skills/lead-orchestrator/references/rules.md`** 禁令 B / 规则 18。
- **图编辑后立即出现 Pass / 编译失败：** 如果日志显示 **PASS** 范围的错误码（**`F4` / `F5`**，参见 **`docs/trouble_shooting/README.md`** → **`pass.md`**），遵循 **`.agents/skills/pypto-pass-error-fixer/SKILL.md`**，对 PyPTO 图进行 **bisect**（例如最后已知正确的 **staged** 文件与当前版本对比），并在大规模重写之前重新检查 API 约束（**`query_op_index`** / `docs/`**）**。在失败文件上重新运行 **`extract_pypto_calls.py`**，查看是否是新的 op 顺序触发了 Pass。

---

## 4. F21004（`INVALID_VAL` / `TileShape::Current().GetVecTile()` 无效）—— `REGISTER_COPY` 和 vector tile shapes

**原因（框架）：** 当 **`TileShape::Current().GetVecTile()`** 无效时，在 **`Operation` 的构造函数** 中抛出 **`F21004`**（例如 `framework/src/interface/operation/operation.cpp` 约 191–195 行）。

**为什么 `REGISTER_COPY` 出现：** **`REGISTER_COPY`** 是一个 **AIV** op。即使 **编译器插入** 该 op（例如内存冲突 Pass），它也需要**有效的 vector tile**。没有单独的"REGISTER_COPY tiling"注册——适用相同的 **vec tile** 规则。

**`VecTile` 何时有效？** 存储的列表必须**非空**且**每个值必须 > 0**（`tile_shape.cpp`）。

### 修复 —— 在 PyPTO Python 中怎么做

1. **在该作用域内进行任何 vector/tensor 操作之前**（包括最终导致 **`REGISTER_COPY`** 的代码），设置 **vector** tile 大小：

   ```python
   import pypto
   pypto.set_vec_tile_shapes(1, 1, 128, 128)   # 按文档使用正整数
   ```

2. 在 **`@jit` / kernel 函数体** 的**开头**调用它，如果你的 API 使用**嵌套作用域**，则在**任何作用域变更之后再次调用**。

3. **不要仅依赖 `set_cube_tile_shapes`** —— cube tile **不能**替代 AIV op（如 **`REGISTER_COPY`**）的 vector tile。

4. 如果同时使用 **cube 和 vector op**：

   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
   pypto.set_vec_tile_shapes(1, 1, 128, 128)
   ```

### 如果仍然失败

- **参数无效或为零** —— 确保所有参数为正整数。参考 **`docs/api/config/pypto-set_vec_tile_shapes.md`** 了解你的 PyPTO 版本要求（参见 **`skills/lead-orchestrator/references/rules.md`** 规则 16 / **`skills/phase4-phase5-integration/SKILL.md`** 5.4b）。
- **顺序错误** —— 任何**添加 op** 的操作（reshape、view、Pass 插入 copy）必须在该执行路径上在 **`set_vec_tile_shapes`** **之后**运行。
- **符号/动态 shape** —— 确保 tile 参数解析为**具体的正整数**（参见你的代码树中 `python/pypto/_controller.py` 中的 `set_vec_tile_shapes` + `SymbolicScalar`）。

**结论：** **`REGISTER_COPY` 上的 `F21004`** 几乎总是**"创建此 op 时当前作用域中没有有效的 `vec_tile_shapes`。"** 在这些 op 运行之前，使用 **`pypto.set_vec_tile_shapes(...)`** 并传入**全部正数大小**来修复。

---

## 5. 当日志指向设备 / AICore 时

- 当失败是 **aicore error** / 设备端 trace 问题时，加载 **`.agents/skills/pypto-aicore-error-locator/SKILL.md`** 并遵循其步骤。

---

## 6. 错误码快速参考（仓库）

- `.agents/skills/pypto-op-develop/references/error-code-troubleshooting.md` —— `Errcode: Fxxxxx!` 的处理流程。
- `docs/trouble_shooting/function.md` —— 例如 **INVALID_VAL (0x21004)**、**UNKNOWN (0x3FFFF)**。
- **F21004 / vec tile / `REGISTER_COPY`：** 参见上方 **§4**。

---

## 7. 何时允许停止

仅当符合 `skills/lead-orchestrator/references/rules.md` 中的**停止条件**时（缺少参考资料、不可能的 golden、根本性的框架阻塞、需要时**缺少**用户提供的日志、或在穷尽所有结构化尝试后**确认的**盲目猜测）。**一条晦涩的错误行永远不够。**

---

## 8. 示例 kernel —— 调试实践（来自 `examples/`）

本节浓缩了 **`.agents/pypto-example-debug-practice/DEBUG_PRACTICE.md`**：在 **`examples/`** 下重新推导可运行 kernel 的笔记（盲写树 **`custom/debug_scratch_examples/`** 与官方脚本的对比）。它补充了 **§1–§7**；按 **`examples/README.md`** 运行示例（`--list`、`--run_mode sim`、NPU 使用 **`TILE_FWK_DEVICE_ID`**）。错误码路由仍然遵循 **`docs/trouble_shooting/README.md`**。

### 8.1 全局模式（去重）

- **运行模式和装饰器：** 大多数示例设置 `global_run_mode = pypto.RunMode.NPU` 然后通过 **`_peek_run_mode_from_argv`** 覆盖，这样模块级的 `@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})` 在 `python3 script.py --run_mode sim` 时能看到 **`sim`**。仅在 **`main()`** 内解析 **`--run_mode`** 会使装饰器绑定到导入时的模式——行为错误或出人意料。当 **`--run_mode sim` 似乎被忽略**时，将你的模式与 **`examples/README.md`** 及初学者示例中的 **`_peek_run_mode_from_argv`** 惯用写法对比。
- **AIV / 隐式 copy 之前的 vector tile：** 在可能触发 AIV / **`REGISTER_COPY`** 的 op 之前，按你的 PyPTO 版本的 `docs/api/config/pypto-set_vec_tile_shapes.md` 要求，使用正整数调用 **`pypto.set_vec_tile_shapes`**（参见 **§4** / `docs/trouble_shooting/function.md` 中的 **`F21004`**）。当扩展 shape 或遇到晦涩的 vec-tile 错误时，查阅你版本的文档和 **`skills/lead-orchestrator/references/rules.md`** 规则 16。
- **`pypto.loop` 与 host `for`：** 图迭代属于 **`pypto.loop`**；不要在 traced kernel 内部用普通 Python **`for range`** 驱动 tile/batch 工作以违反 layout 规则（参见 **`skills/lead-orchestrator/references/rules.md`** 禁令 B）。官方 transform/loop 示例嵌套 **`pypto.loop`** 并使用 **`view` / `assemble`**。
- **`out.move(...)` 与切片赋值：** 许多 kernel 使用 **`out.move(pypto.op(...))`** 而非 **`out[:] = ...`**。两种写法在仓库中都有出现；调试 shape/dtype 不匹配时，检查参考实现使用哪种模式并保持一致。
- **Cube 与 vec：** matmul 类 cube op 使用 **`set_cube_tile_shapes`**；当 vector op 或编译器插入的 copy 参与时，仍需设置 vec tile。
- **错误码路由：** 将 **`Errcode: F2…`** → `docs/trouble_shooting/function.md`，**`F4/F5`** → `pass.md`，**`F9`** → `simulation.md` 等（`docs/trouble_shooting/README.md` 表格）。
- **多输出和大型脚本：** 在失败文件上使用 **`extract_pypto_calls.py`**（路径参见 **§3** 步骤 2）一目了然地查看 op 顺序。
- **环境 / 退出码：** 如果 **`import pypto`** 失败，按 **`docs/install/build_and_install.md`** 修复安装，而不是认为 kernel 逻辑有问题。捕获脚本是否失败时，记住 shell 管道可能会屏蔽 Python 的退出码，除非使用 **`set -e`**、**`set -o pipefail`** 或不屏蔽的 **`python3 ...; echo $?`**。
- **Golden 参考实现模式：** Golden 函数（用于精度验证）可以采用**两种等效策略**：
  - **全量计算：** 一次处理整个输入 tensor（默认，更简单）。
  - **分块计算：** 将输入拆分为小 tile，独立计算每个 tile，然后拼接结果。分块 golden 更接近 PyPTO kernel 的实际执行方式（逐 tile），更适合验证边界处理、累加逻辑和 tile 大小对数值精度的影响。参见 **`skills/pypto-golden-generate/SKILL.md`** → **§4 実装策略：全量 vs 分块** 了解模式和适用场景。

### 8.2 示例清单和常见失败

- **范围（典型代码树）：** **`examples/`** 下有 **23** 个可运行的示例 kernel **`*.py`** 文件（不包括 **`validate_examples.py`** / harness 脚本）。可选的 scratch 镜像位于 **`custom/debug_scratch_examples/<sanitized_path>/`**。
- **PyPTO 可用时的主要常见失败类别：**（1）**无效/缺失 vec tile** → **`F21004` / REGISTER_COPY**；（2）**`run_mode` / 装饰器绑定** 与 **`--run_mode sim`** 的冲突；（3）matmul 上的 **Cube 与 vec** tiling 混淆；（4）大型融合图上的 **Pass / stitch**（**`F4` / `F5`**）；（5）**View/assemble + 动态** shape 不匹配。

### 8.3 集成路径（示例与 ACL / cost model）

- **`examples/03_advanced/aclgraph/aclgraph.py`：** 使用 **`@pypto.frontend.jit()`** 配合 **Torch Dynamo** **`@allow_in_graph`** 和 **`FakeTensor`** 守卫——与仅传递 **`runtime_options={"run_mode": ...}`** 的脚本不同的集成路径。参见 **`docs/tutorials/network_integration/pytorch_integration.md`** 了解与图捕获兼容的返回/赋值模式。
- **`examples/03_advanced/cost_model/cost_model.py`:** 使用 **`runtime_options`** 如 **`stitch_cfgcache_size`** 和 **`run_mode: pypto.RunMode.SIM`**；泳道图 / JSON 制品可能出现在 **`./output`** 下。如果"cost model produced nothing"，在归咎 softmax 数学之前验证 SIM 选项和输出路径。

---

## 最后提醒

**宁可记录十次带日志引用的失败尝试，也不要过早退出。** 未知错误码意味着**升级证据和方法**，而不是**停止**。

---

## 9. PyPTO Kernel 开发通用调试指南

*Agent 从 GDR kernel 开发中学到的模式。发现新模式时添加到此节。*

### 9.1 JIT 签名解析

#### 问题：`from __future__ import annotations` 破坏 JIT

**症状：** `RuntimeError: Non-tensor parameter 'q_in' must not be a torch.Tensor. Use positional arguments for tensors.`

**根因：** PEP 563 字符串注解导致所有类型提示以字符串而非对象形式存储。

**诊断：**
```python
# 检查注解 - 它们应该是 pypto.Tensor 对象，不是字符串
func = kernel._original_func
print(func.__annotations__)  # 如果是字符串，就是 import 问题
```

**解决方案：** 从使用 `@pypto.frontend.jit` 的文件中移除 `from __future__ import annotations`。

```python
# 错误 - 导致 JIT 失败
from __future__ import annotations
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass

# 正确
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass
```

---

### 9.2 动态 Shape

#### 问题：`set_vec_tile_shapes` 需要具体值

**症状：** 使用符号维度时出现 `ValueError: Not concrete value`。

**解决方案：** 使用模块级常量或函数参数作为 tile shape：

```python
# 错误
@pypto.frontend.jit()
def kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)):
    B, T = x.shape
    pypto.set_vec_tile_shapes(B, T, 32, 32)  # 失败 - B, T 是符号值

# 正确 - 使用具体常量
TILE_M, TILE_N = 32, 32
@pypto.frontend.jit()
def kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(TILE_M, TILE_N, 32, 32)  # 可以工作
```

---

#### 问题：`pypto.loop` 使用符号边界

**症状：**
```
ValueError: Invalid value type
Errcode: F21004!
op [MUL]tile shape not set
```

**根因：** `pypto.loop` 需要**具体整数**的 start/stop/step 值。使用来自 tensor shape 的符号表达式如 `B * H` 会失败。

```python
# 错误 - B 和 H 是来自 tensor shape 的符号值
for session in pypto.loop(range(B * H), name="sessions"):
for i in pypto.loop(range(nt), name="chunks"):  # nt = T // bt 是符号值
```

**解决方案：** 将循环边界作为**具体整数参数**传递：

```python
# Kernel 签名：将 B, H, nt 作为具体参数传递
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(
    q_in: pypto.Tensor([], pypto.DT_FP32),
    ...
    B: int, H: int, nt: int,  # 具体的循环边界
):
    for session in pypto.loop(0, B * H, 1, name="sessions"):
        for c in pypto.loop(0, nt, 1, name="chunks"):
            ...

# 调用方：传递具体值
kernel(q, k, ..., B, H, nt)
```

**注意：** 在循环体内部，`b = session // H` 和 `h = session % H` 仍然是符号值，但可以在 `pypto.view` 的偏移量中使用。

---

#### 问题：使用符号索引进行 Tensor 索引

**症状：** 使用 `tensor[symbolic_index]` 时出现 `TypeError: Cannot convert symbols to int`。

**解决方案：** 使用带符号偏移量的 `pypto.view` 代替直接索引：

```python
# 错误
result = tensor[idx, :, :]  # idx 是符号值 - 失败

# 正确 - 使用带偏移量的 view
view = pypto.view(tensor, [1, T, K], [idx, 0, 0])
result = pypto.matmul(...)  # 直接在 view 上操作
```

---

#### 问题：pypto.view shape/offsets 维度不匹配

**症状：**
```
RuntimeError: Errcode: F21004!
Their size actually are 4 and 2, func GetViewValidShape
```

**根因：** `shape` 和 `offsets` 必须具有**相同数量的元素**。

```python
# 错误 - shape 有 2 维，offsets 有 4 个元素
pypto.view(tensor, [K, V], [b, h, 0, 0])

# 正确 - 使用匹配的维度，然后 reshape
pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

**规则：** 始终满足 `len(shape) == len(offsets)`。

**常见模式：**
```python
# Tensor [B, T, H, K], view [bt, K] at [b, t0, h, 0]
view = pypto.view(tensor, [1, bt, 1, K], [b, t0, h, 0]).reshape([bt, K])

# Tensor [B, H, K, V], view [K, V] at [b, h, 0, 0]
view = pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])

# Tensor [B, T, H], view [bt] at [b, t0, h]
view = pypto.view(tensor, [1, bt, 1], [b, t0, h]).reshape([bt])

# Tensor [B, H, nt, bt, bt], view [bt, bt] at [b, h, c, 0, 0]
view = pypto.view(tensor, [1, 1, 1, bt, bt], [b, h, c, 0, 0]).reshape([bt, bt])
```

---

### 9.4 pypto.view 完整指南

#### 签名
```python
pypto.view(
    input: pypto.Tensor,
    shape: List[int],           # 必须是具体整数
    offsets: List[Union[int, pypto.SymbolicScalar]],
    valid_shape: Optional[List[Union[int, pypto.SymbolicScalar]]] = None
) -> pypto.Tensor
```

#### 黄金规则
**`len(shape) == len(offsets)`** —— 这是强制要求！

#### 最佳实践

**1. 始终匹配维度：**
```python
# Tensor shape: [B, T, H, K] (4D)
# Offsets: [b, t0, h, 0] (4 elements)
# View shape 必须是 4D: [1, bt, 1, K]
qc = pypto.view(q_norm, [1, bt, 1, K], [b, t0, h, 0]).reshape([bt, K])
```

**2. 用 1 填充未使用的维度：**
```python
# Tensor [B, H, K, V] → view at [b, h]
# 使用 [1, 1, K, V] 匹配 4 元素 offsets [b, h, 0, 0]
s = pypto.view(state, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

**3. 对于多维 offsets 的 1D/2D tensor：**
```python
# Tensor [B, T, H] with 3D offsets [b, t0, h]
# 使用 [1, bt, 1] 匹配 3 个元素
betac = pypto.view(beta_in, [1, bt, 1], [b, t0, h]).reshape([bt])
```

**4. 对于 5D tensor：**
```python
# Tensor [B, H, nt, bt, bt] with 5D offsets [b, h, c, 0, 0]
# 使用 [1, 1, 1, bt, bt] 匹配 5 个元素
A_c = pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0]).reshape([bt, bt])
```

**5. 用 reshape 组装回去：**
```python
# 组装时，reshape 以匹配原始 tensor 的 view 维度
pypto.assemble(s.reshape([1, 1, K, V]), [b, h, 0, 0], output)
```

#### 常见错误

| 错误 | 报错 | 修复 |
|------|------|------|
| Shape 维度 ≠ offset 维度 | `Their size actually are X and Y` | 用 1 填充 |
| 用 4 个 offsets 使用 `[bt, K]` | 维度不匹配 | 使用 `[1, bt, 1, K]` |
| view 后忘记 reshape | 计算中 shape 错误 | 添加 `.reshape([bt, K])` |

#### 维度填充模式
当期望的 view 维度少于 offsets 时：
```
原始:  [K, V]  期望
Offsets:   [b, h, 0, 0]  有 4 个元素
解决方案:  填充 shape: [1, 1, K, V]
结果:    pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

---

### 9.6 SIM 模式限制

#### 问题：SIM 模式产生垃圾值

**症状：** SIM 模式下极端错误（10^20+），但 kernel 在 NPU 上可能正常。

**根因：** SIM 模式在以下方面有根本性限制：
- 动态 shape 处理
- Cube tile 配置
- 内存操作

**解决方案：**
1. 对照 golden 参考验证数学正确性
2. 在实际 NPU 硬件上测试
3. 不要依赖 SIM 模式进行精度验证

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    ...
```

**警告：** SIM 模式仅用于基本执行测试，不用于精度验证。

---

#### 问题：SIM 模式下 `operand1 dim[0] = -1`

**症状：** 在 SIM 模式下使用动态 shape 时维度变为 -1。

**解决方案：** SIM 模式测试使用静态 shape，或接受 SIM 对动态 shape 的限制。

---

#### 问题：`Invalid tile values: kL0=0, kL1a=0...`

**症状：** SIM 模式下 cube tiling 验证失败。

**解决方案：** 这是 SIM 模式的限制。kernel 在实际 NPU 硬件上可能正常。

---

### 9.7 Tensor 操作

#### pypto.matmul 需要 2D+ tensor

**错误：**
```
RuntimeError: Tensor dimension mismatch. Expect input_dim == mat2_dim and both in [2, 3, 4], got input_dim: 2, mat2_dim: 1.
```

**原因：** `pypto.matmul` 要求两个输入 tensor 都具有 2+ 维度。1D tensor 必须 reshape 为 2D。

**常见情况 —— 向量-矩阵乘法：**
```python
# c_cum 是 [bt, bt] (2D), gc_raw 是 [bt] (1D)
# 错误
g_cum = pypto.matmul(c_cum, gc_raw, ...)

# 正确 - reshape 为 2D
gc_raw_2d = gc_raw.reshape([bt, 1])
g_cum = pypto.matmul(c_cum, gc_raw_2d, ...).reshape([bt])
```

**matmul 产生 1D 结果的模式：**
```python
# 当结果应该是 [bt] 但 matmul 给出 [bt, 1]
result = pypto.matmul(matrix, vector_2d, ...).reshape([bt])
```

#### 广播
使用 `pypto.reshape` 添加/移除维度进行广播：

```python
# 错误
result = tensor * scalar  # scalar 需要显式 reshape

# 正确
scalar_reshaped = pypto.reshape(scalar, [bt, 1])
result = pypto.mul(tensor, scalar_reshaped)
```

#### 转置
```python
transposed = pypto.transpose(tensor, dim0, dim1)
```

#### 零初始化
```python
zeros = pypto.zeros([M, N], pypto.DT_FP32)
```

#### 逐元素操作
```python
negated = pypto.mul(tensor, -1.0)  # 乘以负一
```

---

### 9.8 常见模式

#### 模式：带状态传递的多 session
```python
for session in pypto.loop(range(B * H), name="sessions"):
    b = session // H
    h = session % H

    state = pypto.view(initial_state, [K, V], [b, h, 0, 0])

    for c in pypto.loop(range(nt), name="chunks"):
        # 处理 chunk
        ...
        state = updated_state

    output[b, h, :, :] = state
```

#### 模式：模块级常量 tile shape
```python
# 模块级 tile shape 常量
TILE_SESSIONS = 16
TILE_CHUNKS = 4
TILE_BT = 16
TILE_KV = 64

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    pypto.set_vec_tile_shapes(TILE_SESSIONS, TILE_CHUNKS, TILE_BT, TILE_KV)
    ...
```

#### 模式：在 host 上预计算
对于 PyPTO 不支持的操作（如 `torch.linalg.solve_triangular`）：
```python
def make_host_constants(bt, k, g_raw, beta, device):
    # 在 CPU/GPU 上预计算
    A = torch.linalg.solve_triangular(...)
    return A

# 在 kernel 中，作为常量接收
@pypto.frontend.jit()
def kernel(A_in: pypto.Tensor(...)):
    A_c = pypto.view(A_in, [bt, bt], [b, h, c, 0, 0])
    ...
```

#### 模式：反向迭代
```python
for i in pypto.loop(range(nt), name="chunks_reverse"):
    c = nt - 1 - i  # 反向 chunk 索引
    ...
```

---

### 9.9 调试检查清单

| 检查项 | 命令/方法 |
|--------|----------|
| JIT 签名 | 检查 `_original_func.__annotations__` 不是字符串 |
| 动态 shape | 验证 tile shape 是具体值 |
| SIM 模式 | 接受限制，在 NPU 上测试 |
| 导入 | 确保 `pypto` 正确导入 |
| 类型提示 | 不要有 `from __future__ import annotations` |

---

### 9.10 测试策略

1. **语法检查：** `python -m py_compile module.py`
2. **Golden 对比：** 与 PyTorch 参考实现对比
3. **SIM 模式：** 用于基本执行（非精度）
4. **NPU 模式：** 用于实际精度验证

---

### 9.11 常见错误信息

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Non-tensor parameter 'x' must not be a torch.Tensor` | 字符串注解 | 移除 `from __future__ import annotations` |
| `Not concrete value` | tile shape 中有符号值 | 使用具体常量 |
| `Cannot convert symbols to int` | 循环/索引中有符号值 | 使用带偏移量的 `pypto.view` |
| `ValueError: Invalid value type` | `pypto.loop` 中有符号值 | 将循环边界作为具体 int 参数传递 |
| `Errcode: F21004 tile shape not set` | op 前缺少 `set_vec_tile_shapes` | 先调用 `set_vec_tile_shapes` |
| `Errcode: F21004 Their size actually are X and Y` | `pypto.view` shape/offsets 不匹配 | 确保 `len(shape) == len(offsets)` |
| `operand1 dim[0] = -1` | SIM 模式动态 shape 问题 | 在 NPU 上测试 |
| `Invalid tile values` | SIM 模式 tiling 问题 | 在 NPU 上测试 |

---

### 9.12 PyPTO API 要点

- **`pypto.DYNAMIC`** —— tensor 类型提示中的动态维度标记
- **`pypto.DT_FP32`、`pypto.DT_BF16` 等** —— 数据类型枚举
- **`pypto.RunMode.NPU`** —— 在实际 NPU 硬件上运行
- **`pypto.RunMode.SIM`** —— 在模拟模式下运行（有限制）
- **`pypto.loop(start, stop, step)`** —— Kernel 循环构造
- **`pypto.view(tensor, shape, offsets)`** —— Tensor 视图/切片
- **`pypto.matmul(a, b, out_dtype=...)`** —— 矩阵乘法
- **`pypto.exp`、`pypto.mul`、`pypto.add` 等** —— 逐元素操作
- **`pypto.transpose(t, dim0, dim1)`** —— 转置
- **`pypto.reshape(t, shape)`** —— 用于广播的 reshape
- **`pypto.zeros(shape, dtype)`** —— 创建零 tensor
- **`pypto.set_vec_tile_shapes(...)`** —— 设置 vector tile 配置
- **`pypto.set_cube_tile_shapes(...)`** —— 设置 cube tile 配置

---

### 9.13 Tensor Shape 规格

#### 问题：Shape 大小超过 INT32_MAX

**错误：**
```
RuntimeError: Errcode: FFFFFF!
The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)
```

**根因：** 在 tensor 注解中使用 `pypto.DYNAMIC` 的显式 shape 规格：
```python
# 错误 - 导致 shape 大小错误
x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
```

**解决方案：** 使用空括号 `[]` 进行 shape 推断：
```python
# 正确 - shape 从实际 tensor 推断
x: pypto.Tensor([], pypto.DT_FP32)
```

**参考：** 所有正常工作的 PyPTO 示例都使用 `pypto.Tensor([], dtype)` 模式。

---

### 9.14 PyPTO JIT 内部的 Python 运算符

**发现：** Python 运算符**可以**在 `@pypto.frontend.jit` 装饰的函数内工作！

**来自可运行示例的证据：**
```python
# 来自 pypto_l2norm_bwd
dot_q = (dyq * yq).sum(-1, keepdim=True)
d = dyq * rstd_yq - dot_q * yq * rstd_yq
```

**建议：** 使用 Python 运算符编写更简洁的代码：
```python
# 冗长（不必要）
result = pypto.mul(pypto.mul(a, b), pypto.add(c, d))
result = pypto.sum(x, dim=-1, keepdim=True)
result = pypto.rsqrt(pypto.add(sum_sq, eps))

# 推荐（简洁且有效！）
result = a * b * (c + d)
result = x.sum(-1, keepdim=True)
result = (sum_sq + eps).rsqrt()
```

**支持的方法链：**
```python
tensor.T              # 转置
tensor.exp()          # 逐元素 exp
tensor.reshape([...]) # reshape
tensor.sum(-1)        # 沿最后一维求和
tensor.rsqrt()       # 平方根倒数
tensor.abs()         # 绝对值
tensor.sqrt()        # 平方根
tensor.neg()         # 取反
```

---

### 9.15 Tile Shape 配置

**问题：** 固定 tile shape 可能与实际 tensor 维度不匹配。

**对于没有循环的简单 kernel：**
```python
B, T, H, K = x.shape
pypto.set_vec_tile_shapes(B, H, T, K)
```

**对于有循环的复杂 kernel：**
```python
TILE_0 = 16   # 第一个 tile 维度
TILE_1 = 4    # 第二个 tile 维度
TILE_2 = 8    # 第三个 tile 维度
TILE_3 = 32   # 第四个 tile 维度
pypto.set_vec_tile_shapes(TILE_0, TILE_1, TILE_2, TILE_3)
```

**规则：** Tile shape 值应能整除 tensor 维度以获得最佳性能。

---

### 9.16 转置操作

两种方式都可以：
```python
# 方式 1：.T 属性（推荐 - 更简洁）
transposed = tensor.T

# 方式 2：显式函数
transposed = pypto.transpose(tensor, 0, 1)
```

---

### 9.17 模式速查表

| 操作 | 冗长形式 | 推荐形式 |
|------|---------|---------|
| Tensor shape | `pypto.Tensor([pypto.DYNAMIC, ...], dtype)` | `pypto.Tensor([], dtype)` |
| 乘法 | `pypto.mul(x, y)` | `x * y` |
| 平方 | `pypto.mul(x, x)` | `x * x` |
| 求和 | `pypto.sum(x, dim=-1, keepdim=True)` | `x.sum(-1, keepdim=True)` |
| Rsqrt | `pypto.rsqrt(x)` | `x.rsqrt()` |
| Exp | `pypto.exp(x)` | `x.exp()` |
| 加标量 | `pypto.add(x, scalar)` | `x + scalar` |
| 减标量 | `pypto.sub(x, scalar)` | `x - scalar` |
| 转置 | `pypto.transpose(t, 0, 1)` | `t.T` |
| 类型转换 | `pypto.cast(x, pypto.DT_FP32)` | `x.float()` |
| Reshape | `pypto.reshape(t, [a, b])` | `t.reshape([a, b])` |
| Matmul | `pypto.matmul(a, b, ...)` | `pypto.matmul(a, b, ...)`（保持显式） |

---

### 9.18 关键要点

1. **Shape 推断：** 使用 `pypto.Tensor([], dtype)` 进行自动 shape 推断
2. **Python 运算符：** 在 JIT 内使用 Python 运算符（`*`、`+`、`-`、`/`）
3. **方法链：** PyPTO tensor 支持方法链（`.exp()`、`.T`、`.rsqrt()`）
4. **Tile Shape：** 将 tile shape 匹配到实际 tensor 维度或使用偶数除数
5. **保持 matmul 显式：** `pypto.matmul()` 优于 Python `@` 运算符

---

### 9.19 matmul API 和 Tile Shape

#### 问题：错误的 matmul 语法导致错误

**症状：** matmul 操作以晦涩的错误失败。

**根因：** `pypto.matmul` 的 API 使用错误。

**正确的 matmul 语法：**
```python
# 错误 - 这个语法不工作
result = pypto.matmul(a, b.T, out_dtype=pypto.DT_FP32)
result = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)

# 正确 - 使用 a_trans 和 b_trans 参数
result = pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=True)
result = pypto.matmul(a, b, pypto.DT_FP32)  # 默认都为 False
```

**转置模式：**
```python
# a @ b.T  →  a_trans=False, b_trans=True
result = pypto.matmul(a, b, dtype, a_trans=False, b_trans=True)

# a.T @ b  →  a_trans=True, b_trans=False
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=False)

# a @ b    →  都为 False（默认）
result = pypto.matmul(a, b, dtype)

# a.T @ b.T  →  都为 True
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)
```

**注意：** 不要在传入 matmul 之前对 tensor 使用 `.T`——使用转置标志代替。

#### 问题：需要同时设置 vec 和 cube tile shape

**症状：** matmul 或其他 op 以 tiling 错误失败。

**根因：** 需要同时设置 `set_vec_tile_shapes` 和 `set_cube_tile_shapes`。

**解决方案：**
```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    # matmul 工作需要两者都设置
    pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    # 现在 matmul 操作可以工作
    result = pypto.matmul(a, b, pypto.DT_FP32)
    ...
```

**常见 tile 配置：**
```python
# 正向 kernel
TILE_B = 1
TILE_H = 2
TILE_T = 8
TILE_K = 32
pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 反向 kernel
TILE_BH = 16
TILE_NT = 4
TILE_BT = 8
TILE_KV = 32
pypto.set_vec_tile_shapes(TILE_BH, TILE_NT, TILE_BT, TILE_KV)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

**规则：** 使用 matmul 或复杂 tensor 操作时，始终同时设置两种 tile shape 配置。

#### 问题：pypto.assemble shape 不匹配

**症状：**
```
CHECK FAILED: dest.GetShape().size() == tensor.GetShape().size()
Assemble: src and dest requires same shape
```

**根因：** src tensor shape 与目标位置的期望 view 维度不匹配。

**解决方案：** 将输出 tensor reshape 以匹配期望的 view shape：
```python
# 错误 - out_chunk 是 [bt, V] 但 assemble 期望 [1, bt, 1, V]
pypto.assemble(out_chunk, [b, t0, h, 0], out_out)

# 正确 - reshape 以匹配 view 维度
pypto.assemble(out_chunk.reshape([1, bt, 1, V]), [b, t0, h, 0], out_out)
```

**规则：** `pypto.assemble(tensor, offsets, dest)` 要求 tensor shape 与这些偏移量处的 view 具有相同的维度数。

#### 问题：对多元素 tensor 执行 reshape([1])

**症状：**
```
CHECK FAILED: capacity == 1
Shape size not match, func CheckAndInferShape
```

**根因：** 尝试将具有多个元素的 tensor reshape 为单个元素的 shape。

**解决方案：** 使用 `pypto.view` 配合偏移量提取单个元素，然后 reshape：
```python
# 错误 - bt=4 的 tensor 容量为 4，不能 reshape 为 [1]
gl = g_cum.reshape([1])[bt - 1:bt].reshape([1])

# 正确 - 使用 view 获取最后一个元素，然后 reshape
gl = pypto.view(g_cum, [1], [bt - 1]).reshape([1])
```

**获取最后一个元素的常见模式：**
```python
# 获取大小为 bt 的 1D tensor 的最后一个元素
last_elem = pypto.view(tensor, [1], [bt - 1]).reshape([1])
```

#### 问题：PyPTO tensor 不支持 `.T` 属性

**症状：**
```
AttributeError: 'Tensor' object has no attribute 'T'
```

**根因：** PyPTO tensor 不像 PyTorch tensor 那样支持 `.T` 属性。

**解决方案：** 使用 `pypto.transpose(tensor, 0, 1)` 或使用 matmul 的转置标志：
```python
# 错误 - .T 在 PyPTO tensor 上不工作
kc_t = kc.T
result = pypto.matmul(qc, kc_t, dtype)

# 正确 - 使用 matmul 转置标志
result = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)

# 或使用显式转置
kc_t = pypto.transpose(kc, 0, 1)
result = pypto.matmul(qc, kc_t, dtype)
```

**对于 2D 矩阵转置：** `pypto.transpose(tensor, 0, 1)` 在 tiling 系统下的 2D tensor 上**不工作**——会导致"TileShape dim num should same to input"错误。

**替代方案，使用 matmul 转置标志：**
```python
# 不要用转置，而是在 matmul 中使用 a_trans/b_trans：
kc_t = pypto.transpose(kc, 0, 1)  # 错误 - 不工作！
qk = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)  # 正确

# 对于对称求和 m_mat + m_mat.T，拆分为两个 matmul：
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype)  # m_mat @ kc
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype, a_trans=True, b_trans=False)  # m_mat.T @ kc
```

#### 问题：归约轴需要 32 字节对齐

**症状：**
```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**根因：** PyPTO 归约操作（`.sum()`）要求 tensor 维度是 32 字节对齐的。

**计算：** 对于 FP32（4 字节），维度 × 4 必须能被 32 整除。
- `bt=4` → 4×4=16 字节 ❌ 未对齐
- `bt=8` → 8×4=32 字节 ✅ 已对齐
- `V=16` → 16×4=64 字节 ✅ 已对齐

**解决方案：** 归约轴使用 8 的倍数作为维度：
```python
# 错误 - bt=4 导致对齐错误
bt = 4
result = tensor.sum(-1)

# 正确 - bt=8 是 32 字节对齐
bt = 8
result = tensor.sum(-1)
```

**规则：** 参与 `.sum()`、`.mean()` 或其他归约操作的任何 tensor 维度必须满足 `(dim × bytes_per_element) % 32 == 0`。对于 FP32，使用 8 的倍数的维度。

**常见对齐示例（FP32）：**
- V=16 → 16×4=64 字节 ✅ 已对齐
- V=32 → 32×4=128 字节 ✅ 已对齐
- K=16 → 16×4=64 字节 ✅ 已对齐
- K=32 → 32×4=128 字节 ✅ 已对齐
- bt=8 → 8×4=32 字节 ✅ 已对齐

#### 问题：维度已对齐时 sum 归约仍然失败

**症状：**
```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**问题：** 即使维度在理论上已对齐（如 V=32），在最后一轴上 `.sum(-1)` 仍可能因 PyPTO 内部 tile tensor 的方式而失败。

**解决方案：** 使用基于 matmul 的归约替代 `.sum()`，通过预计算的全一向量：

```python
# 在 host 上创建全一向量（适用于任何维度，不仅限于对齐的）
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)  # [V, 1]
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1]

# 在 kernel 签名中，将全一向量作为参数添加：
@pypto.frontend.jit(...)
def kernel(..., ones_v: pypto.Tensor([], pypto.DT_FP32), ones_k: pypto.Tensor([], pypto.DT_FP32), ...):

    # 用 matmul 替代 .sum(-1)：
    db_c = (dvb * vc).sum(-1)  # 错误 - 可能失败

    db_c = pypto.matmul(dvb * vc, ones_v, pypto.DT_FP32).reshape([bt])  # 正确

    # 用 matmul 替代 .sum(0) 和 .sum(1)：
    d_l_l_mat_0 = (d_l * l_mat).sum(0)  # 错误
    d_l_l_mat_0 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32, a_trans=True, b_trans=False).reshape([bt])  # 正确

    d_l_l_mat_1 = (d_l * l_mat).sum(1)  # 错误
    d_l_l_mat_1 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32).reshape([bt])  # 正确
```

**关键洞察：** 基于 matmul 的归约无论对齐与否都能工作，因为它使用 cube 操作，而 `.sum()` 使用需要严格 32 字节对齐的 vector 操作。

**规则：** 有疑问时，使用 matmul 配合全一向量进行归约，而不是在任何轴上使用 `.sum()`。

#### 问题：matmul 中 K 维度 valid shape 不匹配

**症状：**
```
RuntimeError: K-dimension valid shape mismatch. Got input valid shape: [SymbolicScalar(8), SymbolicScalar(8)], mat2 valid shape: [SymbolicScalar(32), SymbolicScalar(1)], a_trans: False, b_trans: False.
```

**问题：** 使用了错误的全一向量进行 matmul 归约。全一向量必须与被归约的维度匹配。

**解决方案：** 对不同维度使用不同的全一向量：

```python
# 为不同维度创建不同的全一向量
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)   # [V, 1] - 用于归约 V 维度
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1] - 用于归约 K 维度
ones_bt = torch.ones(bt, 1, device=device, dtype=torch.float32) # [bt, 1] - 用于归约 bt 维度

# 在 kernel 签名中，添加所有全一向量：
def kernel(..., ones_v, ones_k, ones_bt, ...):

    # 对于 [bt, V] tensor 归约 V 维（沿最后一轴求和）：
    db_c = pypto.matmul(tensor, ones_v, dtype).reshape([bt])  # 正确

    # 对于 [bt, bt] tensor 归约 bt 维（沿行/列求和）：
    d_l_l_mat_0 = pypto.matmul(tensor, ones_bt, dtype, a_trans=True, b_trans=False).reshape([bt])  # 行求和
    d_l_l_mat_1 = pypto.matmul(tensor, ones_bt, dtype).reshape([bt])  # 列求和
```

**规则：** 全一向量的第一维度必须等于 matmul 操作中被归约的维度。

#### 问题：5D tensor view 配合 4D vec tile shape 导致 "Run pass failed"

**症状：**
```
Errcode: FFFFFF!
Run pass failed., func CompileFunction
```

**根因：** 当 `pypto.set_vec_tile_shapes` 仅设置 4D tile shape 时，使用 5D tensor view（如 `pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0])`）。框架无法用 4D tile 配置处理 5D 操作。

**解决方案：** 在传入 kernel 前将 5D tensor reshape 为 2D，然后使用匹配的 view shape/offsets：

```python
# 在 pypto_function 中，在 kernel 调用前 reshape：
# 原始: [B, H, nt, bt, bt] -> Reshape 为 2D: [B*H*nt, bt*bt]
A_2d = A_5d.reshape([B * H * nt, bt * bt])
w_2d = w_4d.reshape([B * H * nt, bt * K])
S_before_2d = S_before_4d.reshape([B * H * nt, K * V])
v_new_2d = v_new_4d.reshape([B * H * nt, bt * V])
g_cum_2d = g_cum_3d.reshape([B * H * nt, bt])

# 在 kernel 中，使用 2D view 配合 2 个 offsets（len(shape) == len(offsets)）：
session_base = session * nt
for c in pypto.loop(0, nt, 1):
    cache_idx = session_base + c
    # 对于 2D tensor [N, M] 的 view [a, b]: offsets = [cache_idx, 0]
    a = pypto.view(A_2d, [bt, bt], [cache_idx, 0]).reshape([bt, bt])
    w = pypto.view(w_2d, [bt, K], [cache_idx, 0]).reshape([bt, K])
    s_before = pypto.view(S_before_2d, [K, V], [cache_idx, 0]).reshape([K, V])
    v_new = pypto.view(v_new_2d, [bt, V], [cache_idx, 0]).reshape([bt, V])
    # 对于从 2D tensor 获取 1D 结果: shape = [1, size], offsets = [cache_idx, 0]
    g_cum = pypto.view(g_cum_2d, [1, bt], [cache_idx, 0]).reshape([bt])
```

**规则：** `len(shape) == len(offsets)` 对 pypto.view 是强制要求。保持所有 tensor ≤4D，在传入 kernel 前 reshape 为 2D。对于从 2D tensor 获取 1D view，使用 shape [1, size] 配合 2 个 offsets。
