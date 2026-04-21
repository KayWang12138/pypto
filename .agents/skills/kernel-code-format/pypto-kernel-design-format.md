# PyPTO kernel 源码布局与设计格式

本文档定义了 PyPTO 自定义 kernel 的**文件组织模式**和**文档规范**。适用于任何算子（逐元素、归约、attention 类递推、融合等）：按复杂度使用匹配的层级，其余省略。

目标：

- 人类读者和 LLM Agent 可以**按角色导航**（参考实现 vs PyPTO vs host 胶水 vs 测试）。
- **调试**保持可管理：每个编号阶段对应一个具有清晰契约的小函数。
- **可移植性**：可以替换数学、tiling 或融合方案而不丢失整体结构。

**规范 Python 骨架（复杂 kernel 工作流）：** `.agents/skills/kernel-code-format/pypto_kernel_template.py` —— Agent **必须**使用此文件作为 `custom/<op>/` 下**每个**分阶段模块文件和**完整** kernel 的起始布局（参见 **`skills/lead-orchestrator/references/rules.md`** 规则 **17** 和 **`skills/lead-orchestrator/references/rules.md`** 规则 **17**）。本文档（层级 A–L）与该模板保持一致。

---

## 1. 推荐的垂直层级（从上到下）

源文件中的章节大致按以下顺序排列。每个层级仅依赖其上方的层级。

| 层级 | 用途 | 典型命名（示例） |
| --- | --- | --- |
| **A. 工具函数** | 可复用的诊断工具（可选）。 | `tensor_compare_report`、日志辅助函数 |
| **B. 小型数学构建块** | 算法的纯 PyTorch（或 numpy）片段，被参考实现复用，有时在设备侧镜像。 | `norm_fwd`、`softmax_chunk`、… |
| **C. 前向参考实现** | PyTorch 中的基准前向计算，在**显式约束**下编写（见 §3）。 | `forward_ref` |
| **D. Host 侧常量** | 替代参考实现中禁用算子的矩阵和掩码（如通过 `matmul` 实现前缀和）。 | `make_chunk_constants` |
| **E. 反向参考实现（分解）** | 镜像一个循环体或一个流水线阶段的私有辅助函数；完整反向由它们拼接而成。 | `_slice_chunk_inputs`、`_stage_attn`、… |
| **F. Golden 反向** | 单一入口，用 PyTorch 实现完整反向用于数值验证。 | `torch_golden_*_backward_ref` |
| **G. 缓存 / 桥接** | 将 Python 列表、嵌套缓存或布局转换为 NPU 路径所需的扁平 Tensor。 | `prepare_cache_for_npu`、… |
| **H. PyPTO 子 kernel** | 小型、命名的 PyPTO 代码段：view、matmul、融合阶段。每个函数对应一个概念步骤。 | `pypto_slice_inputs`、`pypto_fused_stage_ab`、… |
| **I. Kernel 实现** | 实际的 `pypto.loop` 嵌套，调用 (H)；如果实现与入口分离，此处不使用 `@jit`。 | `_your_op_kernel_impl` |
| **J. JIT 入口** | `@pypto.frontend.jit` 函数：Tensor 签名、`runtime_options`、`debug_options`；委托给 (I)。 | `your_op_kernel_npu` |
| **K. Host wrapper** | 分配输出、打包 torch Tensor、调用 (J)、将结果 reshape 为用户布局。 | `pypto_function` |
| **L. 驱动 / 测试** | `main()` 或 pytest：配置、前向参考、golden、为 PyPTO reshape、比较。 | `main` |

并非每个 kernel 都需要所有层级。一个最小的单目算子可能跳过 (C)、(G) 和大部分 (E)–(H)；训练反向 kernel 通常需要 (C)–(L)。

---

## 2. 命名规范（适用于任何 kernel）

| 模式 | 含义 |
| --- | --- |
| `forward_ref` / `*_forward_ref` | 前向传播的 PyTorch 参考实现。 |
| `torch_golden_*` / `*_golden_*` | 反向或端到端数值检查的完整参考实现。 |
| 前导 `_` | 私有辅助函数：一个逻辑步骤，不作为公共 API。 |
| `pypto_*` | 使用 `pypto` API 的代码（view、matmul、loop、tile shape）。 |
| `*_kernel_impl` | 包含 `pypto.loop` 和子 kernel 调用的实现主体。 |
| `*_kernel_npu`（或 `*_jit`） | 带有类型化 Tensor 签名的 `@pypto.frontend.jit` 入口。 |
| `pypto_function`（或 `launch_*`、`run_*`） | Torch 侧启动器：布局转换 + `kernel_npu(...)` + reshape。 |

每个方向保持**一个主要的 "golden" 名称**（如一个反向 golden），以便测试和文档保持 grep 友好。

---

## 3. 参考实现约束（在代码中记录）

参考实现不是"任意 PyTorch"；它应遵循你在**头部注释中声明的规则**，例如：

- 允许：`matmul`、逐元素算子、沿最后维的 `sum`、对 batch/head/chunk 的显式循环。
- 禁止（如果它们使 NPU lowering 复杂化或与你的 PyPTO 路径不同）：`cumsum`、`masked_fill`、`tril`/`triu` 工厂算子、`flip`，或超出设备支持的高秩 Tensor。

当 PyPTO 侧使用该模式时，使用**显式矩阵**复现相同的数学（如 `C_cum @ x` 而非 `cumsum`）。这使得**差异调试**在参考步骤和 `pypto_*` 块之间保持一一对应。

---

## 4. 长函数内的阶段标记

对于较长的 `forward_ref` 或反向 golden 循环，使用**带标签的阶段**以便映射到 PyPTO 模块：

```text
# ===== (A) 阶段描述 =====
# ===== (B) 阶段描述 =====
# ===== (C) 阶段描述 =====
```

在 PyPTO 融合注释中使用**相同的字母或名称**（如 "融合模块 A + B"），以便不匹配时缩小到一对函数。

---

## 5. 契约表（为你的 kernel 填写）

维护一个简短的 **Tensor 契约**（在模块 docstring 或专用注释块中）：

| 名称 | Shape（符号） | dtype | 生产者 | 消费者 | 备注 |
| --- | --- | --- | --- | --- | --- |
| … | … | … | forward_ref / cache | backward ref / pypto | 如 "每个 chunk 的扁平化 BT×BT" |

对于**前向 → 反向**依赖关系，列出**缓存键**以及每个 Tensor 是否为反向所必需：

| 缓存键 | Shape | 反向是否需要？ |
| --- | --- | --- |
| … | … | yes / no |

---

## 6. 参考步骤到 PyPTO 的映射

使用**一对多表**（在文档中，不一定是代码中）：

| 参考辅助函数 / 阶段 | PyPTO 函数 | 备注 |
| --- | --- | --- |
| `_ref_stage_alpha` | `pypto_stage_alpha` | 相同数学，不同布局或 view。 |
| … | … | … |

当精度偏差时，比较**阶段输出**（保存的 Tensor）而非仅比较最终输出。

---

## 7. PyPTO 子 kernel 职责

每个 `pypto_*` 函数应：

1. **做一件事**（如"构建衰减矩阵"、"融合局部 attn + 递推"）。
2. 如需要则**本地设置 tile / pass 选项**（`set_vec_tile_shapes`、`set_pass_options`、`set_cube_tile_shapes`）并说明原因。
3. **返回**下一阶段所需的所有 Tensor（避免隐藏的全局变量）。

`*_kernel_impl` 因此读起来像一个**高层配方**：切片 → 阶段1 → 阶段2 → 写输出。

---

## 8. JIT 入口与实现分离

- **`_your_op_kernel_impl`**：所有动态索引、`pypto.view`、循环和对 `pypto_*` 辅助函数的调用。
- **`your_op_kernel_npu`**：`@pypto.frontend.jit`，静态签名（`pypto.DYNAMIC` / `pypto.STATIC`）、`runtime_options`、`debug_options`；主体是对 impl 的薄调用。

这种分离使**替换选项**或从测试中**复用 impl** 更容易，无需重新编译不同的 JIT shell。

---

## 9. Host wrapper（`pypto_function`）

职责：

1. 将 Tensor 移动/缓存到 JIT 入口期望的**设备和布局**（flatten、transpose、`expand`、dtype）。
2. **分配**输出缓冲区。
3. 调用 `*_kernel_npu(*inputs, *outputs)`。
4. 将输出 **reshape** 回面向用户的布局（如 `[B, T, H, D]`）。

尽可能将 I/O reshape **排除**在 JIT 函数之外。

---

## 10. 测试驱动（`main` 或 pytest）

建议结构：

1. **配置**：shape、dtype、chunk 大小 `BT`、种子、设备 ID、`run_mode`。
2. **输入**：带有 `requires_grad` 的随机 Tensor（如果测试 autograd 相关路径）。
3. **常量**：`make_chunk_constants` 或等效函数。
4. **前向参考**：运行 `forward_ref`，获取输出 + 缓存。
5. **上游梯度**：随机的 `do`、`dht` 等。
6. **Golden 反向**：`torch.no_grad()` + golden 函数。
7. **PyPTO 路径**：将缓存 Tensor 适配为 PyPTO 布局；调用 `pypto_function`。
8. **比较**：为每个输出使用详细的逐 Tensor 报告辅助函数（或逐 Tensor `torch.allclose`）。

---

## 11. Shape 注释规范

每个 Tensor 赋值和 tiling 配置行**必须**携带行内 shape 注释。

**1. 每个 Tensor 赋值都有 shape 注释**

```python
q = pypto.view(q_in, [B, N, Sq, D])          # [B, N, Sq, D]
k = pypto.view(k_in, [B, N, Skv, D])         # [B, N, Skv, D]
scores = pypto.matmul(q, k, pypto.DT_BF16)   # [B, N, Sq, Skv]
```

**2. Matmul 使用收缩形式**

```python
# [M, K] @ [K, N] -> [M, N]
out = pypto.matmul(a, b, pypto.DT_BF16)      # [M, N]
```

**3. Tile 配置行显示 tile shape**

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)                         # 各维度按文档配置
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256]) # 每个列表为 [L0, L1]；见下文
```

> **`set_cube_tile_shapes` 参数规则** —— `m`、`k`、`n` 各为 **2 元素列表 `[L0, L1]`**，
> 而非单元素列表。约束条件（来自 `docs/api/config/pypto-set_cube_tile_shapes.md`）：
>
> - `0 < mL0 <= mL1` 且 `mL1 % mL0 == 0`（`k`、`n` 同理）。
> - `kL0, kL1, nL0, nL1`：32 字节对齐（**FP32 输入：改为 16 元素对齐**）。
> - L0A/L0B/L0C 和 L1 缓存预算必须满足；FP16/BF16/FP32 的通用安全基线为
>   `[128, 128], [64, 128], [128, 256]`，需按 shape 调优。
> - `enable_split_k=True` 仅在输入为 2D 时有效（非 3D/4D）。
>
> ❌ `pypto.set_cube_tile_shapes([16], [32], [64])` —— 错误（单元素列表）。
> ✅ `pypto.set_cube_tile_shapes([16, 32], [32, 64], [64, 128])` —— 2 元素列表。

**4. 循环体 Tensor 显示切片 shape，而非完整 shape**

```python
for i in pypto.loop(range(Sq // tile_s), idx_name="i"):
    q_tile = pypto.view(q, [tile_s, D], ...)  # [tile_s, D]  (来自 [Sq, D] 的切片)
    s_tile = pypto.matmul(q_tile, k_t, ...)   # [tile_s, Skv]
```

**5. 动态轴使用符号名称，而非 `?`**

```python
x = pypto.view(x_in, [B, S, H])              # [B, S, H]  S=动态
```

**6. 归约操作同时注释输入和输出 shape**

```python
row_max = pypto.amax(scores, dim=-1)          # [B, N, Sq, Skv] -> [B, N, Sq, 1]
```

**不需要注释的内容**：import 行、`print`/日志、普通 Python 标量（`tile_m = 16`）。

---

## 12. 新 kernel 检查清单（通用）

- [ ] 数学规格和**符号 shape**已记录。
- [ ] `forward_ref`（如适用）遵循**已记录的约束**。
- [ ] 反向 golden 匹配前向缓存契约。
- [ ] 每个非平凡的循环体块提取为 `_helper` 或 `pypto_*`，并附有**一行职责注释**。
- [ ] 参考阶段标签与 PyPTO 模块注释对齐。
- [ ] JIT 签名匹配实际缓冲区秩（动态 vs 静态维度）。
- [ ] Host wrapper 记录了**布局**假设（行优先 flatten 顺序、head 分组等）。
- [ ] 测试比较了与 API 相关的**所有**输出。

---

## 12. 最小模板（仅骨架）

```text
# --- 工具函数（可选） ---

# --- 小型 torch 辅助函数 ---

# --- forward_ref（约束在头部注释中） ---

# --- 常量 ---

# --- 反向参考辅助函数 (_*) ---

# --- torch_golden_* 反向 ---

# --- prepare_* 缓存桥接 ---

# --- pypto_* 子 kernel ---

def _my_kernel_impl(...):
    for ... in pypto.loop(...):
        ...
        # 调用 pypto_* 辅助函数

@pypto.frontend.jit(...)
def my_kernel_npu(... typed tensors ...):
    _my_kernel_impl(...)

def pypto_function(... torch tensors ...):
    # 分配、打包、调用 my_kernel_npu、reshape
    ...

def main():
    # 配置 → forward_ref → golden → pypto_function → 比较
```

按需调整深度：仅前向推理的 kernel 省略反向/golden 部分；融合逐元素 kernel 可以将"子 kernel"内联到单个 `pypto_*` 或 impl 中。
