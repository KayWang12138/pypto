# PyPTO 复杂算子 — Layer 组织规范（multi-file 仕様）

本文档定义 **复杂算子**（attention / recurrent / fused 类）的代码组织规范。
简单算子（SiLU / softmax / layer_norm 等）请遵循 `pypto-op-develop` 的标准 3 文件结构，
本文档不适用。

---

## 1. 文件组织（multi-file 仕様）

复杂算子最终交付为 **3 个独立文件 + 1 个 staged 目录**:

```
custom/<op>/
├── <op>_golden.py          ← Layers B–F: 纯 torch reference + 主 oracle
├── <op>_impl.py            ← Layers G–K: PyPTO 实现 + JIT entry + 主机包装
├── test_<op>.py            ← Layer L: 测试驱动（golden vs impl）
├── README.md
├── SPEC.md
├── DESIGN.md
├── MEMORY.md
└── staged/                 ← 开发期 staged set（完成后保留）
    ├── <op>_module1_impl.py
    ├── <op>_module1_golden.py
    ├── test_<op>_module1.py
    ├── <op>_module12_impl.py
    ├── <op>_module12_golden.py
    ├── test_<op>_module12.py
    └── ...
```

各文件职责:

| 文件 | 包含 layer | 职责 |
|------|-----------|------|
| `<op>_golden.py` | B, C, D, E, F | 纯 torch 数学参考 + `torch_golden_reference` 主 oracle |
| `<op>_impl.py` | G, H, I, J, K | PyPTO 子内核 + kernel impl + JIT entry + 主机包装器 |
| `test_<op>.py` | L | 测试驱动: golden vs impl 精度对比 |

---

## 2. Layers A–L 详细定义

| Layer | 角色 | 所在文件 | 典型函数名 |
|-------|------|---------|-----------|
| **A** | Utilities (可选) | `test_<op>.py` 顶部 import | `detailed_tensor_compare`（从 `pypto-op-validate` import）|
| **B** | 小数学 building blocks (pure PyTorch) | `<op>_golden.py` | `norm_fwd`, `softmax_chunk` |
| **C** | Forward reference (可选) | `<op>_golden.py` | `forward_ref` |
| **D** | Host-side constants (可选) | `<op>_golden.py` 或 `<op>_impl.py` | `make_chunk_constants` |
| **E** | Reference helpers (可选) | `<op>_golden.py` | `_slice_chunk_inputs`, `_stage_attn` |
| **F** | **PRIMARY golden** (必须) | `<op>_golden.py` | `<op>_golden` 或 `torch_golden_reference` |
| **G** | Cache / layout 桥接 (可选) | `<op>_impl.py` | `prepare_buffers_for_pypto` |
| **H** | PyPTO 子内核 (小命名区域) | `<op>_impl.py` | `pypto_slice_inputs`, `pypto_fused_stage_ab` |
| **I** | Kernel implementation (`pypto.loop` 嵌套) | `<op>_impl.py` | `_<op>_kernel_impl` |
| **J** | JIT entry (`@pypto.frontend.jit`) | `<op>_impl.py` | `<op>_kernel_npu` |
| **K** | Host wrapper (公共 API) | `<op>_impl.py` | `pypto_function` |
| **L** | Test driver | `test_<op>.py` | `test_<op>_levelN`, `main` |

并不是每个算子都需要每一层。简单 kernel 可以省略 (C), (G), (D)–(E)；
完整 forward+backward kernel 通常需要 (B)–(L) 全部。

---

## 3. 命名约定（grep 友好）

| 模式 | 含义 |
|------|------|
| `forward_ref` / `*_forward_ref` | PyTorch forward reference |
| `<op>_golden` / `torch_golden_reference` | 主 oracle（**始终是 grep 测试的入口名**） |
| 前导 `_` | 私有 helper（一个逻辑步骤，非公共 API） |
| `pypto_*` | 任何使用 pypto API 的代码（views, matmul, loops, tile shapes） |
| `_*_kernel_impl` | 包含 `pypto.loop` + sub-kernel 调用的实现体 |
| `*_kernel_npu` | `@pypto.frontend.jit` 入口（带类型化 tensor 签名） |
| `pypto_function` | torch 侧 launcher（layout 转换 + `kernel_npu(...)` + reshape） |

每个方向（forward / backward）保持 **一个主 golden 名称**，方便测试和文档 grep。

---

## 4. 参考实现约束（在头注释中文档化）

`<op>_golden.py` 中的 reference **不是任意 PyTorch 代码** — 它应当遵守你声明的规则。
在文件头注释中写明:

- **允许**: `matmul`, elementwise ops, `sum` over last dim, batch/head/chunk 显式循环。
- **禁止** (如果它们会复杂化 NPU 下沉或与 PyPTO 路径不一致):
  `cumsum`, `masked_fill`, `tril`/`triu` factory ops, `flip`, 高 rank tensor。

当 PyTorch op 被禁止时，用 **显式矩阵** 实现等价数学
（例如 `C_cum @ x` 替代 `cumsum`）。这保持 reference step 与 `pypto_*` block 的一对一对应。

---

## 5. Stage marker（长函数内部）

对于长 `forward_ref` 或 backward golden，使用 **labeled stages** 让它们映射到 PyPTO 模块:

```python
# ===== (A) stage description =====
# ===== (B) stage description =====
# ===== (C) stage description =====
```

在 `<op>_impl.py` 的 PyPTO fusion 注释中使用 **相同字母或名称**
（例如 "Fused Module A + B"），mismatch 时可以缩小到一对函数。

---

## 6. Tensor contract table

在 `<op>_impl.py` 的 module docstring 或专门注释块中维护一个简短的 tensor contract:

| Name | Shape (symbolic) | dtype | Producer | Consumer | Notes |
|------|------------------|-------|----------|----------|-------|
| q | [B, H, S, D] | bf16 | input | pypto_attention_score | — |
| scores | [B, H, S, S] | fp32 | pypto_attention_score | pypto_softmax | row-major |
| out | [B, H, S, D] | bf16 | pypto_attention_output | return | — |

---

## 7. 关键规则

### 7.1 Layer K 禁止 Python 循环

`pypto_function`（Layer K）中 **绝不允许** `for ... in range(...)` 来驱动 batch/seq/tile/chunk 计算。
迭代必须在 Layer I 用 `pypto.loop` 完成。

```python
# ❌ 错误（在 Layer K 的 pypto_function 中）
def pypto_function(x):
    out = torch.empty_like(x)
    for b in range(B):       # 错误! 不允许
        kernel_npu(x[b], out[b])
    return out

# ✅ 正确（在 Layer I 的 _<op>_kernel_impl 中）
def _<op>_kernel_impl(x, out):
    for b in pypto.loop(range(B), idx_name="b"):
        ...
```

### 7.2 Shape annotation（每个 tensor 行必带）

```python
q = pypto.view(q_in, [B, N, Sq, D])          # [B, N, Sq, D]
scores = pypto.matmul(q, k, pypto.DT_BF16)   # [B, N, Sq, Skv]
row_max = pypto.amax(scores, dim=-1)         # [B, N, Sq, Skv] -> [B, N, Sq, 1]
```

不需要标注: import 行, print/logging, Python 标量 (`tile_m = 16`)。

### 7.3 Tile config

- `pypto.set_vec_tile_shapes(...)` — 在每个 vector op 前必须设置（参见 `docs/api/config/pypto-set_vec_tile_shapes.md`）。
- `pypto.set_cube_tile_shapes(...)` — 在 `pypto.matmul` 前必须设置。
  每个 m/k/n 是 **2 元素 list** `[L0, L1]`（不是 1 元素 list）。
  常用安全 baseline: `[128, 128], [64, 128], [128, 256]`。
- `pypto.loop` 的 `idx_name` 在同一函数内必须唯一。

### 7.4 输出写回

```python
output[:] = result                                # 显式切片赋值
pypto.assemble(result, offset, output)            # 或 assemble
# ❌ output = result  — 错误! 这是 Python rebind，不是 device 写回
```

---

## 8. Staged 目录约定

开发期，`coding` agent 每次 dispatch 在 `custom/<op>/staged/` 中产出 **一组 3 个文件**:

```
staged/<op>_module<suffix_k>_impl.py     ← M_k 累积 PyPTO 实现
staged/<op>_module<suffix_k>_golden.py   ← M_k 累积 torch reference
staged/test_<op>_module<suffix_k>.py     ← M_k 测试驱动
```

`<suffix_k>` 是模块索引连接（`1`, `12`, `123`, ...）。

完成 M_N 后，最后的 staged set rename 到顶层成为最终交付:

```
staged/<op>_module1...N_impl.py    →   <op>_impl.py
staged/<op>_module1...N_golden.py  →   <op>_golden.py
staged/test_<op>_module1...N.py    →   test_<op>.py
```

`staged/` 目录在交付后 **保留**，便于 reviewer 追溯增量构建过程。

---

## 9. New kernel checklist

- [ ] 数学规格和 **符号 shape** 已写下
- [ ] `<op>_golden.py` 遵守文档化的约束
- [ ] backward golden（如有）匹配 forward cache contract
- [ ] 每个非平凡 loop body chunk 提取为 `_helper` 或 `pypto_*`，附 **一行角色注释**
- [ ] reference stage label 与 PyPTO 模块注释对齐
- [ ] JIT 签名匹配实际 buffer rank（dynamic vs static dimensions）
- [ ] `pypto_function` 文档化 **layout** 假设（row-major flatten 顺序、head 分组等）
- [ ] `test_<op>.py` 比较 **所有** 与 API 相关的输出（不只第一个）
- [ ] `pypto_function` 中无 `for ... in range(...)` 驱动算法迭代
