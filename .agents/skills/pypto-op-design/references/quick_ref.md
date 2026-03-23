# PyPTO 开发速查

> 详细信息通过搜索 docs/ 获取，本文件仅提供核心原则和约束。

---

## 1. Tiling 原则

### 1.1 算子类型判断

```
含 matmul/@ → Cube → set_cube_tile_shapes
仅逐元素/归约 → Vector → set_vec_tile_shapes
混合 → 两者都需要
```

### 1.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| Vector TileShape | 每维 > 0，最多 4 维 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Cube 必须设置 | matmul 前必须调用 set_cube_tile_shapes | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| 32B 对齐 | 尾轴需满足对齐要求 | `docs/tutorials/development/tiling.md` |

---

## 2. Loop 原则

### 2.1 是否需要 Loop

```
所有轴编译期已知 & 无多步骤分块 → 不需要 loop
存在动态轴 → 需要 pypto.loop 或 pypto.loop_unroll
多步骤计算（如 attention 分块）→ 需要 loop
```

### 2.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| 静态轴优先 Python for | pypto.loop 将静态轴展开增加编译复杂度 | `docs/tutorials/debug/performance.md` |
| 动态轴使用 pypto.loop | 并补齐边界控制 | `docs/tutorials/development/loops.md` |

### 2.3 标准写法

```python
# 静态轴 — Python for
for i in range(num_heads):
    head_i = pypto.view(x, [seq_len, head_dim], [0, i * head_dim])

# 动态轴 — pypto.loop
for i in pypto.loop(batch_size, name="LOOP_BATCH"):
    x_i = pypto.view(x, [seq_len, hidden], [i * seq_len, 0])
```

---

## 3. Runtime 硬约束

| 规则 | 说明 | 证据 |
|------|------|------|
| run_mode | 0=NPU，1=模拟器 | `docs/api/config/pypto-set_runtime_options.md` |
| NPU 需 CANN | run_mode=0 时需 source CANN 环境 | `docs/install/prepare_environment.md` |

### 示例

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def kernel(...):
    ...
```

---

## 4. from_torch 约束

- **dtype**: FP16/BF16/FP32/INT8-64/BOOL
- **contiguous**: 必须连续（is_contiguous() == True）
- **证据**: `docs/api/others/pypto-from_torch.md`

---

## 5. 搜索优先级

```
docs/（官方文档）→ 最高优先级
models/（生产代码）→ 次优先级
examples/（示例）→ 参考优先级
```
