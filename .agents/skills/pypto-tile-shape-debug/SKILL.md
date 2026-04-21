---
name: pypto-tile-shape-debug
description: 诊断并修复 set_cube_tile_shapes / set_vec_tile_shapes 失败。从错误消息反向推导具体的 [L0, L1] 值。当错误消息提及 "L0 size exceeded"、"L1 size exceeded"、"tile align"、"tile shape not set"、"enable_split_k"，或 matmul 在 tile-config 变更后立即崩溃时，使用此技能。
license: Internal
---

# Tile Shape 调试

当 `pypto.matmul` 或 Cube 算子在 tile-config 或编译阶段失败时，根本原因几乎总是违反了 `docs/api/config/pypto-set_cube_tile_shapes.md` 中 `set_cube_tile_shapes` 的某项约束。本技能将失败症状映射到具体的**补丁提案**（新的 `[L0, L1]` 值）——它不会修改生产代码。

Debug Agent 负责补丁提案；Coding Agent 按照 `.opencode/agents/coding.md` 应用补丁。

---

## 0. API 约定回顾（必须首先阅读）

```python
pypto.set_cube_tile_shapes(
    m: List[int],          # [mL0, mL1]  — 恰好 2 个元素
    k: List[int],          # [kL0, kL1]
    n: List[int],          # [nL0, nL1]
    enable_split_k: bool = False,
)
```

硬性约束（对每个轴 X ∈ {m, k, n}）：

- **参数本身的形状**：`len(X) == 2`。1 元素列表（`[16]`）或 3+ 元素列表是 bug。
- **顺序约束**：`0 < XL0 <= XL1`。
- **整除约束**：`XL1 % XL0 == 0`。
- **对齐约束**（非 FP32）：
  - `kL0, kL1, nL0, nL1` 必须**32 字节对齐**，即 `XLi * sizeof(dtype) % 32 == 0`。
  - FP16 / BF16：**16 元素**的倍数。
  - INT8：**32 元素**的倍数。
- **对齐约束（FP32）**：将 kL0/kL1/nL0/nL1 的 "32 字节" 替换为 "**16 元素**"。
- **A 矩阵 ND 转置**：mL0 也必须 32 字节对齐。
- **NZ 格式**：外轴 16 元素对齐，内轴 32 字节对齐。
- **L0 缓存预算**（dtype FP16/BF16/FP32，cDtype = FP32）：
  - `CeilAlign(mL0,16) * CeilAlign(kL0,16) * sizeof(aDtype) <= L0A_size`
  - `CeilAlign(nL0,16) * CeilAlign(kL0,16) * sizeof(bDtype) <= L0B_size`
  - `CeilAlign(mL0,16) * CeilAlign(nL0,16) * sizeof(FP32)  <= L0C_size`
- **L0 缓存预算**（dtype INT8，cDtype = INT32）：同样的 shape，align=32，各 dtype 对应 sizeof。
- **L1 缓存预算**：
  - `CeilAlign(mL1,16) * CeilAlign(kL1,16) * sizeof(aDtype)`
    `+ CeilAlign(nL1,16) * CeilAlign(kL1,16) * sizeof(bDtype)`
    `<= L1_size`
  - INT8 时将 16 替换为 32。
- **Bias（BTBuffer = 1 KB，上转型为 FP32）**：`nL0 * 4 <= 1024` → `nL0 <= 256`。
- **FixPipe（FixBuffer = 2 KB，scale 为 uint64）**：`nL0 * 8 <= 2048` → `nL0 <= 256`。
- **`enable_split_k=True`**：仅对**2D** 输入有效。3D/4D 必须使用 `enable_split_k=False`。

`CeilAlign(v, a) = ((v + a - 1) // a) * a`。

Atlas A2/A3 上的典型设备预算（请在 `docs/` 中按产品确认）：

| 缓存 | 典型大小 |
|--------|--------------|
| L0A    | 64 KB        |
| L0B    | 64 KB        |
| L0C    | 128 KB       |
| L1     | 512 KB       |

如果错误消息引用了某个限制值，请始终验证当前设备的实际预算。

---

## 1. 症状 → 根因路由表

在提出任何建议之前，通过匹配错误消息或 `validate_custom_kernel_layout.py` 的输出进行路由。

| 症状（在错误/CI 输出中） | 类别 | 跳转到 |
|---|---|---|
| `must be a 2-element list`（来自 `validate_custom_kernel_layout.py`） | 元素数量  | §2 |
| `mL1 % mL0 == 0` / `requires XL0 <= XL1`（来自验证器） | 整除性/顺序 | §3 |
| `tile shape not set` | 缺少调用 | §4 |
| `L0A size exceeded` / `L0B size exceeded` / `L0C size exceeded` / "L0 ... overflow" | L0 预算 | §5 |
| `L1 size exceeded` / "L1 buffer out of range" | L1 预算 | §6 |
| `alignment` / `% 32 != 0` / `% 16 != 0` / "not aligned" | 对齐 | §7 |
| `enable_split_k` 搭配 3D/4D 输入 | split_k 误用 | §8 |
| `BTBuffer` / `FixBuffer` 溢出 | Bias / FixPipe | §9 |
| 运行时结果错误但未崩溃；matmul 后接 vec 算子 | 缺少 `set_vec_tile_shapes` | §10 |

如果错误不匹配任何行，则回退到 §11 中的系统化流程。

---

## 2. 元素数量修复 — 列表必须是 `[L0, L1]`

**症状**：`validate_custom_kernel_layout.py` 报告 `\`m\` must be a 2-element list \`[mL0, mL1]\`, got 1-element list`。

**原因**：类似以下代码：

```python
pypto.set_cube_tile_shapes([16], [32], [64])      # BUG — 缺少 L1
```

**补丁提案**：选择 `L1 = L0 * k`，其中 `k ≥ 1` 为小整数，且仍满足 L0/L1 预算。
在没有其他约束时的安全默认值：

```python
pypto.set_cube_tile_shapes([16, 32], [32, 64], [64, 128])
```

如果 shape 已知：目标 `mL1 ≈ M`，`nL1 ≈ N`，`kL1 ≈ K`，然后选择 `L0` 为一个干净的除数（如 64 或 128），并确认 §5 和 §6 仍然满足。

---

## 3. 整除性 / 顺序修复

**症状**：验证器标记 `XL0 > XL1` 或 `XL1 % XL0 != 0`。

**补丁提案**：将 `L1` 向上取整到 `L0` 的最近倍数；如果超出 L1 预算，则缩小 `L0` 为 `L1` 的 2 的幂除数。

```
# 给定 [L0, L1] = [96, 128]  (128 % 96 = 32，失败)
# 修复方案 A:  [64, 128]     (128 % 64 == 0)  ✅
# 修复方案 B:  [96, 192]     (192 % 96 == 0)  ✅（如果 L1 预算允许）
```

---

## 4. "tile shape not set"

**症状**：在第一个 matmul 处出现 `tile shape not set`。

**原因**：从未调用 `set_cube_tile_shapes`，或在 matmul 之后调用，或在不同于 matmul 调用的函数作用域中设置。

**补丁提案**：在 **`@pypto.frontend.jit` 函数体的顶部**、任何 matmul 之前添加调用；如果 API 要求，在任何嵌套作用域变更后重新设置（参见 `skills/debugging/DEBUG.md §8`）。

如果混合使用 Cube 和 Vec 算子，还需调用 `set_vec_tile_shapes(...)`：

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])
    pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
    ...
```

---

## 5. L0 缓存超限

**症状（示例）**：

- `L0A size exceeded`、`L0A buffer overflow`，或错误引用 "L0A = 65536"
- `L0B`、`L0C` 同理

**步骤 1 — 确定哪个缓存**：

| 缓存 | 必须满足的公式                                                  |
|--------|-------------------------------------------------------------------------|
| L0A    | `CeilAlign(mL0, A_align) * CeilAlign(kL0, A_align) * sizeof(aDtype) <= L0A_size` |
| L0B    | `CeilAlign(nL0, A_align) * CeilAlign(kL0, A_align) * sizeof(bDtype) <= L0B_size` |
| L0C    | `CeilAlign(mL0, A_align) * CeilAlign(nL0, A_align) * sizeof(cDtype) <= L0C_size` |

`A_align = 16`（FP16/BF16/FP32），`32`（INT8）。`cDtype = FP32`（INT8 路径为 INT32）。

**步骤 2 — 缩小违规的 `L0`**：

- `L0A` 超限 → 将较大的 `mL0` 或 `kL0` 减半。
- `L0B` 超限 → 将较大的 `nL0` 或 `kL0` 减半。
- `L0C` 超限 → 将较大的 `mL0` 或 `nL0` 减半。

变更后始终检查 §3：缩小 `XL0` 后，如有需要也缩小 `XL1`，使 `XL1 % XL0 == 0` 仍然成立。

**步骤 3 — 健全性检查 L1**（§6），因为减小 `L0` 可能允许更大的 `L1`。

### 示例演练（FP16，L0A = 64 KB）

```
失败:   mL0=256, kL0=256, sizeof(FP16)=2 → 256*256*2 = 131072 > 65536 ❌
修复:   mL0=128, kL0=256                  → 128*256*2 = 65536  ≤ 65536 ✅
或:     mL0=256, kL0=128                  → 256*128*2 = 65536  ≤ 65536 ✅
```

---

## 6. L1 缓存超限

**症状**：`L1 size exceeded`，或错误引用 "L1 = 524288"。

**公式**：

```
CeilAlign(mL1, align) * CeilAlign(kL1, align) * sizeof(aDtype)
+ CeilAlign(nL1, align) * CeilAlign(kL1, align) * sizeof(bDtype)
<= L1_size
```

`align = 16`（FP16/BF16/FP32），`32`（INT8）。

**补丁策略** — 按优先级排序：

1. **先将 `kL1` 减半**：它同时影响 A 项和 B 项，因此能提供约 2 倍的缓解。
2. 如果不够，将 `mL1` / `nL1` 中较大的减半。
3. 保持 `L0 <= L1` 和 `L1 % L0 == 0`（§3）。
4. 如果工作负载 shape 不支持减小 `L1`，考虑启用 `enable_split_k=True`（仅限 2D 输入——见 §8）以在核心间分配 K。

### 示例演练（FP32，L1 = 512 KB）

```
失败:   mL1=512, kL1=256, nL1=512, sizeof(FP32)=4
         A = 512*256*4 = 524288
         B = 512*256*4 = 524288
         total = 1048576 > 524288 ❌
修复 A:  kL1 减半 → kL1=128
         A = 512*128*4 = 262144
         B = 512*128*4 = 262144
         total = 524288 ≤ 524288 ✅  (恰好达到上限)
修复 B:  nL1 减半 → nL1=256, kL1=256
         A = 512*256*4 = 524288
         B = 256*256*4 = 262144
         total = 786432 > 524288 ❌  (不够——按方案 A 减半 kL1)
```

---

## 7. 对齐错误

**症状（示例）**：`not aligned`、`% 32 != 0`、`% 16 != 0`、"tile align"。

**规则回顾**：

- FP16 / BF16 / FP32：`kL0, kL1, nL0, nL1` → **16 元素对齐**
- INT8：32 元素对齐
- A 转置（ND，shape `[K, M]`）：`mL0` 32 字节对齐
- NZ 格式：外轴 16 元素对齐，内轴 32 字节对齐

**补丁**：将违规值向上取整到所需对齐单位的最近倍数。如果这导致 §5 / §6 / §3 不满足，则缩小其他轴来补偿。

```
失败 (FP32): kL0=12  → 12 不是 16 的倍数 ❌
修复:         kL0=16                               ✅
```

---

## 8. `enable_split_k` 误用

**症状**：错误消息提及 `enable_split_k` 或 "split k not supported for 3D/4D"。

**规则**：`enable_split_k=True` 仅在输入为**2D** 时有效。对于 3D/4D 输入（attention、3D matmul），**必须为 False**（默认值）。

**补丁**：在 host wrapper 中将输入 reshape 为 2D 并以 2D 方式执行 matmul，或设置 `enable_split_k=False`。

```python
# 3D/4D 输入
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])  # 不使用 split_k

# 2D 输入，需要更多并行度
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

---

## 9. Bias / FixPipe 缓存溢出

- **BTBuffer（1 KB，bias 上转型为 FP32）**：`nL0 * 4 <= 1024` → `nL0 <= 256`。
- **FixBuffer（2 KB，scale 为 uint64）**：`nL0 * 8 <= 2048` → `nL0 <= 256`。

**补丁**：当 bias 或 FixPipe 激活时，将 `nL0` 上限设为 256；如果需要更多并行度，则增大 `nL1`（受 L1 预算限制，§6）。

---

## 10. 结果错误（非崩溃）：matmul + vec 但未设置 `set_vec_tile_shapes`

**症状**：混合使用 `matmul` 和向量算子（`mul`、`add`、逐元素操作）的 kernel 产生数值错误。没有明确的 tile-config 错误，但精度检查失败。

**原因**：仅使用 `set_cube_tile_shapes` **不够** —— AIV 算子（包括编译器插入的 `REGISTER_COPY`）仍然需要 `set_vec_tile_shapes`。参见 `skills/debugging/DEBUG.md §8` 的要点 "Do not rely only on `set_cube_tile_shapes`"。

**补丁提案**：**两者都调用**：

```python
pypto.set_vec_tile_shapes(1, 1, 128, 128)
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])
```

---

## 11. 系统化回退流程

如果症状不匹配 §2–§10：

1. **查找调用**：`grep -n "set_cube_tile_shapes\|set_vec_tile_shapes" custom/<op>/`。
2. **导出参数值**（即使是符号值）：每个是否为 2 元素列表？字面量整数是否满足 §3？
3. **手动重新计算每个 L0 / L1 公式**（§5, §6），使用当前 dtype 和 `docs/` 中的设备预算。
4. **识别第一个被违反的约束**；提出恢复该约束的最小变更。
5. 在任何变更后**重新推导 §3**（整除性、顺序）。
6. **重新检查 §5 / §6 / §9** —— 变更可能级联。
7. **将提案写入**计划文件 `custom/plan/<op>.md` 的 `## tile-shape patch proposal` 节下。不要自行修改 `_moduleN.py`；Coding Agent 按 `.opencode/agents/coding.md` 应用补丁。

---

## 12. 输出格式 — 补丁提案

将提案写入 `custom/plan/<op>.md`：

```markdown
## tile-shape patch proposal (cycle N)

- File:   custom/<op>/<op>_module<suffix>.py
- Line:   <失败的 set_cube_tile_shapes 调用的行号>
- Before:
    pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])
- After:
    pypto.set_cube_tile_shapes([128, 256], [64, 128], [128, 256])
- Root cause: L1 预算超限（见 §6）。在 FP32 和 L1=512 KB 时，
    (256*256*4) + (256*256*4) = 524288 已达到上限但结合
    kL1=256 需要减半 ⇒ kL1=128。
- Verification step: Coding Agent 重新运行验证；如果仍然失败，
    返回新的错误消息以便下一轮处理。
```

在每个提案中包含参考 `docs/api/config/pypto-set_cube_tile_shapes.md`，以便 Coding Agent 有唯一的权威来源。

---

## 13. 预算常量速查表

| 符号      | 典型值（A2/A3） | 覆盖来源            |
|-------------|----------------------|-----------------------------|
| `L0A_size`  | 65536 (64 KB)        | `docs/api/config/...`       |
| `L0B_size`  | 65536 (64 KB)        | `docs/api/config/...`       |
| `L0C_size`  | 131072 (128 KB)      | `docs/api/config/...`       |
| `L1_size`   | 524288 (512 KB)      | `docs/api/config/...`       |
| `BTBuffer`  | 1024 (1 KB)          | `docs/api/config/...`       |
| `FixBuffer` | 2048 (2 KB)          | `docs/api/config/...`       |

如果设备的实际错误文本引用了不同的数字（如 `L1 = 786432`），请信任错误消息并相应更新你的计算。

---

## 14. 职责边界（反模式检查）

本技能产出**补丁提案**，而非提交。Debug Agent：

- 不得打开或修改 `custom/<op>/<op>_module*.py`。
- 必须将提案写入 `custom/plan/<op>.md`。
- 每个 `.opencode/agents/debug.md` 最多 3 轮循环；如果仍然失败，上报给 Lead。

Coding Agent 应用补丁，然后 Verification Agent 重新判定。
