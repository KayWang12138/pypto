# Issue: Tile 配置违反约束导致静默精度失败而非报错

**标题**: `[Bug-Report|缺陷反馈]: Tile 配置违反约束导致静默精度失败而非报错`

---

### Describe the current behavior / 问题描述 (Mandatory / 必填)

当 `pypto.set_cube_tile_shapes` 或 `pypto.set_vec_tile_shapes` 的配置违反约束条件时，程序不会抛出错误或警告，而是静默地产生精度失败。这使得用户难以诊断问题，需要通过大量调试才能发现问题根源。

**具体问题场景**：

1. **L0B Buffer 空间约束违反**：
   - 文档规定 `L0B_size = 65536 bytes (64KB)`
   - 当 `nL0 × kL0 > 64KB`（如 `512×512 = 256KB`）时，不报错，但触发 Spill 机制导致精度失败

2. **vector_tile_shape 与 scale tensor 维度不匹配**：
   - `scaled_mm` 的 scale tensor shape 为 `[K/64, M/N, 2]`（第三维固定为 2）
   - 当 `vector_tile_shape[3]` 设置为非 2 的值（如 32）时，导致 scale tensor out-of-bounds 访问
   - 不报错，仅产生精度失败

---

### Environment / 环境信息 (Mandatory / 必填)

- **服务器/NPU 型号**: Ascend 910B
- **PyPTO 版本/Commit**: 最新版本
- **CANN 版本**: 8.5.0
- **Python 版本**: 3.10+
- **操作系统**: Linux

---

### Steps to reproduce the issue / 重现步骤 (Mandatory / 必填)

**场景 1: L0B 空间约束违反**

```python
import pypto

@pypto.frontend.jit()
def test_kernel(a, b, scale_a, scale_b):
    # L0B 配置: nL0=512, kL0=512, FP8 dtype
    # L0B占用: 512×512×1B = 256KB > 64KB (违反约束)
    pypto.set_cube_tile_shapes(
        m_tile_shape=[32, 32],
        k_tile_shape=[512, 512],  # kL0=512 过大
        n_tile_shape=[512, 2048]
    )
    pypto.set_vec_tile_shapes(1, 8, 2048, 2)
    return pypto.scaled_mm(a, b, pypto.DT_FP32, scale_a, scale_b)

# 运行后精度验证失败，但没有任何错误信息
```

**场景 2: vector_tile_shape 维度不匹配**

```python
import pypto

@pypto.frontend.jit()
def test_kernel(a, b, scale_a, scale_b):
    # scale tensor shape: [K/64, M, 2] -> 第三维固定为 2
    # vector_tile_shape[3] = 32 -> 与 scale tensor 第三维不匹配
    pypto.set_cube_tile_shapes([32, 32], [64, 256], [512, 2048])
    pypto.set_vec_tile_shapes(1, 8, 2048, 32)  # dim3=32 错误，应为 2
    return pypto.scaled_mm(a, b, pypto.DT_FP32, scale_a, scale_b)

# 运行后精度验证失败，但没有任何错误信息
```

---

### Describe the expected behavior / 预期结果 (Mandatory / 必填)

当 Tile 配置违反约束条件时，期望：

1. **编译阶段检查**：在编译时检测明显违反的约束（如 L0B 空间超过限制）
2. **运行时警告/错误**：当检测到可能导致精度问题的配置时，抛出警告或错误信息
3. **文档完善**：
   - 在 `pypto.set_vec_tile_shapes` 文档中补充与 `scaled_mm` scale tensor 维度的对应关系说明
   - 明确说明违反约束的具体行为（报错 vs 静默失败）

---

### Related log / screenshot / 日志 / 截图 (Mandatory / 必填)

**实际运行输出**：
- 无错误信息
- 精度验证失败：`assert_allclose` 报错 `rtol/atol` 不满足

**用户反馈**：
> "tile配置不对会精度失败而不是报错"

**相关文档片段**：

`docs/api/config/pypto-set_cube_tile_shapes.md` 第48-63行说明了 L0B 空间约束：
```
CeilAlign(nL0,16) * CeilAlign(kL0,16) * sizeof(bDtype) <= L0B_size
其中 L0B_size = 65536 bytes
```

`docs/api/config/pypto-set_vec_tile_shapes.md` 约束说明过于简单：
```
TileShape需要满足以下约束条件：
每个维度必须大于0。
```

`docs/api/operation/pypto-scaled_mm.md` 第30-31行说明了 scale tensor shape：
```
输入量化参数shape为：当输入量化参数非转置时，对应输入shape为[M, K/64, 2]
```

但没有说明 `vector_tile_shape` 与 scale tensor 各维度的对应关系。

---

### Special notes for this issue/备注 (Optional / 选填)

此问题在开发 `quant_grouped_matmul_inplace_add_mx` 算子过程中发现，经过多轮调试才确定问题根源：

1. **L0B 约束**：最初配置 `nL0=512, kL0=512` 导致 256KB > 64KB，触发 Spill
2. **vector_tile_shape 约束**：`dim3=32` 导致 scale tensor out-of-bounds 访问

正确配置应为：
- `k_tile_shape=[64, 256]`（kL0=64，L0B=512×64×1B=32KB）
- `vector_tile_shape=[1, K_block//64, nL1, 2]`（dim3 必须为 2）

建议在文档中补充 `vector_tile_shape` 各维度与 scale tensor 的对应关系公式：

```python
# scaled_mm 的 scale tensor shape: [K/64, M/N, 2]
vector_tile_shape = [
    1,               # dim0: batch (固定)
    K_block // 64,   # dim1: 对应 scale tensor 第一维
    nL1,             # dim2: 对应 N 维度切分粒度
    2                # dim3: 必须为 2，对应 scale tensor 第三维
]
```

---

## 手动创建说明

请在 GitCode 平台手动创建 Issue：

1. 打开 https://gitcode.com/cann/pypto/issues/new
2. 标题填写: `[Bug-Report|缺陷反馈]: Tile 配置违反约束导致静默精度失败而非报错`
3. 内容复制上述 markdown 内容（不包括本说明）
4. 提交 Issue