# ScatterPaKvCache 算子

## 算子概述

ScatterPaKvCache 算子用于更新 KV Cache 中指定位置的 key 和 value 值，是 Page Attention 推理中的核心操作之一。

## 数学公式

$$
\text{keyCache}[\text{slotMapping}[i]] = \text{key}[i], \quad \forall i \in [0, \text{num\_tokens})
$$

$$
\text{valueCache}[\text{slotMapping}[i]] = \text{value}[i], \quad \forall i \in [0, \text{num\_tokens})
$$

其中 `slotMapping` 指定了每个 token 在 cache 中的存储位置。

## 实现说明

### 当前实现

本实现基于 `pypto.scatter_update` API，实现了场景二（Norm 模式）的基础版本：

- **输入**：
  - `key`: [batch * seq_len, num_heads, head_dim]
  - `value`: [batch * seq_len, num_heads, head_dim]
  - `key_cache`: [num_blocks, block_size, num_heads, head_dim]
  - `value_cache`: [num_blocks, block_size, num_heads, head_dim]
  - `slot_mapping`: [batch * seq_len]

- **输出**：
  - 更新后的 `key_cache` 和 `value_cache`

### 关键 API

```python
pypto.scatter_update(input, dim, index, src)
```

- `input`: 待更新的 cache tensor
- `dim`: 更新维度，固定为 -2
- `index`: slot mapping 索引
- `src`: 待写入的数据

## 编译运行

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux

# 编译并安装 PyPTO
python3 build_ci.py -f python3 --disable_auto_execute
pip install -U build_out/pypto-*.whl --force-reinstall
```

### 运行测试

```bash
python3 custom/scatter_pa_kv_cache/scatter_pa_kv_cache.py
```

## 测试结果

```
============================================================
Test: ScatterPaKvCache
============================================================
Input key shape: torch.Size([8, 1, 64])
Input value shape: torch.Size([8, 1, 64])
Key cache shape: torch.Size([16, 16, 1, 64])
Value cache shape: torch.Size([16, 16, 1, 64])
Slot mapping shape: torch.Size([8])
Slot mapping values: [22, 38, 41, 54, 86, 122, 207, 254]
Key cache max diff: 0.000000
Value cache max diff: 0.000000
✓ ScatterPaKvCache test passed
============================================================
```

## 已知限制

1. **当前版本限制**：
   - 仅支持 `num_heads=1` 的场景
   - 使用 BF16 数据类型

2. **多 head 支持**：
   - 多 head 场景需要使用循环为每个 head 单独调用 `scatter_update`
   - 需要进一步研究 `pypto.loop` 与 `scatter_update` 的正确组合方式

3. **未实现的功能**：
   - PA_NZ 内存排布格式
   - scatter_mode 参数（Alibi、Rope、Omni、Nct）
   - compressLensOptional、seqLensOptional 等可选参数

## 文件结构

```
custom/scatter_pa_kv_cache/
├── scatter_pa_kv_cache.py    # 算子实现和测试
└── README.md                  # 本文档
```

## 参考资料

- [PyPTO scatter_update API 文档](../../docs/api/operation/pypto-scatter_update.md)
- [原始 ScatterPaKvCache 算子文档](/mnt/workspace/gitCode/cann/ops-transformer/attention/scatter_pa_kv_cache/README.md)

## 开发状态

- [x] 基础功能实现（单 head）
- [x] 精度验证通过
- [ ] 多 head 支持
- [ ] 性能优化
- [ ] 完整场景支持