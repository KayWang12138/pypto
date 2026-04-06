# PyPTO 已知限制（实战积累）

本文件记录 autodev 算子开发过程中发现的、docs/api 未充分说明的框架限制。
pypto-api-explorer 生成 api_report.md 时应参考此文件，将涉及的限制写入约束清单。

## 编译器 / 运行时

### pypto.loop 不支持跨迭代状态传递（carry-forward）
- 循环体内对 Python 变量的重赋值不影响下一迭代的计算图
- view+assemble 读写同一 tensor 触发 cycle detected 错误
- **Workaround**: 用 Python `range()` 代替 `pypto.loop`，编译时展开为固定步数
- **影响**: 所有递推类算子（RNN/LSTM/GRU/SSM），至少 6-8 个算子
- **来源**: rg_lru, lightning_attention

### pypto.mul column-expand broadcast 编译失败
- FP32/FP16 文档声称支持多维广播，但倒数第二轴从 1 广播到 N（如 `[1,D,D]*[1,1,D]`）触发 TCOLEXPANDMUL 编译错误
- **Workaround**: 改为 last-axis broadcast（如 `[1,D,D]*[1,D,1]`），或转置后再乘
- **影响**: 矩阵-向量乘、外积等操作
- **来源**: mlstm, gated_delta_net

### pypto.where 编译失败（文档与实际矛盾）
- docs/api/operation/pypto-where.md 声称支持 DT_FP32/FP16/BF16，但实际编译不通过
- **Workaround**: 算术替代 `input * (1 - mask_float) + value * mask_float`
- **影响**: 所有条件选择操作
- **来源**: masked_fill

### pypto.triu 对 -inf 输入产生 NaN
- 填充区域产生 NaN 而非预期的 0
- **Workaround**: 用 `log(1 - triu(ones))` 或算术方式构造 causal mask
- **来源**: causal_mask_generation

### PyPTO parser 不支持闭包变量
- JIT kernel 内无法引用外部作用域的 Python 变量
- **Workaround**: 所有值通过函数参数显式传入
- **来源**: sinusoidal_position_encoding

## Tile 约束

### vec TileShape 最后一维须满足 32 字节对齐
- FP32 至少 8 元素，FP16/BF16 至少 16 元素
- tile `(1,D,1)` 最后一维 1 元素 = 4 字节 < 32 字节，会报 `CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0`
- **Workaround**: tile 最后一维设为满足对齐的值（如 `(1,D,D)`），expand_clone/reshape 可在 tile > 实际 shape 时正常工作
- **来源**: mlstm

## 硬件限制

### pypto.conv 仅支持 Ascend 950
- docs/api 已明确标注，910 上编译失败
- **Workaround**: 用 matmul + im2col 实现卷积
- **来源**: transposed_conv, conv2d

## dtype 文档不准确

- pypto.sum: 文档标注仅 DT_FP32，但 BF16 实际可用
- pypto.sigmoid: 文档标注仅 DT_FP32，但 BF16 实际可用
- pypto.softmax: 仅支持 dim=-1（文档未明确说明）
