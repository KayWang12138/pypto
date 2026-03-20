# {OPERATOR_NAME} 性能测试报告

## 测试环境

| 项目 | 值 |
|------|-----|
| NPU 设备 | Ascend 910 (Chip {DEVICE_ID}) |
| CANN 版本 | 8.5.0 |
| 数据类型 | BF16 |

## 测试规格

| 参数 | 值 |
|------|-----|
| Batch Size (B) | {BATCH_SIZE} |
| Num Heads (N) | {NUM_HEADS} |
| Seq Len (S) | {SEQ_LEN} |
| Head Dim (D) | {HEAD_DIM} |

## 纯 Kernel 时间对比

| 实现 | 时间 (μs) | 时间 (ms) |
|------|----------|-----------|
| **PyPTO** | {PYPTO_TIME} | {PYPTO_TIME_MS} |
| **Ascend C** | {ASCEND_TIME} | {ASCEND_TIME_MS} |

## 数据来源

| 实现 | 文件 | 字段 |
|------|------|------|
| PyPTO | `bubble_analysis.log` | Core Total Work Time |
| Ascend C | `op_summary.csv` | Task Duration(us) |

## 结论

PyPTO 比 Ascend C {SLOWER/FASTER} {RATIO}x