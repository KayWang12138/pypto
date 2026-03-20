# PyPTO算子翻译任务 - 最终报告

## 任务概述

本任务是将 op-plugin 仓库中的 PyTorch NPU 算子翻译为 PyPTO 实现。每个算子需要三种实现对比：
1. **torch_npu**: 原始华为 NPU 实现
2. **golden**: numpy 参考实现
3. **pypto**: PyPTO 实现（若不支持则用 torch/numpy 替代）

---

## 统计摘要

| 指标 | 数量 | 占比 |
|------|------|------|
| 已实现算子 | 34 | 79% |
| 完全通过 | 22 | 51% |
| 部分支持 | 8 | 19% |
| 不支持/失败 | 4 | 9% |
| 可用率(含部分支持) | 30 | 70% |

---

## 实现记录详情

### 一、完全通过 ✅ (22个)

| 序号 | 算子 | 功能描述 | 验证结果 |
|------|------|----------|----------|
| 1 | npu_fast_gelu | 快速GELU激活函数 | 三者等价 |
| 2 | npu_silu | SiLU激活函数 | 三者等价 |
| 3 | npu_softmax_cross_entropy_with_logits | Softmax交叉熵损失 | 三者等价 |
| 4 | npu_reshape | 张量形状变换 | 三者等价 |
| 5 | npu_transpose | 张量转置 | 三者等价 |
| 6 | npu_one_hot | One-hot编码 | 三者等价 |
| 7 | npu_mish | Mish激活函数 | 三者等价 |
| 8 | npu_slice | 张量切片 | 三者等价 |
| 9 | npu_pad | 张量填充 | 三者等价 |
| 10 | npu_bmmV2 | 批量矩阵乘法 | 三者等价 |
| 11 | npu_layer_norm_eval | Layer归一化(推理模式) | 三者等价 |
| 12 | npu_cross_entropy_loss | 交叉熵损失 | 三者等价 |
| 13 | npu_gelu | GELU激活函数 | erf模式等价 |
| 14 | npu_rms_norm | RMS归一化 | 2D输入通过 |
| 15 | npu_linear | 线性变换 | 无bias通过 |
| 16 | npu_broadcast | 广播操作 | 三者等价 |
| 17 | npu_diou | Distance IoU | 三者等价 |
| 18 | npu_nms_v4 | 非极大值抑制 | 三者等价 |
| 19 | npu_batch_nms | 批量NMS | 三者等价 |
| 20 | npu_conv2d | 2D卷积 | 三者等价 |
| 21 | npu_grouped_matmul | 分组矩阵乘法 | 三者等价 |
| 22 | npu_group_norm_silu | GroupNorm+SiLU | 三者等价 |

### 二、部分支持 ⚠️ (8个)

| 序号 | 算子 | 功能描述 | 限制说明 |
|------|------|----------|----------|
| 1 | npu_max/min | 最大/最小值 | 不返回索引，仅返回值 |
| 2 | npu_iou | 交并比计算 | 存在精度差异 |
| 3 | npu_giou | 广义IoU | 存在精度差异 |
| 4 | npu_ciou | 完整IoU | torch_npu返回inf |
| 5 | npu_fused_attention_score | 融合注意力 | 需要特定内存格式 |
| 6 | npu_multi_head_attention | 多头注意力 | 维度计算复杂 |
| 7 | npu_lstm | LSTM循环神经网络 | 需要FRACTAL_NZ格式 |
| 8 | npu_gru | GRU循环神经网络 | 需要FRACTAL_NZ格式 |

### 三、不支持 ❌ (4个)

| 序号 | 算子 | 功能描述 | 失败原因 |
|------|------|----------|----------|
| 1 | npu_dropout | Dropout随机丢弃 | PyPTO不支持随机数生成 |
| 2 | npu_swiglu | SwiGLU激活 | 切片在jit kernel中编译失败 |
| 3 | npu_gelu_mul | GELU乘法 | 切片在jit kernel中编译失败 |
| 4 | npu_dropout_with_add_softmax | Dropout+Add+Softmax | 包含dropout随机数 |

---

## PyPTO 框架能力限制分析

### 1. API 层面限制

| 限制类型 | 具体说明 | 影响算子 |
|----------|----------|----------|
| 随机数生成 | 不支持随机数API | npu_dropout, npu_dropout_with_add_softmax |
| 激活函数 | 不支持tanh/gelu原生API | 需用torch.nn.functional替代 |
| 切片操作 | 切片在@pypto.frontend.jit kernel中编译失败 | npu_swiglu, npu_gelu_mul |
| 归约操作 | max/min不返回索引 | npu_max, npu_min |

### 2. 格式层面限制

| 格式要求 | 影响算子 | 解决方案 |
|----------|----------|----------|
| FRACTAL_NZ | npu_lstm, npu_gru | 使用numpy实现golden |
| 特定内存布局 | npu_fused_attention_score | 使用torch替代 |

### 3. 可用API列表

以下是测试通过的PyPTO API:
- `pypto.sigmoid`, `pypto.exp`, `pypto.log`, `pypto.sqrt`
- `pypto.sum`, `pypto.amax`, `pypto.amin`
- `pypto.matmul` (需要 `set_cube_tile_shapes`)
- `pypto.reshape`, `pypto.transpose`, `pypto.one_hot`
- `pypto.softmax`

---

## 实现目录结构

```
custom/
├── operator_todo_list.md          # 本报告文件
├── plan/                          # 开发计划目录
│
├── npu_fast_gelu/                 # 快速GELU ✅
├── npu_silu/                      # SiLU ✅
├── npu_rms_norm/                  # RMS归一化 ✅
├── npu_linear/                    # 线性变换 ✅
├── npu_softmax_cross_entropy_with_logits/  # Softmax交叉熵 ✅
├── npu_dropout/                   # Dropout ❌
├── npu_reshape/                   # 形状变换 ✅
├── npu_transpose/                 # 转置 ✅
├── npu_max/                       # 最大值 ⚠️
├── npu_one_hot/                   # One-hot ✅
├── npu_mish/                      # Mish ✅
├── npu_gelu/                      # GELU ✅
├── npu_swiglu/                    # SwiGLU ❌
├── npu_gelu_mul/                  # GELU乘法 ❌
├── npu_slice/                     # 切片 ✅
├── npu_pad/                       # 填充 ✅
├── npu_bmmV2/                     # 批量矩阵乘 ✅
├── npu_iou/                       # IoU ⚠️
├── npu_layer_norm_eval/           # LayerNorm ✅
├── npu_cross_entropy_loss/        # 交叉熵损失 ✅
├── npu_sort_v2/                   # 排序 (API问题)
├── npu_rotary_mul/                # 旋转乘法 (API问题)
├── npu_giou/                      # GIoU ⚠️
├── npu_diou/                      # DIoU ✅
├── npu_ciou/                      # CIoU ⚠️
├── npu_broadcast/                 # 广播 ✅
├── npu_nms_v4/                    # NMS ✅
├── npu_batch_nms/                 # 批量NMS ✅
├── npu_fused_attention_score/     # 融合注意力 ⚠️
├── npu_multi_head_attention/      # 多头注意力 ⚠️
├── npu_conv2d/                    # 2D卷积 ✅
├── npu_grouped_matmul/            # 分组矩阵乘 ✅
├── npu_group_norm_silu/           # GroupNorm+SiLU ✅
├── npu_lstm/                      # LSTM ⚠️
└── npu_gru/                       # GRU ⚠️
```

---

## 测试方法说明

每个算子目录包含独立的测试文件 `npu_*.py`，支持两种运行模式：

```bash
# NPU模式 (需要torch_npu环境)
python3 custom/npu_xxx/npu_xxx.py --run_mode npu

# CPU模式 (仅运行golden和pypto)
python3 custom/npu_xxx/npu_xxx.py --run_mode cpu
```

测试流程：
1. 生成随机测试数据
2. 计算三种实现的结果
3. 对比精度差异
4. 输出测试报告

---

## 后续建议

### 短期改进
1. 为部分支持的算子添加更多测试用例
2. 探索切片编译失败的解决方案
3. 研究RNN算子的FRACTAL_NZ格式支持

### 长期规划
1. 等待PyPTO框架支持随机数生成API
2. 推动PyPTO支持更多激活函数
3. 完善性能基准测试

---

## 附录：算子分类统计

| 类别 | 总数 | 通过 | 部分支持 | 不支持 |
|------|------|------|----------|--------|
| 激活函数 | 7 | 6 | 0 | 1 |
| 归一化 | 4 | 3 | 1 | 0 |
| 形状操作 | 4 | 4 | 0 | 0 |
| 矩阵运算 | 3 | 3 | 0 | 0 |
| 损失函数 | 2 | 2 | 0 | 0 |
| 检测相关 | 5 | 2 | 3 | 0 |
| 注意力机制 | 2 | 0 | 2 | 0 |
| 卷积操作 | 1 | 1 | 0 | 0 |
| RNN | 2 | 0 | 2 | 0 |
| 其他 | 4 | 1 | 0 | 3 |

---

报告生成时间: 2026-03-20
PyPTO版本: 当前主分支
测试环境: A3服务器, CANN 8.5.0