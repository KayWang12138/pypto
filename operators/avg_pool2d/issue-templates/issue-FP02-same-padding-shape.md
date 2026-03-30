# [Bug] avg_pool2d golden SAME padding 输出 shape 计算与 PyTorch 不一致

> **Issue 类型**: Bug Report
> **优先级**: P1 (高)
> **来源**: avg_pool2d 算子开发断裂点检测
> **断裂点 ID**: FP-02
> **生成时间**: 2026-03-29

---

## Bug 描述

在 golden 实现验证过程中，SAME_large 测试用例的输出 shape 不匹配。预期输出 shape 为 [8, 64, 28, 28]，但实际得到 [8, 64, 27, 27]。这表明 SAME padding 的 ceil 模式实现与 PyTorch 的行为存在差异。

## 复现步骤

1. 使用 PyTorch avg_pool2d 实现 SAME padding golden
2. 输入 shape: [8, 64, 56, 56], kernel=(3,3), stride=(2,2)
3. 预期输出 shape: [8, 64, 28, 28]
4. 观察到 shape 不匹配

```python
import torch
import torch.nn.functional as F

# SAME padding 计算
input_tensor = torch.randn(8, 64, 56, 56)
kernel_size = 3
stride = 2

# SAME padding: output = ceil(input / stride)
# 预期: ceil(56 / 2) = 28
output = F.avg_pool2d(input_tensor, kernel_size, stride=stride, padding=padding)
print(output.shape)  # 应该是 [8, 64, 28, 28]
```

## 期望行为

SAME padding 模式下，输出 shape 应该为 `ceil(in_h / stride)`，即 28x28。

```python
# 正确的 SAME padding 实现
output_h = math.ceil(input_h / stride_h)
output_w = math.ceil(input_w / stride_w)
```

## 实际行为

初始实现使用 floor 模式，导致输出 shape 为 27x27。

```
[SAME_large]: FAILED - shape mismatch. Expected (8, 64, 28, 28), got torch.Size([8, 64, 27, 27])
```

修复后的验证通过：
```
[SAME_large]: PASSED - shape torch.Size([8, 64, 28, 28])
```

## 根本原因

`pypto-golden-generator` skill 在生成 SAME padding 的 golden 实现时，没有正确设置 `ceil_mode=True`。

## 建议改进

### 1. Skill 改进

`pypto-golden-generator` skill 应该自动处理 SAME padding 的 ceil_mode 设置：

```python
# 在 golden 生成逻辑中添加
if padding_mode == "SAME":
    ceil_mode = True  # SAME padding 需要 ceil_mode
```

### 2. 增加验证

在 golden 生成后，自动验证 SAME/VALID 两种模式的输出 shape 是否符合预期：

```python
def validate_padding_mode(golden_output, input_shape, kernel, stride, mode):
    if mode == "SAME":
        expected_h = math.ceil(input_shape[2] / stride[0])
        expected_w = math.ceil(input_shape[3] / stride[1])
    else:  # VALID
        expected_h = math.floor((input_shape[2] - kernel[0]) / stride[0] + 1)
        expected_w = math.floor((input_shape[3] - kernel[1]) / stride[1] + 1)

    assert golden_output.shape[2:] == (expected_h, expected_w)
```

### 3. 文档补充

在 skill 文档中添加 SAME padding 与 PyTorch API 的对照说明。

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672d75a8a3f97762083892c71f
- **服务器类型**: Ascend 910B (A3)
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 相关链接

- 断裂点报告: `operators/avg_pool2d/fracture-point-2026-03-29.md`
- 参考实现: `models/experimental/vector/AvgPool2d/avg_pool2d.py`
- PyTorch avg_pool2d: https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html

---

**建议标签**: `bug`, `golden-generator`, `padding`, `needs-triage`
