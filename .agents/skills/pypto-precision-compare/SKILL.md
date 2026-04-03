---
name: pypto-precision-compare
description: PyPTO 算子精度问题调试技能。提供三种精度对比方法：中间结果校验法（使用 pypto.pass_verify_save 和 torch.save）、Pass 校验方法和二分对比方法（使用检查点 tensor）。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题调试技能

提供三种精度对比方法，用于快速定位 PyPTO 算子中导致精度问题的具体 op 或 Pass。

## 调试路线总览

```
查看 Tensor Graph 校验结果
    │
    ├── FAIL → 使用 pass_verify_save 保存中间结果比对，找出首个失败的 op
    │           参考 → verify.md（中间结果校验法）
    │
    └── PASS → 进行 Pass 校验，找到首个出错的 Pass，dump Pass 数据对比
                │
                ├── Pass 校验 FAIL → 定位出错的 Pass/OP
                │                     参考 → pass.md（Pass 校验）
                │
                └── Pass 校验 PASS → 上板二分定位，找到首个出错的 op
                                      参考 → binary-search.md（二分对比方法）
```

## 精度校验流程

### 步骤一：开启 tensor_graph 校验

**编译安装**（最初执行一次即可）：

```bash
python3 -m pip install . --verbose --no-build-isolation
```

在 PyPTO 算子实现文件中配置 `verify_options`，开启 tensor_graph 校验：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,  # 保存中间tensor数据用于分析
    "pass_verify_pass_filter": []     # 跳过pass校验
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
) -> pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    return input0 + input1
```

设置 Golden 数据（**必须在算子运行前设置**）：

```python
# 计算 golden
torch_output = torch.add(input_data0, input_data1)

# 设置 golden（必须放在 pypto 算子运行前设置！）
pypto.set_verify_golden_data(goldens=[None, None, torch_output])

# 执行 pypto 算子
pypto_output = your_kernel(input_data0, input_data1)
```

**关键注意事项**：
- `pypto.set_verify_golden_data()` 必须在 `your_kernel()` 调用**之前**执行
- 如果放在 pypto 算子运行之后设置，仅能校验 Pass 阶段，无法校验 Tensor Graph 阶段
- goldens 列表顺序对应算子的输入输出数量：`[input0, input1, ..., output]`
- goldens 列表中的输入项不参与校验，统一设置为 `None`

### 步骤二：查看验证结果

运行测试后，在最新的 `{work_path}/output/output_*/verify_*/verify_exception.log` 目录下生成Tensor Graph校验结果：

**查看校验结果**：打开 `verify_exception.log` 文件，根据日志内容判断下一步操作：

- **tensor_graph Verify FAIL** → 前端代码问题，参考"情况 A"
- **tensor_graph Verify PASS，Pass 级别 Verify FAIL** → Pass 处理问题，参考"情况 B"
- **所有验证 PASS 但上板精度不对** → Codegen/Machine 问题，参考"情况 C"

### 步骤三：根据校验结果选择调试路线

参考上方"调试路线总览"，根据 `verify_exception.log` 中的校验结果判断下一步操作。

#### 情况 A：tensor_graph Verify FAIL → 前端代码问题

| 项目 | 说明 |
|-----|------|
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **原因** | 前端代码书写问题，构图阶段已出错 |
| **处理** | 使用**中间结果校验法**定位首个失败的 op |

**操作步骤**：
1. 阅读 [verify.md](reference/verify.md) 了解详细步骤
2. 在 kernel 中添加 `pypto.pass_verify_save()` 调用
3. 在 golden 中添加 `torch.save()` 调用
4. 运行测试生成数据
5. 使用对比工具分析结果，找出首个不匹配的检查点

#### 情况 B：tensor_graph Verify PASS → 进入 Pass 校验

| 项目 | 说明 |
|-----|------|
| **日志特征** | tensor_graph 阶段无报错 |
| **原因** | 前端构图正确，问题可能在 Pass 处理阶段 |
| **处理** | 使用 Pass 校验定位首个出错的 Pass |

**操作步骤**：
1. 阅读 [pass.md](reference/pass.md) 了解详细步骤
2. 分析验证结果，查看 `{work_path}/output/output_*/` 目录下的验证数据：
3. 根据错误码和验证阶段定位问题：
   - **具体 OP 报错**（`0xB200FU`）：查看对应 Pass 的 IR 图，分析 OP 属性
   - **精度问题**（`0xB4001U`）：使用 `pass_compare.py` 对比失败 Pass 与前置 Pass

#### 情况 C：Pass 校验也 PASS 精度仍不对 → 上板二分定位

| 项目 | 说明 |
|-----|------|
| **日志特征** | 所有验证都 PASS，但上板结果与 golden 不一致 |
| **原因** | 问题可能在 Codegen 或 Machine 执行阶段 |
| **处理** | 使用上板 dump 能力，二分打印上板数据找到首个出错的 op |

**操作步骤**：
1. 阅读 [binary-search.md](reference/binary-search.md) 了解详细步骤
2. 修改 kernel 函数签名，添加检查点 tensor 参数
3. 修改 golden 函数，返回检查点数据
4. 修改测试函数，创建检查点 tensor 并对比
5. 从关键计算点开始，二分定位精度问题，直到找到具体出错的 op

## 方法选择

### 方法对比

| 特性 | 中间结果校验法 | 二分对比方法 | Pass 校验 |
|------|--------------|--------------|-----------|
| **实现方式** | 使用 `pypto.pass_verify_save()` 保存到文件，使用 `torch.save()` 保存 golden | 使用检查点 tensor 作为输入参数，在内存中直接对比 | 使用 `enable_pass_verify` 自动校验各 Pass 中间结果 |
| **适用场景** | Tensor Graph 校验失败，需要快速定位首个失败的 op | Pass 校验通过但上板精度不对，需要上板真实数据 | Tensor Graph 校验通过，需要定位出错的 Pass |
| **循环支持** | 只保存 `idx=0` 的数据 | 支持保存所有循环迭代数据 | 自动处理所有循环 |
| **代码修改** | 不需要修改 kernel 函数签名 | 需要修改 kernel 函数签名，添加检查点参数 | 只需配置 verify_options |
| **数据类型** | 直接保存原始类型，对比时统一转换 | 在内存中直接对比，类型需一致 | 自动处理类型转换 |
| **使用难度** | 简单，只需添加检查点调用 | 较复杂，需要管理检查点 tensor | 简单，配置开关即可 |

### 选择指南

**Tensor Graph 校验失败** → 使用中间结果校验法（参考 [verify.md](reference/verify.md)）：
- 需要快速找到问题范围
- 只需要对比单次循环的数据，不需要对比多个循环迭代
- 不想修改 kernel 函数签名，保持代码简洁
- 复杂算子，循环较多，二分对比方法难以将数据搬运到循环外

**Pass 校验失败** → 使用 Pass 校验方法（参考 [pass.md](reference/pass.md)）：
- Tensor Graph 校验通过，但精度仍不对
- 需要定位是哪个 Pass 处理阶段引入的精度偏差
- 需要对比 Pass 间的 OP 节点变化

**Pass 校验通过但上板精度不对** → 使用二分对比方法（参考 [binary-search.md](reference/binary-search.md)）：
- 需要对比多个循环迭代的数据（如 idx=0,1,2,...）
- 需要 NPU 上板真实执行数据
- 保存的数据文件过大时，用二分法在内存中直接对比减少文件 IO 开销

## 快速开始

### 使用中间结果校验法（Tensor Graph 校验失败）

1. 阅读 [verify.md](reference/verify.md) 了解详细步骤
2. 在 kernel 中添加 `pypto.pass_verify_save()` 调用
3. 在 golden 中添加 `torch.save()` 调用
4. 运行测试生成数据
5. 使用对比工具分析结果

### 使用 Pass 校验方法（Tensor Graph 校验通过）

1. 阅读 [pass.md](reference/pass.md) 了解详细步骤
2. 配置 `verify_options` 开启 Pass 校验
3. 编译并运行测试
4. 分析验证结果，定位出错的 Pass

### 使用二分对比方法（Pass 校验通过但上板精度不对）

1. 阅读 [binary-search.md](reference/binary-search.md) 了解详细步骤
2. 修改 kernel 函数签名，添加检查点 tensor 参数
3. 修改 golden 函数，返回检查点数据
4. 修改测试函数，创建检查点 tensor 并对比
5. 运行测试并分析结果

## 核心原则

### 数据对齐原则

无论使用哪种方法，都必须确保：
- golden 和 kernel 的计算逻辑、切块方式、数据维度完全一致
- 如果实现不一致，改写 golden 函数使其与 kernel 一致
- 检查点的位置和顺序必须一一对应

### 检查点命名原则

- 使用有意义的名称，反映计算步骤
- 按计算顺序添加数字前缀（如 `1_after_matmul`, `2_after_softmax`）
- 确保命名约定一致，便于对比工具自动匹配

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
- 中间结果校验法: [verify.md](reference/verify.md)
- Pass 校验详解: [pass.md](reference/pass.md)
- 二分对比方法: [binary-search.md](reference/binary-search.md)
