---
name: pypto-binary-search-verify
description: PyPTO 算子查找精度问题调试技能。利用精度工具通过tensor数据比对快速定位算子精度问题，支持循环场景下的条件性数据保存。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题查找调试技能

利用精度工具快速定位 PyPTO 算子中导致精度问题的具体 op。

## 核心原理

1. **数据匹配**：golden 和 kernel 函数的实现需对应，计算逻辑和数据切块方式需要完全一致才可以添加检查点
2. **kernel 保存**：在 kernel 函数中使用 `pypto.pass_verify_save()` 保存中间结果（循环场景使用 `cond=(idx == 0)`）
3. **golden 保存**：在 golden 函数中使用 `numpy.tofile()` 保存中间结果（循环场景使用 `if idx == 0:`）
4. **数据对比**：使用对比工具检查 kernel 的 `.data` 文件和 golden 的 `.bin` 文件
5. **二分查找**：结果相同→往后 dump，不同→往前 dump
6. **定位问题**：定位第一个计算结果不同的 op
7. **插入原则**：一次性添加的检查点不要多，从关键的节点开始，插入的检查点数据切片必须一致
8. **数据类型**：读取数据时根据 csv 文件中的 dtype 自动判断数据类型（BF16/FP32 等）

## 核心原则

### 原则 1：插入检查点（kernel 函数）

```python
# 必须启用验证选项
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": []
}

@pypto.frontend.jit(verify_options=verify_options)
def kernel(inputs, outputs):
    # 基础场景
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "checkpoint1_after_op1")

    # 循环场景：只保存 idx=0 的数据
    # 多层循环使用 cond=((idx1 == 0) * (idx2 == 0)) 中间用 * 连接
    for idx in range(batch_size):
        temp = pypto.compute_op(inputs[idx])
        pypto.pass_verify_save(temp, "checkpoint_idx_$idx", cond=(idx == 0), idx=0)
```

### 原则 2：保存 golden 中间结果

```python
def golden(inputs, outputs):
    # 基础场景
    temp1 = compute_op1(inputs[0])
    temp1.cpu().numpy().tofile("golden_checkpoint1_after_op1.bin")

    # 循环场景：只保存 idx=0 的数据
    for idx in range(batch_size):
        temp = compute_op(inputs[idx])
        if idx == 0:
            temp.cpu().numpy().tofile(f"golden_checkpoint_idx{idx}.bin")
```

**文件命名约定**：
- golden 文件： `golden_{checkpoint_name}.bin`
- jit 文件: `{checkpoint_name}_{number}.data` （自动生成）
- 名称必须匹配（去掉 `golden_` 前缀和数字后缀）
- 循环场景需在名称中包含 `idx` 信息

### 原则 3：二分查找策略

```
输入 [op1] [op2] [op3] ... [opN] 输出
  ↑                              ↑
正确                          不正确

1. 在中间位置插入输出点，对比 golden
2. 结果相同 → 问题在后面 → 往后二分
3. 结果不同 → 问题在前或此处 → 往前二分
4. 重复直到找到第一个结果不同的 op
```

## 易错点及修正方案

### 1. 执行路径问题

**问题**：Output 文件和bin文件生成在执行路径下，工具需要在正确的目录下执行

**修正**：
```bash
# 在算子代码所在目录执行对比工具
cd /path/to/operator
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v

# 或使用 -w 参数指定工作目录
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w /path/to/operator -v
```

### 2. 数据类型读取问题

**问题**：读取数据时要根据保存的类型读取（BF16/FP32/INT32 等）

**修正**：对比工具会根据 CSV 文件中的 dtype 自动判断数据类型：
- dtype=8: BF16 格式（2字节），转换为 FP32 进行对比
- dtype=7: FP32 格式（4字节），直接读取、

### 3. 检查点插入位置问题

**问题**：golden 和 kernel 的计算步骤不一致

**原因**：kernel 分步计算并插入多个检查点，golden 合并了某些步骤

**原则**：检查点位置要一一对应
- 如果 kernel 在 A→B→C 三个步骤后都插入检查点，golden 也需要在对应步骤保存
- 不能因为某些步骤可以合并计算就跳过中间检查点
- 保持两边计算逻辑和保存时机完全一致

### 4. 切块计算问题

**问题**：kernel 可能切块计算（如 batch 循环），golden 可能一次性计算

**原则**：保持保存的数据维度一致

**两种解决思路**：
1. **golden 保存对应的切片数据**（推荐）
   - golden 一次性计算完整数据
   - 只保存与 kernel 对应的切片（如 `result[0:tile_b]`，根据循环层数修改）
   - 数据范围与 kernel 的 `idx=0` 切片完全一致

2. **kernel 在循环外保存完整数据**
   - kernel 在循环外定义完整大小的 tensor
   - 循环内使用 assemble 或赋值填充数据
   - 循环结束后保存完整 tensor
   - golden 一次性计算完整数据

**选择依据**：
- 如果切块是为了性能优化（分批处理），用方案 1（推荐）
- 如果需要验证完整输出的正确性，用方案 2

### 5. 精度标准问题

**问题**：不同数据类型需要不同的容差标准

**修正**：对比工具根据数据类型自动设置容差：

| 数据类型 | dtype | rtol | atol |
|---------|-------|------|------|
| BF16 | 8 | 0.05 | 0.005 |
| FP32 | 7 | 1e-5 | 1e-5 |
| FP16 | 6 | 1e-3 | 1e-3 |
| INT32 | 3 | 1e-5 | 1e-5 |

### 6. 对比逻辑问题

**问题**：只看最大差异容易误判，应统计不匹配率

**修正**：对比工具使用 `np.isclose` 统计不匹配个数：
- 判断条件：不匹配个数 < 总数 * max(rtol, atol)

## 完整工作流程

### 步骤 1：插入检查点

在 jit 和 golden 函数中插入对应的检查点（参考原则 1 和 2）。

**循环场景关键点**：
- 使用 `cond=(idx == 0)` 只保存一批数据
- 确保 kernel 和 golden 保存相同的 idx 数据
- 在检查点名称中包含 idx 信息

### 步骤 2：运行测试生成数据

```bash
python3 test_operator.py
```

### 步骤 3：对比检查点

使用通用对比工具或手动对比：

```bash
# 使用通用工具（推荐）
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

### 步骤 4：继续二分

根据对比结果：
- **匹配** → 问题在后面，在后面插入新检查点
- **不匹配** → 问题在前或此处，在前面插入新检查点

### 步骤 5：定位并修复

重复步骤 1-4，直到定位到具体的 op，然后修复问题。

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤：
```python
# ✓ 推荐
pypto.pass_verify_save(sij, "checkpoint1_after_qk_matmul")
pypto.pass_verify_save(sij, "checkpoint1_qk_idx$idx", cond=(idx == 0))

# ✗ 不推荐
pypto.pass_verify_save(sij, "temp1")
```

### 2. 循环场景处理

**关键要点**：
- 使用 `cond=(idx == 0)` 确保只保存一批数据
- kernel 和 golden 必须保存相同的 idx 数据
- 避免生成过多文件

### 3. 渐进式二分

```
第1轮：输入 → 中间 → 输出（3个检查点）
  ↓ 发现中间不匹配
第2轮：在中间位置前后插入检查点（5个检查点）
  ↓ 继续缩小范围
第3轮：在问题范围内插入更多检查点
  ↓
定位到具体 op
```

### 4. 清理调试代码

修复问题后：
```bash
# 移除调试文件
rm -f golden_*.bin
rm -rf output/output_*

# 移除调试代码
# - 删除 pypto.pass_verify_save() 调用
# - 删除 verify_options 参数
# - 删除 golden 中的 tofile() 调用
```

## 常见问题

### Q1: kernel 和 golden 保存的数据不一致

**原因**：kernel 保存 idx=0，但 golden 保存了其他 idx / golden 没有切块计算

**解决**：确保两者使用相同的条件，参考"易错点 4：切块计算问题"

### Q2: 找不到检查点文件

**检查**：
- jit 代码中是否使用了 `pypto.pass_verify_save()`
- 是否设置了 `verify_options={"enable_pass_verify": True}`
- 文件命名是否符合约定
- 是否在正确的目录下执行对比工具

## 通用对比工具

本技能提供了通用对比脚本，自动完成检查点扫描和对比：

```bash
# 自动检测并对比所有检查点
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py

# 列出所有检查点
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py --list

# 显示详细对比
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py --verbose

# 指定工作目录
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w /path/to/operator -v
```

工具功能：
- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 根据数据类型自动设置容差标准
- ✓ 统计不匹配率而非只看最大差异
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比

## 检查清单

使用二分查找调试时，确保：

- [ ] **步骤 1**：插入检查点
  - [ ] 设置 `verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True, "pass_verify_pass_filter": []}`
  - [ ] kernel 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] golden 函数中使用 `numpy.tofile()` 保存中间结果
  - [ ] 循环场景使用 `cond=(idx == 0)` 和 `if idx == 0:`
  - [ ] 文件命名遵循约定
  - [ ] 检查点插入位置要一一对应（避免连乘 vs 分步）
  - [ ] 切块计算要保持一致（golden 切块或 kernel assemble）
- [ ] **步骤 2**：运行测试生成数据
- [ ] **步骤 3**：对比检查点
  - [ ] 在正确目录下使用通用工具
  - [ ] 查看对比结果和不匹配率
- [ ] **步骤 4**：继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
- [ ] **步骤 5**：定位并修复
  - [ ] 定位到具体的 op
  - [ ] 修复问题
  - [ ] 重新验证
  - [ ] 清理调试代码

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
