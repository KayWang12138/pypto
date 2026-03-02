# PyPTO 算子开发常见问题总结

本文档总结了 PyPTO 算子开发过程中遇到的常见问题和解决方案，帮助开发者避免重复踩坑。

---

## 一、关键技术问题

### 1. BFloat16 转 NumPy 问题 ⭐⭐⭐⭐⭐

**错误现象**:
```python
TypeError: Got unsupported ScalarType BFloat16
```

**错误代码**:
```python
output_np = output.cpu().numpy()  # ❌ 错误
golden_np = golden.cpu().numpy()  # ❌ 错误
```

**正确做法**:
```python
output_np = output.cpu().float().numpy()  # ✅ 正确
golden_np = golden.cpu().float().numpy()  # ✅ 正确
```

**经验教训**:
- PyTorch 的 bfloat16 tensor 不能直接转换为 numpy
- 必须先 `.float()` 转为 float32，再 `.numpy()`
- **所有使用 bfloat16 的测试代码都要注意这个问题**

---

### 2. 使用 PyTorch 作为 Golden 函数 ⭐⭐⭐⭐⭐

**问题描述**:
- 使用 NumPy 实现 golden 函数时，bfloat16 数据类型转换不够准确
- NumPy 对 bfloat16 的支持有限，可能导致精度损失

**错误做法**:
```python
# ❌ 使用 NumPy 实现 golden
def golden_matmul_add(a, b, c):
    return np.dot(a, b.T) + c

# 验证时需要多次类型转换
expected = torch.from_numpy(golden_matmul_add(a.float().cpu().numpy(), ...))
```

**正确做法**:
```python
# ✅ 使用 PyTorch 实现 golden
def matmul_add_golden(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of matmul_add."""
    b_transposed = b.transpose(-2, -1)
    matmul_result = torch.matmul(a, b_transposed)
    result = matmul_result + c
    return result

# 验证时直接使用，无需类型转换
expected = matmul_add_golden(a, b, c)
```

**经验教训**:
- **优先使用 PyTorch 作为 golden 函数**，避免数据类型转换
- PyTorch 对 bfloat16 有原生支持，计算更准确
- 可以保留 NumPy 版本作为备选验证，但优先使用 PyTorch 版本
- 这样可以避免 `float().cpu().numpy()` 的多次转换，减少精度损失

---

### 2. 环境变量设置问题 ⭐⭐⭐⭐

**错误现象**:
```
If no NPU environment is available
```

**正确做法**:
```bash
# 1. 先设置 TILE_FWK_DEVICE_ID=0
export TILE_FWK_DEVICE_ID=0

# 2. 如果失败，检查可用 NPU 设备
npu-smi info

# 3. 根据输出设置正确的 device ID
export TILE_FWK_DEVICE_ID=<正确的ID>
```

**经验教训**:
- **第一步就要设置环境变量**，不要等到运行时才想起来
- 如果 TILE_FWK_DEVICE_ID=0 失败，再用 `npu-smi info` 检查
- 参考 AGENTS.md 第 226-234 行的环境设置说明

---

## 二、API 使用问题

### 3. 动态轴定义位置 ⭐⭐⭐⭐

**错误做法**:
```python
def matmul_add_op(...):
    @pypto.frontend.jit(...)
    def kernel(...):
        m = pypto.frontend.dynamic("m")  # ❌ 在函数内部定义
        ...
```

**正确做法**:
```python
def matmul_add_op(...):
    m = pypto.frontend.dynamic("m")  # ✅ 在 jit 函数外部定义
    
    @pypto.frontend.jit(...)
    def kernel(
        a: pypto.Tensor((m, k), pypto.DT_BF16),  # 在类型注解中使用
        ...
    ):
        ...
```

**经验教训**:
- 动态轴必须在模块级别或函数级别定义，**不能在 jit 函数内部定义**
- 定义后才能在 Tensor 的类型注解中使用
- 参考 `docs/api/pypto-frontend-dynamic.md` 的说明

---

### 4. matmul 转置参数理解错误 ⭐⭐⭐

**需求**: `y = a @ b^T + c`

**错误理解**:
```python
# ❌ 错误：以为需要先 transpose b
b_t = pypto.transpose(b, 0, 1)
result = pypto.matmul(a, b_t, pypto.DT_BF16)
```

**正确做法**:
```python
# ✅ 正确：直接使用 b_trans 参数
result = pypto.matmul(a, b, pypto.DT_BF16, b_trans=True)
```

**经验教训**:
- **优先使用 API 内置的参数**，不要手动实现已提供的功能
- matmul 的 `b_trans=True` 参数就是用来处理 `b^T` 的
- 参考 `docs/api/operation/pypto-matmul.md` 第 32-33 行

---

### 5. Tile Shape 设置遗漏 ⭐⭐⭐

**错误现象**:
```
编译或运行时警告/错误
```

**正确做法**:
```python
# matmul 前设置 cube tile shapes
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])

# add 前设置 vec tile shapes
pypto.set_vec_tile_shapes(1, 64, 1, 64)
```

**经验教训**:
- matmul 操作**必须**先调用 `pypto.set_cube_tile_shapes`
- add 等 vector 操作需要 `pypto.set_vec_tile_shapes`
- 参考 API 文档的 "约束说明" 章节

---

## 三、测试和验证问题

### 6. 精度标准设置不当 ⭐⭐⭐

**错误做法**:
```python
# ❌ 使用 float32 的精度标准
assert_allclose(output, golden, rtol=1e-5, atol=1e-8)
```

**正确做法**:
```python
# ✅ 使用适合 bfloat16 的精度标准
atol = 0.0001      # 绝对误差
rtol = 0.0078125   # 相对误差 (1/128)
assert_allclose(output, golden, rtol=rtol, atol=atol)
```

**经验教训**:
- **bfloat16 的精度远低于 float32**
- 需求文档明确给出精度标准：atol=0.0001, rtol=0.0078125
- 参考 `docs/tutorials/debug/precision.md` 的说明

---

### 7. 测试用例设计不全 ⭐⭐⭐

**遗漏的测试**:
- 只测试了随机数据，没有测试边界情况
- 没有测试零值、全1等特殊情况

**正确做法**:
```python
# Level 2: 边界情况测试
test_cases.append(("zeros", ...))  # 零值
test_cases.append(("ones", ...))   # 全1
test_cases.append(("extreme", ...))  # 极值
```

**经验教训**:
- **必须设计多级测试用例**（Level 0~3）
- 参考 AGENTS.md 第 71-80 行的测试分级说明
- 边界情况是发现问题的关键

---

## 四、开发流程问题

### 8. 没有先查看已有实现 ⭐⭐⭐⭐

**问题**:
- 从零开始实现，花费时间摸索
- 没有发现项目中已有类似实现

**正确做法**:
```bash
# 1. 先搜索是否已有类似实现
find . -name "*相关关键词*" -type f

# 2. 参考已有实现
cat custom/existing_implementation.py

# 3. 参考官方示例
ls examples/
```

**经验教训**:
- **开发前先搜索已有代码**，避免重复造轮子
- 参考 `examples/` 中的官方示例
- 参考 `AGENTS.md` 的 "原则 2：基于官方文档实现"

---

### 9. 编译时间过长 ⭐⭐

**问题**:
- 使用 `loop_unroll` 后编译时间增加
- 没有心理准备，以为卡住了

**正确做法**:
```python
# 1. 理解这是正常现象
# loop_unroll 生成多个展开路径，编译时间必然增加

# 2. 可以限制展开因子减少编译时间
unroll_list=[1, 2, 4, 8]  # 移除 16，减少一个路径
```

**经验教训**:
- **loop_unroll 的编译代价是正常的**
- 如果编译超过 10 分钟，可以中断检查代码
- 参考 AGENTS.md 第 134 行的提示

---

## 五、文档和代码规范问题

### 10. 代码注释不充分 ⭐⭐

**问题**:
- 初始版本注释较少
- 没有说明关键参数的含义

**正确做法**:
```python
# ✅ 添加详细注释
# loop_unroll 优化：自动生成多个展开路径（1, 2, 4, 8, 16）
# 运行时根据动态轴 m 的大小自动选择最优路径
for m_offset, tile_m in pypto.loop_unroll(
    0, m, 1,
    name="LOOP_M_TILE",
    idx_name="m_idx",
    unroll_list=[1, 2, 4, 8, 16]
):
    ...
```

**经验教训**:
- **关键参数和优化技术必须注释**
- 参考 `examples/` 中的官方示例的注释风格
- 便于后续维护和理解

---

## 六、开发前 Checklist

### ✅ 环境准备
- [ ] 设置 `export TILE_FWK_DEVICE_ID=0`
- [ ] 检查 `npu-smi info` 确认设备可用
- [ ] 设置 `export PTO_TILE_LIB_CODE_PATH=...`

### ✅ 资料查阅
- [ ] 搜索 `find . -name "*相关关键词*"` 查找已有实现
- [ ] 阅读 `docs/api/` 中的相关 API 文档
- [ ] 参考 `examples/` 中的类似示例
- [ ] 阅读 `AGENTS.md` 的开发规范

### ✅ 代码实现
- [ ] 动态轴在 jit 函数**外部**定义
- [ ] BFloat16 转 numpy 时先 `.float()`
- [ ] matmul 前设置 `set_cube_tile_shapes`
- [ ] add 前设置 `set_vec_tile_shapes`
- [ ] 优先使用 API 内置参数（如 `b_trans`）

### ✅ 测试验证
- [ ] 设计 Level 0~3 多级测试用例
- [ ] 包含边界情况（零值、全1、极值）
- [ ] 使用正确的精度标准（bfloat16: atol=0.0001, rtol=0.0078125）
- [ ] 所有测试级别通过

### ✅ 文档编写
- [ ] 编写 README.md（算子概述、使用方法、测试结果）
- [ ] 编写性能分析报告
- [ ] 更新 plan 文档
- [ ] 关键代码添加注释

---

## 七、快速参考表

| 问题类型 | 常见错误 | 正确做法 | 优先级 |
|---------|---------|---------|--------|
| 数据类型 | `bfloat16.numpy()` | `bfloat16.float().numpy()` | ⭐⭐⭐⭐⭐ |
| 环境变量 | 未设置 TILE_FWK_DEVICE_ID | 先 `export TILE_FWK_DEVICE_ID=0` | ⭐⭐⭐⭐⭐ |
| 动态轴 | 在 jit 内定义 | 在 jit 外定义 | ⭐⭐⭐⭐ |
| API 使用 | 手动 transpose | 使用 `b_trans=True` | ⭐⭐⭐ |
| Tile Shape | 未设置 | matmul 前调用 `set_cube_tile_shapes` | ⭐⭐⭐ |
| 精度标准 | 使用 float32 标准 | 使用 bfloat16 标准 | ⭐⭐⭐ |
| 测试用例 | 只测试随机数据 | 包含边界情况 | ⭐⭐⭐ |
| 资料查阅 | 从零实现 | 先搜索已有代码 | ⭐⭐⭐⭐ |
| 编译时间 | 以为卡住 | 理解 loop_unroll 代价 | ⭐⭐ |
| 代码注释 | 注释不足 | 关键参数添加注释 | ⭐⭐ |

---

## 八、核心经验总结

1. **先查资料再动手**：搜索已有实现，参考官方示例
2. **环境变量第一步**：先设置 TILE_FWK_DEVICE_ID
3. **BFloat16 特殊处理**：转 numpy 前必须 `.float()`
4. **动态轴在函数外**：不要在 jit 函数内定义
5. **使用 API 内置功能**：不要手动实现已有功能
6. **必须设置 Tile Shape**：matmul 前设置 cube tile shapes
7. **精度标准要合理**：根据数据类型设置
8. **测试用例要全面**：Level 0~3 + 边界情况
9. **编译时间会增加**：loop_unroll 的正常代价
10. **注释要充分**：便于理解和维护

**遵循这些经验，可以避免 95% 的常见错误！**

---

## 九、相关参考资料

- [AGENTS.md](../../../AGENTS.md) - 开发规范和工作流程
- [docs/api/](../../../docs/api/) - API 文档
- [examples/](../../../examples/) - 官方示例代码
- [docs/tutorials/debug/precision.md](../../../docs/tutorials/debug/precision.md) - 精度调试指南
