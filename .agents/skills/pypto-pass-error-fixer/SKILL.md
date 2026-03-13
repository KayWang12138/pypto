---
name: pypto-pass-error-fixer
description: PyPTO 自动修复与验证技能。根据 pass 模块错误分析结果，自动定位问题代码，尝试修复错误，并验证修复结果。当需要自动修复 PyPTO 错误时使用此技能。
tag: [PyPTO, 自动修复, 错误验证]
---

# PyPTO 自动修复与验证技能

本技能提供 PyPTO 错误的自动修复能力，定位问题代码并尝试修复。

## 功能概述

- 根据错误分析结果定位问题代码
- 应用自动修复策略
- 验证修复结果
- 生成修复报告
- 支持回滚操作

## 工作流程

### 步骤 1：问题定位

**定位策略**：

1. **基于堆栈跟踪定位**
   - 从堆栈中提取文件路径和行号
   - 定位到具体的错误代码行

2. **基于 pass 模块定位**
   - 在 `framework/passes/` 目录中搜索 pass 模块
   - 查找 pass 模块的实现代码

3. **基于错误模式定位**
   - 使用 grep 搜索错误消息
   - 在相关文件中查找匹配的代码

### 步骤 2：修复策略应用

**修复策略库**：

| 错误类型 | 修复策略 | 自动修复级别 |
|---------|---------|-------------|
| **ShapeMismatch** | 调整输入shape、添加reshape | 半自动 |
| **TypeMismatch** | 添加类型转换、调整dtype | 自动 |
| **InvalidParameter** | 修正参数值、使用默认值 | 自动 |
| **MissingParameter** | 添加缺失参数 | 半自动 |
| **UnsupportedOp** | 提供替代实现 | 手动 |
| **OutOfMemory** | 减少batch size、优化内存 | 半自动 |
| **DeviceNotAvailable** | 设置正确的device_id | 自动 |

### 步骤 3：代码修复

**修复操作类型**：

1. **参数修复**
```python
# 修复前
result = pypto.matmul(x, y)

# 修复后
result = pypto.matmul(x, y, transpose_b=True)
```

2. **类型转换修复**
```python
# 修复前
result = pypto.add(x, y)

# 修复后
x = x.to(pypto.float32)
y = y.to(pypto.float32)
result = pypto.add(x, y)
```

3. **Shape 调整修复**
```python
# 修复前
result = pypto.reshape(x, [batch, -1])

# 修复后
x = x.reshape([batch, seq_len, hidden_dim])
result = pypto.reshape(x, [batch, -1])
```

4. **内存优化修复**
```python
# 修复前
x = pypto.zeros([10000, 10000])

# 修复后
x = pypto.zeros([batch_size, seq_len, hidden_dim])
```

### 步骤 4：修复验证

**验证步骤**：

1. **语法验证**
```bash
python3 -m py_compile file.py
```

2. **编译验证**
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

3. **运行验证**
```bash
python3 custom/operator/operator.py --run_mode npu
```

4. **精度验证**
```bash
python3 custom/operator/operator.py --verify
```

### 步骤 5：生成修复报告

**报告格式**（JSON）：
```json
{
  "fix_id": "unique_fix_id",
  "error_id": "related_error_id",
  "status": "success",
  "fixes_applied": [
    {
      "file": "path/to/file.py",
      "line": 123,
      "original_code": "old code",
      "fixed_code": "new code",
      "fix_type": "parameter_fix"
    }
  ],
  "verification": {
    "syntax_check": "passed",
    "compilation_check": "passed",
    "runtime_check": "passed",
    "accuracy_check": "passed"
  },
  "rollback_available": true,
  "warnings": []
}
```

## 使用方法

### 方式 1：自动修复单个错误

```python
from pypto_auto_fixer import auto_fix

# 从错误分析得到的结果
analysis_result = {
  "pass_module": "ShapeInference",
  "error_type": "ShapeMismatch",
  "related_code": [...]
}

fix_result = auto_fix(analysis_result)
print(fix_result)
```

### 方式 2：交互式修复

```python
from pypto_auto_fixer import interactive_fix

# 显示修复建议，等待用户确认
fix_result = interactive_fix(analysis_result)
```

### 方式 3：批量修复

```python
from pypto_auto_fixer import batch_fix

analyses = [analysis1, analysis2, analysis3]
fix_results = batch_fix(analyses)
```

### 方式 4：回滚修复

```python
from pypto_auto_fixer import rollback_fix

# 回滚指定的修复
rollback_fix(fix_id)
```

## 修复策略详解

### 策略 1：参数自动修正

**适用场景**：参数值错误、参数类型错误

**修复逻辑**：
1. 从 API 文档中获取正确的参数信息
2. 比较当前参数值与期望值
3. 自动修正参数值

**示例**：
```python
# 错误：transpose_b 参数类型错误
# 修复前
result = pypto.matmul(x, y, transpose_b="true")

# 修复后
result = pypto.matmul(x, y, transpose_b=True)
```

### 策略 2：类型自动转换

**适用场景**：dtype 不匹配

**修复逻辑**：
1. 识别期望的 dtype
2. 在操作前添加类型转换
3. 确保所有输入 dtype 一致

**示例**：
```python
# 错误：dtype 不匹配
# 修复前
x = pypto.tensor([1, 2, 3], dtype=pypto.int32)
y = pypto.tensor([4.0, 5.0, 6.0], dtype=pypto.float32)
result = pypto.add(x, y)

# 修复后
x = x.to(pypto.float32)
result = pypto.add(x, y)
```

### 策略 3：Shape 自动调整

**适用场景**：Shape 不兼容

**修复逻辑**：
1. 分析期望的 shape
2. 添加 reshape 或 view 操作
3. 保持数据的语义正确性

**示例**：
```python
# 错误：shape 不匹配
# 修复前
x = pypto.randn([32, 128])
y = pypto.randn([32, 64])
result = pypto.add(x, y)

# 修复后
y = y.reshape([32, 128])
result = pypto.add(x, y)
```

### 策略 4：内存优化

**适用场景**：内存不足

**修复逻辑**：
1. 识别大张量分配
2. 减少不必要的中间张量
3. 使用 inplace 操作

**示例**：
```python
# 错误：内存不足
# 修复前
x = pypto.randn([10000, 10000])
y = pypto.randn([10000, 10000])
z = x + y

# 修复后
batch_size = 32
x = pypto.randn([batch_size, 128])
y = pypto.randn([batch_size, 128])
z = x + y
```

## 安全措施

### 1. 备份机制

**修复前自动备份**：
```bash
# 创建备份
cp file.py file.py.backup
```

### 2. 原子操作

**确保修复的原子性**：
- 要么全部成功，要么全部回滚
- 使用事务模式进行多个修复

### 3. 验证机制

**修复后必须验证**：
- 语法检查
- 编译检查
- 运行检查
- 精度检查

### 4. 回滚机制

**修复失败时自动回滚**：
```python
try:
    apply_fixes(fixes)
    verify_fixes()
except Exception as e:
    rollback_fixes()
    raise e
```

## 验证标准

### Level 0：语法验证
```bash
python3 -m py_compile file.py
```

### Level 1：编译验证
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

### Level 2：运行验证
```bash
export TILE_FWK_DEVICE_ID=0
python3 custom/operator/operator.py --run_mode npu
```

### Level 3：精度验证
```bash
python3 custom/operator/operator.py --verify --rtol=1e-3 --atol=1e-5
```

## 常见问题

### Q1: 自动修复失败怎么办？

**A**:
1. 查看修复报告中的错误信息
2. 检查备份是否完整
3. 手动回滚到修复前状态
4. 根据错误信息手动修复

### Q2: 如何禁用自动修复？

**A**: 设置环境变量：
```bash
export PYPTO_AUTO_FIX=false
```

### Q3: 如何查看修复历史？

**A**: 查看修复日志：
```bash
cat .pypto_fix_history.log
```

### Q4: 修复后精度不通过怎么办？

**A**:
1. 检查修复是否改变了计算逻辑
2. 使用更严格的精度标准验证
3. 回滚修复并手动调整

## 参考文件

| 文件 | 内容 |
|------|----------|
| `scripts/auto_fix.py` | 自动修复脚本 |
| `scripts/verify_fix.py` | 修复验证脚本 |
| `tests/test_auto_fixer.py` | 单元测试 |
| `knowledge_base/fix_strategies.json` | 修复策略知识库 |
