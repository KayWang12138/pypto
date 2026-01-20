# Verifier Python Bindings

本文档介绍如何在Python中使用PyPTO的IR Verifier功能。

## 概述

Verifier提供了一套灵活的验证框架，用于验证TileValue是否满足特定的规则。它支持：
- 注册自定义验证规则
- 单个规则验证
- 批量规则验证
- 规则管理（添加、删除、清空）

## 类和枚举

### VerifyStatus (枚举)
表示验证状态：
- `ir.VerifyStatus.PASS`: 验证通过
- `ir.VerifyStatus.FAIL`: 验证失败

### VerifyResult (类)
验证结果对象：
- `status`: VerifyStatus枚举值
- `error_msg`: 错误消息（字符串）
- `passed()`: 返回布尔值，表示是否通过验证

### Verifier (类)
验证器主类，用于管理和执行验证规则。

## 基本用法

### 创建Verifier实例

```python
from pypto.pypto_impl import ir

verifier = ir.Verifier()
```

### 注册验证规则

验证规则是一个接受TileValue参数的函数，可以返回以下两种格式：

1. **元组格式**: `(bool, str)` - (是否通过, 错误消息)

```python
def check_square_shape(tile):
    shape = tile.shape
    if len(shape) == 2 and shape[0] == shape[1]:
        return (True, "")
    return (False, f"Shape is not square: {shape}")

verifier.register_rule("shape_square", check_square_shape)
```

2. **VerifyResult格式**: 返回`ir.VerifyResult`对象

```python
def check_dtype_float32(tile):
    dtype = tile.type.dtype
    if dtype == ir.DataType.float32:
        return ir.VerifyResult(ir.VerifyStatus.PASS, "")
    return ir.VerifyResult(ir.VerifyStatus.FAIL, f"Expected float32, got {dtype}")

verifier.register_rule("dtype_float32", check_dtype_float32)
```

### 验证Tile

**验证单个规则**:
```python
result = verifier.verify_rule("shape_square", test_tile)
if result.passed():
    print("验证通过")
else:
    print(f"验证失败: {result.error_msg}")
```

**验证所有规则**:
```python
result = verifier.verify_all_rules(test_tile)
if not result.passed():
    print(f"验证失败: {result.error_msg}")
```

### 规则管理

```python
# 检查规则是否存在
if verifier.has_rule("shape_square"):
    print("规则存在")

# 获取所有规则名称
rule_names = verifier.get_rule_names()
print(f"已注册规则: {rule_names}")

# 获取规则数量
count = verifier.get_rule_count()
print(f"规则数量: {count}")

# 删除规则
verifier.remove_rule("shape_square")

# 清空所有规则
verifier.clear_rules()

# 检查是否为空
if verifier.is_empty():
    print("没有注册任何规则")
```

## 完整示例

```python
from pypto.pypto_impl import ir

# 创建IR构建器和上下文
builder = ir.IrBuilder()
ctx = ir.IrBuilderContext()

# 创建函数
sig = ir.FunctionSignature()
func = builder.create_function("test_func", ir.FunctionKind.DataFlow, sig)
builder.enter_function(ctx, func)

# 创建测试tile
tile_shape = [256, 256]
test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

# 创建验证器
verifier = ir.Verifier()

# 注册多个验证规则
def check_2d(tile):
    return (len(tile.shape) == 2, "Must be 2D")

def check_square(tile):
    shape = tile.shape
    return (shape[0] == shape[1], "Must be square")

def check_power_of_2(tile):
    shape = tile.shape
    size = shape[0]
    if size & (size - 1) != 0:
        return (False, f"Size {size} is not a power of 2")
    return (True, "")

verifier.register_rule("check_2d", check_2d)
verifier.register_rule("check_square", check_square)
verifier.register_rule("check_power_of_2", check_power_of_2)

# 执行验证
result = verifier.verify_all_rules(test_tile)
if result.passed():
    print("所有验证通过!")
else:
    print(f"验证失败: {result.error_msg}")

# 清理
ctx.pop_scope()
```

## 高级用法

### 复杂验证规则

```python
def validate_matmul_tile(tile):
    """验证矩阵乘法tile的要求"""
    shape = tile.shape
    dtype = tile.type.dtype

    # 检查维度
    if len(shape) != 2:
        return ir.VerifyResult(ir.VerifyStatus.FAIL, "Matmul tile must be 2D")

    # 检查数据类型
    if dtype not in [ir.DataType.float16, ir.DataType.float32]:
        return ir.VerifyResult(ir.VerifyStatus.FAIL, f"Unsupported dtype: {dtype}")

    # 检查尺寸对齐
    if shape[0] % 16 != 0 or shape[1] % 16 != 0:
        return ir.VerifyResult(ir.VerifyStatus.FAIL, "Shape must be aligned to 16")

    return ir.VerifyResult(ir.VerifyStatus.PASS, "")

verifier.register_rule("matmul_requirements", validate_matmul_tile)
```

### 访问Tile属性

在验证规则中，可以访问Tile的各种属性：

```python
def inspect_tile(tile):
    # 访问shape
    shape = tile.shape

    # 访问数据类型
    dtype = tile.type.dtype

    # 访问strides
    strides = tile.strides

    # 访问offset
    offset = tile.offset

    # 访问memory
    memory = tile.memory

    print(f"Shape: {shape}, DType: {dtype}")
    return (True, "")
```

## 测试

运行测试文件：
```bash
cd /data/y00955915/pypto-1120
python python/tests/ut/ir/test_verifier.py
```

或使用pytest：
```bash
pytest python/tests/ut/ir/test_verifier.py -v
```

## 注意事项

1. 验证规则函数必须返回`VerifyResult`或`(bool, str)`元组
2. 验证规则中的Python异常会被捕获并转换为验证失败
3. 使用`verify_all_rules`时，如果没有注册任何规则，会返回失败状态
4. 规则名称是唯一的，重复注册会覆盖之前的规则
5. 验证规则函数会收到TileValue的共享指针，不要修改tile的状态

## 相关文件

- Python绑定实现: `/data/y00955915/pypto-1120/python/src/bindings/ir.cpp`
- C++ Verifier头文件: `/data/y00955915/pypto-1120/framework/include/ir/verifier/verifier.h`
- 测试文件: `/data/y00955915/pypto-1120/python/tests/ut/ir/test_verifier.py`
