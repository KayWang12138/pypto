# PyPTO 错误码处理指南

本技能文档提供了 PyPTO 开发中常见错误码的详细说明，包括错误原因、解决办法和示例代码。

## 目录

- [通用错误码](#通用错误码)
- [配置错误码](#配置错误码)
- [Tensor 错误码](#tensor-错误码)
- [Function 错误码](#function-错误码)
- [文件操作错误码](#文件操作错误码)
- [TensorSlot 错误码](#tensorslot-错误码)
- [符号操作错误码](#符号操作错误码)

---

## 通用错误码

### DUPLICATE_OPERATION (0x10003)

**错误描述：** 重复操作错误

**出现原因：**
- 尝试添加已经存在的操作
- 操作名称或标识符重复

**解决办法：**
- 检查操作是否已存在
- 使用唯一的操作名称
- 在添加操作前进行去重检查

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def duplicate_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.add(x, x)  # 可能导致重复操作
    return op1

# 正确示例
@pypto.frontend.jit
def correct_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.mul(x, 2)  # 使用不同的操作
    return op1
```

### INVALID_OPERATION_OPERAND (0x10005)

**错误描述：** 无效的操作数

**出现原因：**
- 操作数类型不匹配
- 操作数为空或未初始化
- 操作数数量不符合要求

**解决办法：**
- 检查操作数类型是否正确
- 确保操作数已正确初始化
- 验证操作数数量

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def invalid_operand_example(x):
    y = None
    return pypto.add(x, y)  # y 未初始化

# 正确示例
@pypto.frontend.jit
def correct_operand_example(x):
    y = pypto.ones_like(x)
    return pypto.add(x, y)  # y 已正确初始化
```

### INVALID_FUNCTION_TYPE (0x10006)

**错误描述：** 无效的函数类型

**出现原因：**
- 函数类型不匹配
- 期望的函数类型与实际类型不符

**解决办法：**
- 检查函数声明类型
- 使用正确的函数装饰器
- 确保函数返回类型正确

**示例：**

```python
# 错误示例
def invalid_function_type(x):
    return x + 1  # 缺少 @pypto.frontend.jit 装饰器

# 正确示例
@pypto.frontend.jit
def correct_function_type(x):
    return x + 1  # 使用正确的装饰器
```

### TENSOR_CONSISTENCY_ERROR (0x10007)

**错误描述：** Tensor 一致性错误

**出现原因：**
- Tensor 形状或类型不一致
- 操作过程中 Tensor 状态发生变化

**解决办法：**
- 检查 Tensor 形状和类型
- 确保操作前后 Tensor 一致一致
- 使用类型转换函数

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def consistency_error_example(x, y):
    # x 和 y 形状不一致时可能导致错误
    return pypto.add(x, y)

# 正确示例
@pypto.frontend.jit
def correct_consistency_example(x, y):
    # 确保形状一致
    if x.shape != y.shape:
        y = pypto.reshape(y, x.shape)
    return pypto.add(x, y)
```

### OPERATION_NOT_FOUND (0x10008)

**错误描述：** 操作未找到

**出现原因：**
- 引用了不存在的操作
- 操作名称拼写错误
- 操作未正确注册

**解决办法：**
- 检查操作名称拼写
- 确保操作已注册
- 使用正确的操作 API

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def operation_not_found_example(x):
    return pypto.nonexistent_op(x)  # 操作不存在

# 正确示例
@pypto.frontend.jit
def correct_operation_example(x):
    return pypto.relu(x)  # 使用存在的操作
```

### SHAPE_MISMATCH (0x1000A)

**错误描述：** 形状不匹配

**出现原因：**
- 两个 Tensor 形状不匹配
- 在替换 tensor 时形状不一致
- 操作要求特定形状但未满足

**解决办法：**
- 检查 Tensor 形状
- 使用 reshape 或 broadcast 调整形状
- 确保操作前形状匹配
- 在替换操作时确保形状一致

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def shape_mismatch_example(x):
    # x 形状为 [4, 4]，尝试着与 [2, 2] 相加
    y = pypto.ones((2, 2), pypto.DT_FP32)
    return pypto.add(x, y)

# 正确示例
@pypto.frontend.jit
def correct_shape_example(x):
    # 调整形状以匹配
    y = pypto.ones_like(x)
    return pypto.add(x, y)
```

### PRODUCER_CONSUMER_ERROR (0x1000B)

**错误描述：** 生产者-消费者错误

**出现原因：**
- Tensor 的生产者和消费者关系不正确
- 数据流图中存在循环依赖

**解决办法：**
- 检查数据流图结构
- 确保生产者-消费者关系正确
- 避免循环依赖

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def producer_consumer_error_example(x):
    y = pypto.add(x, 1)
    z = pypto.mul(y, 2)
    # 错着地重新使用 y 导致生产者-消费者关系混乱
    w = pypto.add(y, z)
    return w

# 正确示例
@pypto.frontend.jit
def correct_producer_consumer_example(x):
    y = pypto.add(x, 1)
    z = pypto.mul(y, 2)
    # 正确的数据流
    w = pypto.add(x, z)
    return w
```

---

## 配置错误码

### CONFIG_INVALID_TYPE (0x11004)

**错误描述：** 配置类型无效

**出现原因：**
- 配置值类型与期望类型不匹配
- 配置参数类型错误

**解决办法：**
- 检查配置值类型
- 使用正确的数据类型
- 参考 API 文档中的配置类型要求

**示例：**

```python
# 错误示例
# 配置文件中
{
    "tile_size": "128"  # 应该是数字而不是字符串
}

# 正确示例
# 配置文件中
{
    "tile_size": 128  # 正确的数字类型
}
```

### CONFIG_FIELD_MISSING (0x11005)

**错误描述：** 配置字段缺失

**出现原因：**
- 必需的配置字段未提供
- 配置文件不完整

**解决办法：**
- 检查配置文件完整性
- 添加缺失的必需字段
- 参考 API 文档中的配置要求

**示例：**

```python
# 错误示例
# 配置文件中缺少必需字段
{
    "tile_size": 128
    # 缺少 "vec_tile_size" 字段
}

# 正确示例
# 配置文件中包含所有必需字段
{
    "tile_size": 128,
    "vec_tile_size": 64
}
```

### CONFIG_KEY_NOT_LOADED (0x11007)

**错误描述：** 配置键未加载

**出现原因：**
- 配置键不存在
- 配置文件未正确加载

**解决办法：**
- 检查配置键名称
- 确保配置文件已加载
- 使用正确的配置键

**示例：**

```python
# 错误示例
config = pypto.load_config("config.json")
value = config.get("nonexistent_key")  # 键不存在

# 正确示例
config = pypto.load_config("config.json")
value = config.get("tile_size")  # 使用存在的键
```

### CONFIG_VALUE_CONVERT_FAILED (0x11008)

**错误描述：** 配置值转换失败

**出现原因：**
- 配置值无法转换为目标类型
- 配置值格式不正确

**解决办法：**
- 检查配置值格式
- 使用正确的值类型
- 确保值可以转换

**示例：**

```python
# 错误示例
# 配置文件中
{
    "tile_size": "abc"  # 无法转换为数字
}

# 正确示例
# 配置文件中
{
    "tile_size": "128"  # 可以转换为数字
}
```

### CONFIG_VALUE_OVERFLOW (0x11009)

**错误描述：** 配置值溢出

**出现原因：**
- 配置值超出允许范围
- 数值过大或过小

**解决办法：**
- 检查配置值范围
- 使用合理的配置值
- 参考 API 文档中的值范围限制

**示例：**

```python
# 错误示例
# 配置文件中
{
    "tile_size": 999999999  # 值过大
}

# 正确示例
# 配置文件中
{
    "tile_size": 128  # 合理的值
}
```

---

## Tensor 错误码

### TENSOR_AXIS_OUT_OF_RANGE (0x12002)

**错误描述：** Tensor 轴超出范围

**出现原因：**
- 访问的轴索引超出 Tensor 维度
- 轴参数为负数或过大
- 在 reduce 操作中使用了无效的轴参数

**解决办法：**
- 检查 Tensor 维度数量
- 确保轴索引在 [0, dim-1] 范围内
- 使用正确的轴索引或使用默认值

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def axis_out_of_range_example(x):
    # x 形状为 [4, 4]，但尝试访问轴 2
    return pypto.sum(x, axis=2)  # 轴 2 超出范围（有效轴为 0, 1）

# 正确示例
@pypto.frontend.jit
def correct_axis_example(x):
    # 访问有效的轴 0 或 1
    return pypto.sum(x, axis=0)  # 轴 0 在范围内
```

### TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)

**错误描述：** Tensor 视图维度不匹配

**出现原因：**
- View 操作的维度与源 Tensor 不匹配
- 视图形状参数错误
- 在 cube 操作中，形状维度与 offset 维度不匹配

**解决办法：**
- 检查 View 操作的维度参数
- 确保视图形状与源 Tensor 兼容
- 使用正确的视图参数
- 确保形状维度与 offset 维度一致

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def view_dimension_mismatch_example(x):
    # x 形状为 [8, 8]，但视图形状为 [4, 4, 4]
    return pypto.view(x, [4, 4, 4])  # 维度不匹配

# 正确示例
@pypto.frontend.jit
def correct_view_example(x):
    # 视图形状与源 Tensor 兼容
    return pypto.view(x, [4, 4])  # 正确的视图形状
```

### TENSOR_VIEW_OFFSET_MISMATCH (0x12004)

**错误描述：** Tensor 视图偏移不匹配

**出现原因：**
- View 操作的偏移参数错误
- 偏移超出源 Tensor 范围

**解决办法：**
- 检查 View 操作的偏移参数
- 确保偏移在有效范围内
- 使用正确的偏移值

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def view_offset_mismatch_example(x):
    # x 形状为 [8, 8]，但偏移 [10, 10] 超出范围
    return pypto.view(x, [4, 4], offset=[10, 10])  # 偏移超出范围

# 正确示例
@pypto.frontend.jit
def correct_view_offset_example(x):
    # 偏移在有效范围内
    return pypto.view(x, [4, 4], offset=[2, 2])  # 正确的偏移
```

### TENSOR_SHAPE_OUT_OF_BOUNDS (0x12005)

**错误描述：** Tensor 形状越界

**出现原因：**
- Tensor 形状值超出允许范围
- 形状维度过多或过大

**解决办法：**
- 检查 Tensor 形状值
- 使用合理的形状参数
- 参考 API 文档中的形状限制

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def shape_out_of_bounds_example():
    # 形状过大
    x = pypto.tensor([100000, 100000], pypto.DT_FP32)
    return x

# 正确示例
@pypto.frontend.jit
def correct_shape_example():
    # 合理的形状
    x = pypto.tensor([1024, 1024], pypto.DT_FP32)
    return x
```

### TENSOR_INVALID_SHAPE (0x12006)

**错误描述：** 无效的 Tensor 形状

**出现原因：**
- Tensor 形状包含负数或零
- 形状参数格式错误

**解决办法：**
- 检查 Tensor 形状参数
- 确保形状值为正整数
- 使用有效的形状格式

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def invalid_shape_example():
    # 形状包含负数
    x = pypto.tensor([-1, 4], pypto.DT_FP32)
    return x

# 正确示例
@pypto.frontend.jit
def correct_shape_example():
    # 形状为正整数
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    return x
```

### TENSOR_DATATYPE_MISMATCH (0x12008)

**错误描述：** Tensor 数据类型不匹配

**出现原因：**
- 两个 Tensor 数据类型不匹配
- 操作要求特定数据类型但未满足

**解决办法：**
- 检查 Tensor 数据类型
- 使用类型转换函数
- 确保操作前数据类型匹配

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def datatype_mismatch_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = pypto.tensor([4, 4], pypto.DT_INT32)
    return pypto.add(x, y)  # 数据类型不匹配

# 正确示例
@pypto.frontend.jit
def correct_datatype_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = pypto.tensor([4, 4], pypto.DT_FP32)
    return pypto.add(x, y)  # 数据类型匹配
```

### TENSOR_NOT_UNDER_DYNAMIC_FUNCTION (0x1200D)

**错误描述：** Tensor 不在动态函数中

**出现原因：**
- 在动态函数外使用需要动态上下文的操作
- Tensor 操作需要在 @pypto.frontend.jit 装饰的函数内执行
- 使用了 GetTensorData/GetTensorDataInt32 等 API 但不在动态函数上下文中

**解决办法：**
- 将 Tensor 操作放在 @pypto.frontend.jit 装饰的函数内
- 确保操作在正确的上下文中执行
- GetTensorData 操作必须在动态函数中调用

**示例：**

```python
# 错误示例
def not_under_dynamic_function_example():
    x = pypto.tensor([4, 4], pypto.DT_INT32)
    y = x[0, 0]  # GetTensorData 需要在动态函数中执行
    return y

# 正确示例
@pypto.frontend.jit
def correct_dynamic_function_example(x):
    y = x[0, 0]  # GetTensorData 在动态函数中执行
    return y
```

**重要提示：**
- `x[0, 0]` 索引操作（GetTensorData）**仅支持 DT_INT32 类型的 Tensor**
- 对于其他数据类型（DT_FP32、DT_FP16、DT_INT64、DT_BOOL 等），使用索引操作会失败
- 如果需要对非 DT_INT32 类型 Tensor 进行元素访问，请使用其他操作如 reshape、view 等

### TENSOR_NO_ACTIVE_FUNCTION (0x1200F)

**错误描述：** 无活动函数

**出现原因：**
- 没有活动的函数上下文
- 在函数外执行需要函数上下文的操作

**解决办法：**
- 确保在函数上下文中执行操作
- 使用 @pypto.frontend.jit 装饰器
- 检查函数调用栈

**示例：**

```python
# 错误示例
def no_active_function_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = pypto.add(x, 1)  # 没有活动函数上下文
    return y

# 正确示例
@pypto.frontend.jit
def correct_active_function_example(x):
    y = pypto.add(x, 1)  # 有活动函数上下文
    return y
```

---

## Function 错误码

### FUNCTION_TYPE_MISMATCH (0x13007)

**错误描述：** 函数类型不匹配

**出现原因：**
- 函数返回类型与期望类型不符
- 函数参数类型不匹配

**解决办法：**
- 检查函数声明类型
- 使用正确的参数和返回类型
- 确保函数签名正确

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def function_type_mismatch_example(x: pypto.Tensor([4, 4], pypto.DT_FP32)):
    # 期望返回 FP32，但返回了 INT32
    return pypto.cast(x, pypto.DT_INT32)

# 正确示例
@pypto.frontend.jit
def correct_function_type_example(x: pypto.Tensor([4, 4], pypto.DT_FP32)):
    # 返回正确的类型
    return x
```

### FUNCTION_NOT_FOUND_BY_MAGIC (0x13008)

**错误描述：** 通过 magic 名称未找到函数

**出现原因：**
- 函数 magic 名称不存在
- 函数未正确注册

**解决办法：**
- 检查函数 magic 名称
- 确保函数已正确注册
- 使用正确的函数名称

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def function_not_found_example(x):
    y = pypto.call_function("nonexistent_function", x)
    return y

# 正确示例
@pypto.frontend.jit
def my_function(x):
    return pypto.add(x, 1)

@pypto.frontend.jit
def correct_function_call_example(x):
    y = pypto.call_function("my_function", x)
    return y
```

### FUNCTION_MAIN_NOT_FOUND (0x13009)

**错误描述：** 主函数未找到

**出现原因：**
- 程序缺少主函数
- 主函数名称不正确

**解决办法：**
- 定义主函数
- 使用正确的主函数名称
- 确保主函数已注册

**示例：**

```python
# 错误示例
# 缺少主函数

# 正确示例
@pypto.frontend.jit
def main(x):
    return pypto.add(x, 1)

# 调用主函数
result = main(input_tensor)
```

### FUNCTION_UNROLL_TIMES_MUST_BE_POSITIVE (0x1300C)

**错误描述：** 函数展开次数必须为正数

**出现原因：**
- 循环展开次数为零或负数
- unroll 参数值无效

**解决办法：**
- 确保展开次数为正整数
- 使用有效的 unroll 参数
- 检查循环展开配置

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def unroll_times_negative_example(x):
    for i in pypto.unroll(0):  # 展开次数为 0
        x = pypto.add(x, 1)
    return x

# 正确示例
@pypto.frontend.jit
def correct_unroll_times_example(x):
    for i in pypto.unroll(4):  # 展开次数为正数
        x = pypto.add(x, 1)
    return x
```

**注意：** 此错误码当前版本中可能未在代码中实际使用。

### FUNCTION_LOOP_INDEX_NAME_DUPLICATE (0x13010)

**错误描述：** 循环索引名称重复

**出现原因：**
- 嵌套循环使用相同的索引变量名
- 循环变量名冲突

**解决办法：**
- 使用不同的循环索引变量名
- 避免变量名冲突
- 使用有意义的变量名

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def loop_index_duplicate_example(x):
    for i in pypto.unroll(4):
        for i in pypto.unroll(4):  # 重复的索引名
            x = pypto.add(x, 1)
    return x

# 正确示例
@pypto.frontend.jit
def correct_loop_index_example(x):
    for i in pypto.unroll(4):
        for j in pypto.unroll(4):  # 不同的索引名
            x = pypto.add(x, 1)
    return x
```

**注意：** 此错误码当前版本中可能未在代码中实际使用。

### FUNCTION_LOOP_INDEX_NAME_DUPLICATE (0x13010)

**错误描述：** 循环索引名称重复

**出现原因：**
- 嵌套循环使用相同的索引变量名
- 循环变量名冲突

**解决办法：**
- 使用不同的循环索引变量名
- 避免变量名冲突
- 使用有意义的变量名

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def loop_index_duplicate_example(x):
    for i in pypto.unroll(4):
        for i in pypto.unroll(4):  # 重复的索引名
            x = pypto.add(x, 1)
    return x

# 正确示例
@pypto.frontend.jit
def correct_loop_index_example(x):
    for i in pypto.unroll(4):
        for j in pypto.unroll(4):  # 不同的索引名
            x = pypto.add(x, 1)
    return x
```

---

## 文件操作错误码

**注意：** 以下文件操作错误码在当前版本中可能未在代码中实际使用，这些错误码主要在 `error_code.h` 中定义，但实际文件操作可能使用其他错误处理机制。

### FILE_READ_FAILED (0x14001)

**错误描述：** 文件读取失败

**出现原因：**
- 文件不存在
- 文件权限不足
- 文件路径错误

**解决办法：**
- 检查文件是否存在
- 确保文件权限正确
- 使用正确的文件路径

**示例：**

```python
# 错误示例
import pypto

# 尝试读取不存在的文件
data = pypto.read_file("nonexistent_file.bin")

# 正确示例
import os

# 检查文件是否存在
if os.path.exists("data.bin"):
    data = pypto.read_file("data.bin")
else:
```

### FILE_WRITE_FAILED (0x14002)

**错误描述：** 文件写入失败

**出现原因：**
- 磁盘空间不足
- 文件权限不足
- 文件路径无效

**解决办法：**
- 检查磁盘空间
- 确保文件权限正确
- 使用有效的文件路径

**示例：**

```python
# 错误示例
import pypto

# 尝试写入到只读目录
data = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.write_file("/readonly/data.bin", data)

# 正确示例
import os

# 确保目录可写
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.write_file(f"{output_dir}/data.bin", data)
```

### FILE_CREATE_FAILED (0x14003)

**错误描述：** 文件创建失败

**出现原因：**
- 目录不存在
- 文件权限不足
- 文件名无效

**解决办法：**
- 检查目录是否存在
- 确保文件权限正确
- 使用有效的文件名

**示例：**

```python
# 错误示例
import pypto

# 尝试在不存在的目录中创建文件
pypto.create_file("/nonexistent_dir/new_file.bin")

# 正确示例
import os

# 确保目录存在
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pypto.create_file(f"{output_dir}/new_file.bin")
```

### FILE_DELETE_FAILED (0x14004)

**错误描述：** 文件删除失败

**出现原因：**
- 文件不存在
- 文件权限不足
- 文件被占用

**解决办法：**
- 检查文件是否存在
- 确保文件权限正确
- 关闭文件句柄

**示例：**

```python
# 错误示例
import pypto

# 尝试删除不存在的文件
pypto.delete_file("nonexistent_file.bin")

# 正确示例
import os

# 检查文件是否存在
if os.path.exists("data.bin"):
    pypto.delete_file("data.bin")
else:
    print("文件不存在")
```

### DIR_CREATE_FAILED (0x14101)

**错误描述：** 目录创建失败

**出现原因：**
- 父目录不存在
- 目录权限不足
- 目录名无效

**解决办法：**
- 检查父目录是否存在
- 确保目录权限正确
- 使用有效的目录名

**示例：**

```python
# 错误示例
import pypto

# 尝试创建多层目录（父目录不存在）
pypto.create_dir("/nonexistent/subdir")

# 正确示例
import os

# 递归创建目录
output_dir = "./output/subdir"
os.makedirs(output_dir, exist_ok=True)
```

### DIR_DELETE_FAILED (0x14102)

**错误描述：** 目录删除失败

**出现原因：**
- 目录不存在
- 目录不为空
- 目录权限不足

**解决办法：**
- 检查目录是否存在
- 清空目录内容
- 确保目录权限正确

**示例：**

```python
# 错误示例
import pypto

# 尝试删除非空目录
pypto.delete_dir("./output")

# 正确示例
import shutil

# 递归删除目录
if os.path.exists("./output"):
    shutil.rmtree("./output")
```

### FILE_WRITE_FAILED (0x14002)

**错误描述：** 文件写入失败

**出现原因：**
- 磁盘空间不足
- 文件权限不足
- 文件路径无效

**解决办法：**
- 检查磁盘空间
- 确保文件权限正确
- 使用有效的文件路径

**示例：**

```python
# 错误示例
import pypto

# 尝试写入到只读目录
data = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.write_file("/readonly/data.bin", data)

# 正确示例
import os

# 确保目录可写
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.write_file(f"{output_dir}/data.bin", data)
```

### FILE_CREATE_FAILED (0x14003)

**错误描述：** 文件创建失败

**出现原因：**
- 目录不存在
- 文件权限不足
- 文件名无效

**解决办法：**
- 检查目录是否存在
- 确保文件权限正确
- 使用有效的文件名

**示例：**

```python
# 错误示例
import pypto

# 尝试在不存在的目录中创建文件
pypto.create_file("/nonexistent_dir/new_file.bin")

# 正确示例
import os

# 确保目录存在
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pypto.create_file(f"{output_dir}/new_file.bin")
```

### FILE_DELETE_FAILED (0x14004)

**错误描述：** 文件删除失败

**出现原因：**
- 文件不存在
- 文件权限不足
- 文件被占用

**解决办法：**
- 检查文件是否存在
- 确保文件权限正确
- 关闭文件句柄

**示例：**

```python
# 错误示例
import pypto

# 尝试删除不存在的文件
pypto.delete_file("nonexistent_file.bin")

# 正确示例
import os

# 检查文件是否存在
if os.path.exists("data.bin"):
    pypto.delete_file("data.bin")
else:
    print("文件不存在")
```

### DIR_CREATE_FAILED (0x14101)

**错误描述：** 目录创建失败

**出现原因：**
- 父目录不存在
- 目录权限不足
- 目录名无效

**解决办法：**
- 检查父目录是否存在
- 确保目录权限正确
- 使用有效的目录名

**示例：**

```python
# 错误示例
import pypto

# 尝试创建多层目录（父目录不存在）
pypto.create_dir("/nonexistent/subdir")

# 正确示例
import os

# 递归创建目录
output_dir = "./output/subdir"
os.makedirs(output_dir, exist_ok=True)
```

### DIR_DELETE_FAILED (0x14102)

**错误描述：** 目录删除失败

**出现原因：**
- 目录不存在
- 目录不为空
- 目录权限不足

**解决办法：**
- 检查目录是否存在
- 清空目录内容
- 确保目录权限正确

**示例：**

```python
# 错误示例
import pypto

# 尝试删除非空目录
pypto.delete_dir("./output")

# 正确示例
import shutil

# 递归删除目录
if os.path.exists("./output"):
    shutil.rmtree("./output")
```

---

## TensorSlot 错误码

### TENSORSLOT_NOT_FOUND_IN_INDEX_DICT (0x15003)

**错误描述：** TensorSlot 在索引字典中未找到

**出现原因：**
- TensorSlot 未正确注册
- 索引查找失败

**解决办法：**
- 检查 TensorSlot 注册
- 确保索引正确
- 验证 TensorSlot 存在性

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def tensorslot_not_found_example(x):
    # 尝试访问未注册的 TensorSlot
    y = pypto.get_tensorslot(999)  # 无效的索引
    return y

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # 使用有效的 TensorSlot
    y = pypto.add(x, 1)
    return y
```

### TENSORSLOT_ALREADY_EXISTS (0x15004)

**错误描述：** TensorSlot 已存在

**出现原因：**
- 尝试添加已存在的 TensorSlot
- TensorSlot 名称或索引重复

**解决办法：**
- 检查 TensorSlot 是否已存在
- 使用唯一的 TensorSlot 标识符
- 避免重复注册

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def tensorslot_already_exists_example(x):
    # 重复添加相同的 TensorSlot
    slot1 = pypto.register_tensorslot("my_slot", x)
    slot2 = pypto.register_tensorslot("my_slot", x)  # 重复
    return slot1

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # 使用不同的名称
    slot1 = pypto.register_tensorslot("slot1", x)
    slot2 = pypto.register_tensorslot("slot2", x)
    return slot1
```

### TENSORSLOT_NOT_FOUND_IN_INPUT_DICT (0x15006)

**错误描述：** TensorSlot 在输入字典中未找到

**出现原因：**
- TensorSlot 不在输入列表中
- 输入参数不匹配

**解决办法：**
- 检查输入参数
- 确保所有 TensorSlot 都在输入中
- 验证函数签名

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def tensorslot_not_in_input_example():
    # 使用未在输入中的 TensorSlot
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    return x  # x 不在输入中

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # x 在输入中
    y = pypto.add(x, 1)
    return y
```

### TENSORSLOT_CHECKPOINT_STACK_ERROR (0x15007)

**错误描述：** TensorSlot 检查点堆栈错误

**出现原因：**
- 检查点堆栈操作失败
- 堆栈不平衡

**解决办法：**
- 检查检查点操作
- 确保堆栈平衡
- 正确使用 push/pop 操作

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def checkpoint_stack_error_example(x):
    # 不平衡的检查点操作
    pypto.push_checkpoint()
    pypto.push_checkpoint()
    pypto.pop_checkpoint()
    # 缺少一个 pop_checkpoint
    return x

# 正确示例
@pypto.frontend.jit
def correct_checkpoint_example(x):
    # 平衡的检查点操作
    pypto.push_checkpoint()
    pypto.push_checkpoint()
    pypto.pop_checkpoint()
    pypto.pop_checkpoint()
    return x
```

### TENSORSLOT_NOT_IN_INDEX_DICT (0x15008)

**错误描述：** TensorSlot 不在索引字典中

**出现原因：**
- TensorSlot 未正确索引
- 索引查找失败

**解决办法：**
- 检查 TensorSlot 索引
- 确保正确注册
- 验证索引字典完整性

**示例：**

```python
# 错误示例
@pypto.frontend.jit
def tensorslot_not_in_index_example(x):
    # 尝试访问未索引的 TensorSlot
    y = pypto.lookup_tensorslot("unknown_slot")
    return y

# 正确示例
@pypto.frontend.jit
def correct_tensorslot_example(x):
    # 正确访问 TensorSlot
    y = pypto.add(x, 1)
    return y
```

---

## 符号操作错误码

### SYMBOLIC_OPERAND_COUNT_MISMATCH (0x19001)

**错误描述：** 符号操作数数量不匹配

**出现原因：**
- 操作数数量与期望不符
- 符号表达式参数错误

**解决办法：**
- 检查操作数数量
- 使用正确的参数数量
- 参考 API 文档中的参数要求

**示例：**

```python
# 错误示例
x = pypto.SymbolicScalar("x")
y = pypto.SymbolicScalar("y")

# 操作数数量不匹配
z = x + y + 1  # 可能导致操作数数量错误

# 正确示例
x = pypto.SymbolicScalar("x")
y = pypto.SymbolicScalar("y")

# 正确的操作数数量
z = x + y
```

### SYMBOLIC_OPERAND_SIZE_INVALID (0x19001)

**错误描述：** 符号操作数大小无效

**出现原因：**
- 操作数大小不符合要求
- 符号表达式维度错误

**解决办法：**
- 检查操作数大小
- 使用正确的操作数维度
- 确保符号表达式有效

**示例：**

```python
# 错误示例
x = pypto.SymbolicScalar("x")
y = pypto.SymbolicScalar("y")

# 尝试对符号进行无效的操作
z = pypto.reshape(x, [4, 4])  # 符号不能 reshape

# 正确示例
x = pypto.SymbolicScalar("x")
y = pypto.SymbolicScalar("y")

# 正确的符号操作
z = x + y
```

---

## 调试技巧

### 1. 启用详细日志

```python
import pypto

# 启用详细日志
pypto.set_log_level(pypto.LogLevel.DEBUG)
```

### 2. 使用断言检查

```python
@pypto.frontend.jit
def debug_example(x):
    assert x.shape == [4, 4], "输入形状必须为 [4, 4]"
    assert x.dtype == pypto.DT_FP32, "输入类型必须为 FP32"
    y = pypto.add(x, 1)
    return y
```

### 3. 检查 Tensor 信息

```python
@pypto.frontend.jit
def check_tensor_info(x):
    print(f"Shape: {x.shape}")
    print(f"Dtype: {x.dtype}")
    print(f"Device: {x.device}")
    y = pypto.add(x, 1)
    return y
```

### 4. 使用 try-except 捕获错误

```python
try:
    result = my_function(input_tensor)
except Exception as e:
    print(f"错误发生: {e}")
    print(f"错误类型: {type(e).__name__}")
```

### 5. 验证输入输出

```python
import numpy as np

# 验证输入
input_data = np.random.rand(4, 4).astype(np.float32)
print(f"输入形状: {input_data.shape}")
print(f"输入类型: {input_data.dtype}")

# 执行函数
result = my_function(input_data)

# 验证输出
print(f"输出形状: {result.shape}")
print(f"输出类型: {result.dtype}")
```

---

## 常见错误模式

### 1. 形状不匹配

**症状：** SHAPE_MISMATCH 错误

**解决方案：**
```python
@pypto.frontend.jit
def handle_shape_mismatch(x, y):
    # 检查形状
    if x.shape != y.shape:
        y = pypto.reshape(y, x.shape)
    return pypto.add(x, y)
```

### 2. 数据类型不匹配

**症状：** TENSOR_DATATYPE_MISMATCH 错误

**解决方案：**
```python
@pypto.frontend.jit
def handle_datatype_mismatch(x, y):
    # 统一数据类型
    if x.dtype != y.dtype:
        y = pypto.cast(y, x.dtype)
    return pypto.add(x, y)
```

### 3. 动态函数上下文错误

**症状：** TENSOR_NOT_UNDER_DYNAMIC_FUNCTION 错误

**解决方案：**
```python
# 错误：在动态函数外操作
def wrong_example():
    x = pypto.tensor([4, 4], pypto.DT_FP32)
    y = x[0, 0]  # 错误
    return y

# 正确：在动态函数内操作
@pypto.frontend.jit
def correct_example(x):
    y = x[0, 0]  # 正确
    return y
```

---

## 重要提示

### 未在代码中实际使用的错误码

以下错误码在 `error_code.h` 中定义，但可能在当前版本的代码中未实际使用：

1. **Function 错误码**：
   - `FUNCTION_UNROLL_TIMES_MUST_BE_POSITIVE (0x1300C)`
   - `FUNCTION_LOOP_INDEX_NAME_DUPLICATE (0x13010)`

2. **文件操作错误码**：
   - `FILE_READ_FAILED (0x14001)`
   - `FILE_WRITE_FAILED (0x14002)`
   - `FILE_CREATE_FAILED (0x14003)`
   - `FILE_DELETE_FAILED (0x14004)`
   - `DIR_CREATE_FAILED (0x14101)`
   - `DIR_DELETE_FAILED (0x14102)`

3. **TensorSlot 错误码**：
   - `TENSORSLOT_NOT_FOUND_IN_INDEX_DICT (0x15003)`
   - `TENSORSLOT_ALREADY_EXISTS (0x15004)`
   - `TENSORSLOT_NOT_FOUND_IN_INPUT_DICT (0x15006)`
   - `TENSORSLOT_CHECKPOINT_STACK_ERROR (0x15007)`
   - `TENSORSLOT_NOT_IN_INDEX_DICT (0x15008)`

4. **符号操作错误码**：
   - `SYMBOLIC_OPERAND_COUNT_MISMATCH (0x19001)`
   - `SYMBOLIC_OPERAND_SIZE_INVALID (0x19001)`

这些错误码可能在未来的版本中使用，或者在其他模块中使用。如果在代码中遇到这些错误，请参考本文档中的解决方案。

### 实际常用的错误处理机制

当前框架中主要使用以下错误处理机制：

1. **ASSERT 宏**：用于检查条件，失败时抛出异常
   - 示例：`ASSERT(currDynFunc != nullptr) << "Not under dynamic function!\n"`

2. **CHECK 宏**：类似于 ASSERT，用于运行时检查
   - 示例：`CHECK(t.GetDataType() == DT_INT32) << "Tensor dtype must be DT_INT32."`

3. **FUNCTION_LOGE/FUNCTION_LOGD**：日志记录
   - 示例：`FUNCTION_LOGE("Error: No active function to add operation.")`

---

## 参考资源

- [PyPTO API 文档](./docs/api/)
- [PyPTO 示例代码](./examples/)
- [错误处理最佳实践](./docs/best_practices/error_handling.md)
- [调试指南](./docs/guides/debugging.md)

---

## 总结

本技能文档涵盖了 PyPTO 开发中最常见的错误码，包括：

- **通用错误码**：8 个
- **配置错误码**：5 个
- **Tensor 错误码**：8 个
- **Function 错误码**：5 个
- **文件操作错误码**：6 个
- **TensorSlot 错误码**：5 个
- **符号操作错误码**：2 个

遇到错误时，请按照以下步骤处理：

1. **识别错误码**：查看错误信息中的错误码
2. **查找文档**：在本文档中查找对应的错误码
3. **理解原因**：阅读错误出现的原因
4. **应用解决方案**：按照文档中的解决方案修复问题
5. **参考示例**：查看示例代码了解正确用法

如果问题仍未解决，请：
- 检查完整的错误堆栈信息
- 启用详细日志获取更多信息
- 参考官方示例代码
- 查阅 API 文档
