# pypto.max

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算多个值中的最大值，用于符号标量和动态形状场景下的边界管理。支持以下用法：
- 两个参数：`max(a, b)`
- 多个参数：`max(a, b, c, d, ...)`
- 可迭代对象：`max([a, b, c])` 或 `max(iterable)`
- 单个值：`max(a)` - 返回值本身

## 函数原型

```python
# 可迭代对象
@overload
def max(iterable: Iterable[Union[SymbolicScalar, int]]) -> SymbolicScalar:
    ...

# 两个参数
@overload
def max(a: Union[SymbolicScalar, int], b: Union[SymbolicScalar, int]) -> SymbolicScalar:
    ...

# 多个参数
@overload
def max(
    a: Union[SymbolicScalar, int],
    b: Union[SymbolicScalar, int],
    *args: Union[SymbolicScalar, int],
) -> SymbolicScalar:
    ...

# 实际实现
def max(*args, **kwargs) -> SymbolicScalar:
    ...
```

## 参数说明

| 参数名    | 输入/输出 | 说明                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| iterable  | 输入      | 可迭代对象，元素可以是整数或符号标量（SymbolicScalar）。当只传入一个参数时，如果该参数是可迭代对象，则返回其元素中的最大值；如果该参数是整数或符号标量，则返回其本身。 |
| a         | 输入      | 第一个参数，可以是整数或符号标量（SymbolicScalar） |
| b         | 输入      | 第二个参数，可以是整数或符号标量（SymbolicScalar） |
| *args     | 输入      | 可选的额外参数，可以是整数或符号标量（SymbolicScalar） |

## 返回值说明

SymbolicScalar: 返回所有参数中的最大值，类型为符号标量。

## 约束说明

- 传入参数可以是以下形式之一：
  - 单个可迭代对象：返回其元素中的最大值
  - 单个整数或符号标量：返回其本身（包装为 SymbolicScalar）
  - 两个或多个整数或符号标量：返回其中的最大值
- 参数类型可以是整数或符号标量（SymbolicScalar）

## 调用示例

```python
import pypto

# 示例1：两个参数（向后兼容）
a = pypto.symbolic_scalar(10)
b = pypto.symbolic_scalar(20)
result = pypto.max(a, b)  # 返回 SymbolicScalar(20)

# 示例2：多个参数
result = pypto.max(10, 30, 5, 20)  # 返回 SymbolicScalar(30)

# 示例3：可迭代对象
values = [pypto.symbolic_scalar(10), 30, pypto.symbolic_scalar(20)]
result = pypto.max(values)  # 返回 SymbolicScalar(30)

# 示例4：单个值
result = pypto.max(20)  # 返回 SymbolicScalar(20)
result = pypto.max(pypto.symbolic_scalar(10))  # 返回 SymbolicScalar(10)

# 示例5：在 kernel 中使用（未使用装饰器）
def kernel():
    b_offset = 0
    tile_b = 32
    b_offset_end = pypto.max(b_offset + tile_b, b_offset + 1)

# 示例6：在 kernel 中使用（使用装饰器，无需 pypto.max 包装）
@pypto.frontend.jit
def kernel():
    b_offset = 0
    tile_b = 32
    b_offset_end = max(b_offset + tile_b, b_offset + 1)
```
