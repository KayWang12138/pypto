# pypto.max

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算两个值中的最大值，实现 Python 中的 max() 功能。支持符号标量和动态形状场景下的边界管理。

## 函数原型

```python
max(a: Union[SymbolicScalar, int], b: Union[SymbolicScalar, int]) -> SymbolicScalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| a      | 输入      | 第一个参数，可以是整数或符号标量（SymbolicScalar） |
| b      | 输入      | 第二个参数，可以是整数或符号标量（SymbolicScalar） |

## 返回值说明

SymbolicScalar: 返回两个参数中的最大值，类型为符号标量。

## 约束说明

-   只支持两个参数
-   参数类型可以是整数或符号标量（SymbolicScalar）
-   当函数未使用 @pypto.frontend.jit 或 @pypto.frontend.function 装饰器修饰时，需要使用 pypto.max 而非内置的 max

## 调用示例

```python
# 未使用装饰器，需要用 pypto.max 包装
def kernel():
    ...
    b_offset_end = pypto.max(b_offset + tile_b, b_offset + 1)

# 使用装饰器，无需 pypto.max 包装
@pypto.frontend.jit
def kernel():
    ...
    b_offset_end = max(b_offset + tile_b, b_offset + 1)
```
