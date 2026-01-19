# pypto.frontend.function

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

`pypto.frontend.function` 用于定义可复用的计算子图或函数模块，允许在多个内核函数中复用相同的计算逻辑。该功能旨在提高代码的模块化和可维护性。

预期特性：
- **代码复用**: 将常用的计算模式封装为函数
- **模块化设计**: 构建复杂算子时可以组合多个小函数
- **类型安全**: 函数签名中明确输入输出类型
- **自动优化**: 编译器可以内联或优化函数调用

 与 pypto.frontend.jit 的区别：

| 特性 | pypto.frontend.jit | pypto.frontend.function |
|------|-------------------|------------------------|
| 用途 | 定义可执行的内核函数 | 定义可复用的子函数 |
| 编译 | 编译为完整的计算图 | 作为子图被内联或优化 |
| 调用方式 | 可被外部直接调用 | 只能在 JIT 函数内调用 |
| 执行 | 独立执行单元 | 嵌入到父函数中执行 |

## 函数原型

```python
# 基本语法
@pypto.frontend.function
def function_name(
    arg1: pypto.Tensor,
    arg2: pypto.Tensor,
    ...
) -> pypto.Tensor:
    # 函数实现
    ...

# 具体示例：定义一个可复用的加法函数
@pypto.frontend.function
def add_tensors(
    x: pypto.Tensor,
    y: pypto.Tensor
) -> pypto.Tensor:
    z = x + y
    return z

# 在 JIT 函数中使用
@pypto.frontend.jit
def kernel_function(
    a: pypto.Tensor,
    b: pypto.Tensor,
    c: pypto.Tensor
) -> pypto.Tensor:
    # 复用 add_tensors 函数
    temp = add_tensors(a, b)
    result = add_tensors(temp, c)
    return result
```

## 参数说明

暂无参数配置。装饰器直接应用于函数。

## 返回值说明

返回装饰后的可复用函数对象。

## 约束说明

1. 只支持函数名调用，不支持方法调用
2. 参数：提供完整数量(无缺省值)严格按声明顺序传递(无乱序/关键词)
3. 不支持递归函数
