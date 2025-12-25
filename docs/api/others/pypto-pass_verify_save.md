# pypto.pass\_verify\_save

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

在精度调试 Verify 特性使能时，使用该接口保存指定TTensor拟计算的结果到数据文件。

## 函数原型

```python
pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar],
) -> None
```

## 参数说明


| 参数名   | 输入/输出 | 说明                                                                 |
|----------|-----------|----------------------------------------------------------------------|
| tensor   | 输入      | 含义：总体使能开关，决定所有 *pass_verify_* 选项、接口是否有效。 <br> 说明：True：代表使能。 <br> 类型：bool <br> 取值范围：True/False <br> 默认值：False |
| fname    | 输入      | 含义：文件名模板，定义tensor保存的文件名前缀，tensor的内存转储保存至{fname}.data、tensor的元数据（shape,dtype）保存至{fname}.csv。保存路径为：${work_path}/output/output_*/tensor/ <br> 说明：str：简单文件名前缀；包含"$NAME"的待匹配字符串：将$NAME替换为kwargs中NAME对应的值，然后以替换后的字符串作为文件名前缀。 <br> 类型：str <br> 取值范围：NA <br> 默认值：tensor的变量名 |
| cond     | 输入      | 含义：指定打印数据的满足条件 <br> 说明：表达式计算结果为1：打印指定数据；表达式计算结果为0：不打印数据。 <br> 类型：Optional[int,pypto.SymbolicScalar] <br> 取值范围：0,1 <br> 默认值：1 |
| **kwargs | 输入      | 指定fname参数中待匹配字符串的值。 |

## 返回值说明

无。

## 约束说明

需重新编译、安装PyPTO后才能使用该接口。编译安装流程如下：

1.  根据docs/context/build\_and\_install.md指导安装编译环境。
2.  确认Python环境中已按照安装依赖安装了PyTorch。
3.  确认GCC版本安装、升级到9.4.0或更高版本。
4.  在编译、安装命令中增加选项 --no-build-isolation，例如：

    ```
    python3 -m pip install . --verbose --no-build-isolation ... ...
    ```

5.  卸载当前安装的pypto Python包，执行编译、安装命令

## 调用示例

```python
verify_options = {
        "enable_pass_verify": True,
      }

@pypto.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        # 保存 t2 到文件 t2.*
        pypto.pass_verify_save(t2)
        t3 = pypto.SOME_OP2(t0, t2)
        # 当 idx==5 时，保存 t3 到文件 t3_debug_loop_5.* 中
        pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
         ...

```

