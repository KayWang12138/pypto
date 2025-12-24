# pypto.get\_verify\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取当前已配置的精度调试 Verify 特性选项。

## 函数原型

```python
get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## 参数说明

无。

## 返回值说明

获取精度调试 Verify 特性的当前设定值。

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
pypto.get_verify_options()
```

