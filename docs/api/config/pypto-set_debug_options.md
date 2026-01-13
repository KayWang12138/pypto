# pypto.set\_debug\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置debug的选项。

## 函数原型

```python
set_debug_options(*,
                  compile_debug_mode: Optional[Union[int, DebugMode]] = None,
                  runtime_debug_mode: Optional[Union[int, DebugMode]] = None,
                  ) -> None
```

## 参数说明


| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| compile_debug_mode   | 输入      | 含义：设置编译阶段调试模式 <br> 说明：DebugMode.NONE (0)：代表默认不使能编译阶段调试模式； <br> DebugMode.ALL (1)：代表图使能编译阶段调试模式，一键开启图编译相关配置，包括计算图； <br> 类型：int 或 DebugMode <br> 取值范围：DebugMode.NONE (0) 或 DebugMode.ALL (1) <br> 默认值：DebugMode.NONE (0) <br> 影响Pass范围：NA |
| runtime_debug_mode   | 输入      | 含义：设置执行阶段调试模式 <br> 说明：DebugMode.NONE (0)：代表默认不使能执行阶段调试模式； <br> DebugMode.ALL (1)：代表使能执行阶段调试模式，一键开启图执行相关配置，包括泳道图、aicpu仿真； <br> 类型：int 或 DebugMode <br> 取值范围：DebugMode.NONE (0) 或 DebugMode.ALL (1) <br> 默认值：0 <br> 影响Pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
# 使用整数值
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)

# 使用枚举类（推荐）
pypto.set_debug_options(compile_debug_mode=pypto.DebugMode.ALL)
pypto.set_debug_options(runtime_debug_mode=pypto.DebugMode.ALL)
```

## 说明

当 `runtime_debug_mode=pypto.DebugMode.ALL`时，会启用以下功能：
- 泳道图：自动启用性能分析，生成泳道图用于性能调优
- aicpu仿真：自动启用在仿真模式下运行aicpu仿真
