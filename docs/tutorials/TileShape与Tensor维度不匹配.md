# TileShape与Tensor维度不匹配<a name="ZH-CN_TOPIC_0000002498161028"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

算子执行时出现如下报错：

```
2025-12-18 10:33:06.107 E | [ExpandFunction][Function][ERROR]: FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0] ExpandFunction failed: Tile shape size 1 is not matched the output shape size 2.
2025-12-18 10:33:06.107 E | Run pass [ExpandFunction] failed.
2025-12-18 10:33:06.107 E | Run pass <ExpandFunction> failed
```

## 可能原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

某个操作的Tile Shape设置的维度过小，小于该操作的输出Tensor的Shape维度，导致出现错误。

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

根据报错提示定位到相应的循环，如下所说，问题代码出现在b\_loop循环中。

```
FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0]
```

找到对应的循环后，根据日志提示的错误维度以及代码逻辑，确定代码中Tile Shape的维度为1，而输出Shape的维度为2，将Tile Shape重新设置为2维即可。

