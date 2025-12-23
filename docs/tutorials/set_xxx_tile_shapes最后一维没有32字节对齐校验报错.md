# set\_xxx\_tile\_shapes最后一维没有32字节对齐校验报错<a name="ZH-CN_TOPIC_0000002530240943"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

```
with pypto.function("TENSOR_COS_CONTENT_FP32", [x], [res]):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(x.sum())
```

通过pypto.set\_xxx\_tile\_shapes设置tileshape大小最后一维需要32字节对齐，否则会校验报错。

![](figures/zh-cn_image_0000002531503839.png)

## 问题原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

硬件指令限制处理的数据需要32字节对齐。

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

通过pypto.set\_xxx\_tile\_shapes设置tileshape大小需要将最后一维大小设置成32字节对齐的数，即tileshape\[-1\] \* sizeof\(dtype\) % 32 == 0。

