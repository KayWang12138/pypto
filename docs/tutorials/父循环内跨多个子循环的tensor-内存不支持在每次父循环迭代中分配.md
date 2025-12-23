# 父循环内跨多个子循环的tensor 内存不支持在每次父循环迭代中分配<a name="ZH-CN_TOPIC_0000002530240947"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

两层或者两层以上循环嵌套下，在父循环中定义了一个 tensor，在一个子循环中写入该 tensor，在另一个子循环中使用该 tensor 时，存在精度错误。

## 可能原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

该 GM 上的 tensor 在父循环的多次迭代，每次迭代分配的内存地址都是相同的，导致不同的迭代应该使用不同的临时内存来保存数据，但实际使用的相同的地址。由于我们的不同循环迭代实际是并行执行的，因此当不同循环迭代同时执行时，后一个执行的迭代会覆盖前一次迭代的临时内存，因此产生精度问题。

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

在后一个子循环上添加 submit\_before\_loop=True，在后一个子循环启动前下发任务，强制使得多次迭代串行运行，从而避免并行执行时存在内存覆盖和踩踏导致的精度问题。

```
for outer in pypto.loop(...): # 父循环，执行至少两次，如果只执行一次，不存在多次并行覆盖的问题
    t = pypto.Tensor(...)  # 定义一个临时tensor
    for inner0 in pypto.loop(...): # 第一个子循环，对临时tensor t赋值
        ...
        t[...] = ... 
    for inner1 in pypto.loop(..., submit_before_loop=True): # 第二个子循环，使用了临时 tensor t，
                                                   # 添加 submit_before_loop，确保父循环多次迭代不在同一个并行执行块中
        x[:] = t[:] + t[:]
```

