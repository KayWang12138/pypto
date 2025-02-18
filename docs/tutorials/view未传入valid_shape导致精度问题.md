# view未传入valid\_shape导致精度问题<a name="ZH-CN_TOPIC_0000002498320990"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

部分场景使用view时未传入valid\_shape导致精度问题

## 问题原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

当view接口的输入tensor没有一个正确的validshape时，框架无法正确推导出输出的validshape。

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

当怀疑view部分的validshape推导有问题时，首先给view传入一个validshape，观察输出结果是否符合预期。

一个典型的需要传入validshape的场景：

当输入的validShape依赖别的tensor标识，必须传入dynValidShape。如下所示场景，q0的validshape  curSeq来自于另外一个tensor，无法通过推导得到

```
输入 input [B, S, H]
输入 act_seqs [B]
输出 out [B, S, H]
计算过程  AddS
代码如下：
for b_idx in pypto.loop(B, name="b_loop", idx_name="b"):
    cur_seq = act_seqs[b_idx]
    a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
    a1 = a0 + 1.0
    out[b_idx:, :, :] = a1
```

