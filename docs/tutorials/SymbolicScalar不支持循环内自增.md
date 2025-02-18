# SymbolicScalar不支持循环内自增<a name="ZH-CN_TOPIC_0000002498320996"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

```
@pypto.jit
def add_kernel_1(a, b, c):
    count = 0
    for i in pypto.loop(20):
        count = count + 1
```

实际执行到i=1时， count并不会如用户预期的从0依次增加到20

## 可能原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

当前PyPto框架只Capture了用户的Tensor操作，并没有Capture的用户的scalar操作，不会将count处理为变量，目前只有循环变量可以自增

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

通过循环变量表达自增逻辑

