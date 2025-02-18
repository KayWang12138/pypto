# 使用未初始化的 Tensor<a name="ZH-CN_TOPIC_0000002530080971"></a>

## 问题现象描述<a name="zh-cn_topic_0000001265073070_section32145724"></a>

使用 pypto.tensor 声明了一个 Tensor，误以为它与 torch.empty 类似会申请一块脏数据内存，在再次写入前直接读它（如使用 view），出现框架校验错误或精度错误。

## 问题原因<a name="zh-cn_topic_0000001265073070_section20876063"></a>

在 PyPTO 中，除 pypto.full/pypto.zeros 等显式包含初始化行为的 Tensor 声明行为，还可使用 pypto.tensor 声明 Tensor，但该接口**不包含初始化行为**。PyPTO 中要求每个 Tensor 必须先写再读，即要先有 producer 再有 consumer，未初始化的 Tensor 不申请内存。框架中通常会校验直接使用 no producer Tensor 并报错，但也出现过因校验遗漏而出现上板精度错误的情况。

## 处理步骤<a name="zh-cn_topic_0000001265073070_section13239568"></a>

避免使用未经初始化的 Tensor。

