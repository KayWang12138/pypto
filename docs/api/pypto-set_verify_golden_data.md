# pypto.set\_verify\_golden\_data<a name="ZH-CN_TOPIC_0000002532132903"></a>

精度调试 Verify 特性开启时，设置粗检模式的用户 golden 基准数据。

-   设置的 golden 数据用于和算子的模拟计算比较、粗粒度检验算子正确性。

## 产品支持情况<a name="section1550532418810"></a>

<a name="table0318142155520"></a>
<table><thead align="left"><tr id="row11318625556"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p103183216558"><a name="p103183216558"></a><a name="p103183216558"></a><span id="ph131818275517"><a name="ph131818275517"></a><a name="ph131818275517"></a>AI处理器类型</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p731814235515"><a name="p731814235515"></a><a name="p731814235515"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row83182245518"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p73181217557"><a name="p73181217557"></a><a name="p73181217557"></a><span id="ph1531810211556"><a name="ph1531810211556"></a><a name="ph1531810211556"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1731816220558"><a name="p1731816220558"></a><a name="p1731816220558"></a>√</p>
</td>
</tr>
<tr id="row431872155513"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p19318729554"><a name="p19318729554"></a><a name="p19318729554"></a><span id="ph431892195513"><a name="ph431892195513"></a><a name="ph431892195513"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1131820275516"><a name="p1131820275516"></a><a name="p1131820275516"></a>√</p>
</td>
</tr>
<tr id="row1431852185515"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p83185265512"><a name="p83185265512"></a><a name="p83185265512"></a><span id="ph183181529556"><a name="ph183181529556"></a><a name="ph183181529556"></a><term id="zh-cn_topic_0000001312391781_term354143892110"><a name="zh-cn_topic_0000001312391781_term354143892110"></a><a name="zh-cn_topic_0000001312391781_term354143892110"></a>Ascend 310B</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p124951335517"><a name="p124951335517"></a><a name="p124951335517"></a>☓</p>
</td>
</tr>
<tr id="row173191727550"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1031982135514"><a name="p1031982135514"></a><a name="p1031982135514"></a><span id="ph731910215512"><a name="ph731910215512"></a><a name="ph731910215512"></a><term id="zh-cn_topic_0000001312391781_term4363218112215"><a name="zh-cn_topic_0000001312391781_term4363218112215"></a><a name="zh-cn_topic_0000001312391781_term4363218112215"></a>Ascend 310P</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p182521413105513"><a name="p182521413105513"></a><a name="p182521413105513"></a>☓</p>
</td>
</tr>
<tr id="row4319162125515"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p9319172115519"><a name="p9319172115519"></a><a name="p9319172115519"></a><span id="ph23191215518"><a name="ph23191215518"></a><a name="ph23191215518"></a><term id="zh-cn_topic_0000001312391781_term71949488213"><a name="zh-cn_topic_0000001312391781_term71949488213"></a><a name="zh-cn_topic_0000001312391781_term71949488213"></a>Ascend 910</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p112553131552"><a name="p112553131552"></a><a name="p112553131552"></a>☓</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

该接口包括以下必要功能：

-   功能1：将用户执行算子时实际的输入、输出列表设置到工具中，用于工具在粗检时使用同样的输入进行模拟计算
-   功能2：将用户已有的计算基准数据（golden）设置到工具中，用于工具在计算出模拟结果后与基准输出对比，进而粗粒度确定模拟结果的正确性

## 函数原型<a name="section1814166202715"></a>

```
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## 参数说明<a name="section27141942204919"></a>

<a name="table332075643414"></a>
<table><thead align="left"><tr id="row173201156103418"><th class="cellrowborder" valign="top" width="20.75%" id="mcps1.1.5.1.1"><p id="p3552162123616"><a name="p3552162123616"></a><a name="p3552162123616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="6.950000000000001%" id="mcps1.1.5.1.2"><p id="p85521928363"><a name="p85521928363"></a><a name="p85521928363"></a>Python 类型</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.5.1.3"><p id="p25261435213"><a name="p25261435213"></a><a name="p25261435213"></a>框架默认值</p>
</th>
<th class="cellrowborder" valign="top" width="64.78%" id="mcps1.1.5.1.4"><p id="p255272123616"><a name="p255272123616"></a><a name="p255272123616"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row15320125613420"><td class="cellrowborder" valign="top" width="20.75%" headers="mcps1.1.5.1.1 "><p id="p10780849134914"><a name="p10780849134914"></a><a name="p10780849134914"></a>in_out_tensors</p>
</td>
<td class="cellrowborder" valign="top" width="6.950000000000001%" headers="mcps1.1.5.1.2 "><p id="p586575610352"><a name="p586575610352"></a><a name="p586575610352"></a>List[Union(pypto.Tensor, torch.Tensor)]</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.3 "><p id="p152654105210"><a name="p152654105210"></a><a name="p152654105210"></a>None</p>
</td>
<td class="cellrowborder" valign="top" width="64.78%" headers="mcps1.1.5.1.4 "><p id="p354484515316"><a name="p354484515316"></a><a name="p354484515316"></a>将用户执行算子时实际的输入、输出列表按照相同位置对应地设置到检测工具。</p>
<p id="p55641956145111"><a name="p55641956145111"></a><a name="p55641956145111"></a><strong id="b867635217527"><a name="b867635217527"></a><a name="b867635217527"></a>注：jit 调用模式下该选项不需设置，相应数据由 jit 自动传入精度工具</strong></p>
</td>
</tr>
<tr id="row332055610344"><td class="cellrowborder" valign="top" width="20.75%" headers="mcps1.1.5.1.1 "><p id="p2890201018711"><a name="p2890201018711"></a><a name="p2890201018711"></a>goldens</p>
</td>
<td class="cellrowborder" valign="top" width="6.950000000000001%" headers="mcps1.1.5.1.2 "><p id="p879516274294"><a name="p879516274294"></a><a name="p879516274294"></a>List[Union(pypto.Tensor, torch.Tensor)]</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.3 "><p id="p9526144195211"><a name="p9526144195211"></a><a name="p9526144195211"></a>None</p>
</td>
<td class="cellrowborder" valign="top" width="64.78%" headers="mcps1.1.5.1.4 "><p id="p779085023211"><a name="p779085023211"></a><a name="p779085023211"></a>将用户已有的计算基准数据（golden）输出列表，按照实际输出数据在列表中的位置设置到工具中。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

void：Set方法无返回值。设置操作成功即生效。

## 约束说明<a name="section753174210543"></a>

无。

## 调用示例<a name="section646021318230"></a>

```
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```

