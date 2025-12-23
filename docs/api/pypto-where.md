# pypto.where<a name="ZH-CN_TOPIC_0000002503358003"></a>

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

## 功能说明<a name="section15101187760"></a>

condition 为一个布尔类型的掩码张量（mask tensor）。对于张量中任意位置的元素，该操作基于布尔掩码张量 condition 进行逐元素选择：当 condition 在对应位置的值为 True 时，输出在对应位置的值取自 input对应位置的值；当其值为 False 时，输出在对应位置的值取自 other对应位置的值。其中，input 与 other 均可以是张量或标量值；若input为标量，当 condition 在对应位置的值为 True 时，result在对应位置的值填充iuput的值，若other为标量，当 condition 在对应位置的值为 False 时，result在对应位置的值填充other的值。其计算行为可形式化表示为如下表达式。

![](figures/zh-cn_formulaimage_0000002499856712.png)

condition 须为Tensor ，input 和 other 可以为 Tensor、 float  以及 Element，广播规则如下（只支持单轴广播）：

1.  input, other, condition 均为 Tensor 时，result 的 Shape 由三者广播得到。

    例：iuput:\[1,20,20\], other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

2.  只有 input, condition 为 Tensor 时，result 的 Shape 由两者广播得到。

    例：iuput:\[1,20,20\], condition:\[20,20,1\], result:\[20,20,20\]

3.  只有 other, condition 为 Tensor 时，result 的 Shape 由两者广播得到。

    例：other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

4.  只有 condition 为 Tensor 时，result 的 Shape 与 condition 一致。

## 函数原型<a name="section19138102360"></a>

```
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

## 参数说明<a name="section75724101161"></a>

<a name="zh-cn_topic_0146324969_table29998725"></a>
<table><thead align="left"><tr id="zh-cn_topic_0146324969_row8953505"><th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0146324969_p54145286"><a name="zh-cn_topic_0146324969_p54145286"></a><a name="zh-cn_topic_0146324969_p54145286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0146324969_p23692060"><a name="zh-cn_topic_0146324969_p23692060"></a><a name="zh-cn_topic_0146324969_p23692060"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.87%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0146324969_p19480441"><a name="zh-cn_topic_0146324969_p19480441"></a><a name="zh-cn_topic_0146324969_p19480441"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>condition</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p1175532913131"><a name="p1175532913131"></a><a name="p1175532913131"></a>Tensor，数据类型为DT_BOOL，作为条件选择input或者other的元素。（Tensor支持2维到4维）</p>
</td>
</tr>
<tr id="row78991020134214"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p08991420144215"><a name="p08991420144215"></a><a name="p08991420144215"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p54121938114215"><a name="p54121938114215"></a><a name="p54121938114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p1689917200424"><a name="p1689917200424"></a><a name="p1689917200424"></a>Tensor or float or Element标量，输入类型支持DT_FP32和DT_FP16。作为被选择的元素组成 result，condition 的对应元素为 true 时选择 input 的元素。（Tensor支持2维到4维）</p>
</td>
</tr>
<tr id="zh-cn_topic_0146324969_row46369059"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p041119307251"><a name="p041119307251"></a><a name="p041119307251"></a>other</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p161002045193414"><a name="p161002045193414"></a><a name="p161002045193414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p9307133912448"><a name="p9307133912448"></a><a name="p9307133912448"></a>Tensor or float  or Element 标量，输入类型支持 DT_FP32 和 DT_FP16。作为被选择的元素组成 result，condition 的对应元素为 false 时选择 other 的元素。（Tensor支持2维到4维）</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26662142616"></a>

result ：Tensor，shape由输入的广播得到,详细广播场景可看上文。数据类型和input、other保持一致。

## 约束说明<a name="section666154916252"></a>

1. 建议优先使用 Element，传入 float 标量对于 fp16 场景，不保证正确性。

## 调用示例<a name="section4127133461016"></a>

```
cond1 = pypto.tensor([4], pypto.DT_BOOL)
a1 = pypto.tensor([4], pypto.DT_FP32)
b1 = pypto.tensor([4], pypto.DT_FP32)
out1 = pypto.where(cond1, a1, b1)

# Using scalar inputs
out2 = pypto.where(cond1, 1, 0) 

# Broadcasting example
cond2 = pypto.tensor([2, 2], pypto.DT_BOOL) 
a2 = pypto.tensor([2], pypto.DT_FP32)  
b2 = 0
out3 = pypto.where(cond2, a2, b2)
```

结果示例如下：

```
输入数据cond1: [True, False, True, False]
输入数据a1: [1, 2, 3, 4]
输入数据b1: [10, 20, 30, 40]
输出数据out1: [ 1, 20,  3, 40]

输出数据out2: [1, 0, 1, 0]

输入数据cond2 = [[True, False], [False, True]]
输入数据a2: [1, 2]
输出数据out3: [[1, 0], [0, 2]]
```

