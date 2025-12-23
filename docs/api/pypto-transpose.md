# pypto.transpose<a name="ZH-CN_TOPIC_0000002470438116"></a>

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

返回一个张量，该张量是输入张量的转置版本。指定的维度 dim0 和 dim1 将被交换。

## 函数原型<a name="section19138102360"></a>

```
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
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
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a>源操作数，输入张量。</p>
<p id="p1872462420360"><a name="p1872462420360"></a><a name="p1872462420360"></a>支持的数据类型为：DT_FP32，DT_FP16，DT_BF16</p>
<p id="p14554937181812"><a name="p14554937181812"></a><a name="p14554937181812"></a>不支持空Tensor，Shape仅支持2-5维，且Shape Size不大于2147483647（即INT32_MAX）。</p>
<p id="p17426193952117"><a name="p17426193952117"></a><a name="p17426193952117"></a>算子对不同 Shape 支持不同，详见约束说明。</p>
</td>
</tr>
<tr id="row49032035143315"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p7903133516337"><a name="p7903133516337"></a><a name="p7903133516337"></a>dim0</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p1090363511338"><a name="p1090363511338"></a><a name="p1090363511338"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p89031135183311"><a name="p89031135183311"></a><a name="p89031135183311"></a>源操作数，要交换的第一个维度的索引，从0开始计数。</p>
</td>
</tr>
<tr id="row3240155531715"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p340165814173"><a name="p340165814173"></a><a name="p340165814173"></a>dim1</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p13401115861716"><a name="p13401115861716"></a><a name="p13401115861716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p6401135815176"><a name="p6401135815176"></a><a name="p6401135815176"></a>源操作数，要交换的第二个维度的索引，从0开始计数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26662142616"></a>

返回一个与输入数据类型一致的Tensor，其中 dim0 与 dim1 的维度位置被对调。

## 约束说明<a name="section156225416916"></a>

1. tileshape和输入input维度一致，用于切分input。

2.输入维度dim0，dim1 必须大于0，小于input维度。

3.当前Transpose实现存在约束，只能支持以下场景转置：

-   2维：任意轴
-   3维：任意轴
-   4维：支持：0 2，1 3，2 3, 1 2,  不支持：0 3,  0 1
-   5维：支持：3 4， 其他不支持

4.涉及尾轴转置的场景，需要预留一块临时空间，用来搬运。

示例：

input : \[a, b, c, d\]  tileshape为\[t0, t1, t2, t3\] 数据类型为DT\_FP32

dim0: 2

dim1: 3

预留的临时空间为：t0 \* t1 \* align\(t2, 16\) \* align\(t3, 32 / sizeof\(DT\_FP32\)\)

## 调用示例<a name="section4127133461016"></a>

```
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.transpose(x, 0, 1)
```

结果示例如下：

```
输入数据x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
输出数据y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```

