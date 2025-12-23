# pypto.experimental.load<a name="ZH-CN_TOPIC_0000002492397340"></a>

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

## 功能说明<a name="section112531620115513"></a>

该接口为定制接口，约束较多。不保证稳定性。

从 GM（Global Memory）中根据源数据以及在源数据中的偏移列表将所需数据离散的加载到  UB（Unified Buffer） 中。

## 函数原型<a name="section13445759105712"></a>

```
load(src: Tensor, offsets: Tensor):
```

参数说明

<a name="zh-cn_topic_0146324969_table29998725"></a>
<table><thead align="left"><tr id="zh-cn_topic_0146324969_row8953505"><th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0146324969_p54145286"><a name="zh-cn_topic_0146324969_p54145286"></a><a name="zh-cn_topic_0146324969_p54145286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0146324969_p23692060"><a name="zh-cn_topic_0146324969_p23692060"></a><a name="zh-cn_topic_0146324969_p23692060"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.7%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0146324969_p19480441"><a name="zh-cn_topic_0146324969_p19480441"></a><a name="zh-cn_topic_0146324969_p19480441"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a>源操作数。</p>
<p id="p1872462420360"><a name="p1872462420360"></a><a name="p1872462420360"></a>支持的数据类型为：DT_INT32、DT_FP32、DT_FP16等。</p>
<p id="p517314131419"><a name="p517314131419"></a><a name="p517314131419"></a>不支持空Tensor，支持任意维度。</p>
</td>
</tr>
<tr id="row148718148574"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p138721410572"><a name="p138721410572"></a><a name="p138721410572"></a>offsets</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p1787014105716"><a name="p1787014105716"></a><a name="p1787014105716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p687151410574"><a name="p687151410574"></a><a name="p687151410574"></a>待加载数据在源操作数一维视图下的偏移</p>
<p id="p143281141171719"><a name="p143281141171719"></a><a name="p143281141171719"></a>支持的数据类型为DT_INT32、DT_INT64</p>
<p id="p1725416441574"><a name="p1725416441574"></a><a name="p1725416441574"></a>不支持空Tensor，支持两维或三维。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回输出Tensor，Tensor的数据类型和src相同，Shape和offsets相同。

## 调用示例<a name="section4127133461016"></a>

```
src = pypto.tensor([3, 2], pypto.DT_FP32)
offsets = pypto.tensor([2, 3], pypto.DT_INT32)
dst = pypto.load(src, offsets)
```

结果示例如下：

```
输入数据src: [[2.2, 3],
              [7.3, 1],
              [-1.5, 100.2]]
输入数据src的一维视图：[2.2, 3, 7.3, 1, -1.5, 100.2]

输入数据offsets: [[2, 5, 3],
                  [1, 2, 0]]
输出数据dst: [[7.3, 100.2, 1],
              [3, 7.3, 2.2]]
```

