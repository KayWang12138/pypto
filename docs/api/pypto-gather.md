# pypto.gather<a name="ZH-CN_TOPIC_0000002470278126"></a>

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

对输入的 input，按照指定维度 dim 和索引 index 提取原始 Tensor 的对应值，最后返回结果。例如对3维 Tensor，有以下计算公式：

![](figures/zh-cn_formulaimage_0000002509262495.png)

## 函数原型<a name="section8786125214915"></a>

```
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## 参数说明<a name="section644919345515"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p57634810538"><a name="p57634810538"></a><a name="p57634810538"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1329911375318"><a name="p1329911375318"></a><a name="p1329911375318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p12845598191"><a name="p12845598191"></a><a name="p12845598191"></a>源操作数。</p>
<p id="p16444423204"><a name="p16444423204"></a><a name="p16444423204"></a>支持的数据类型为：DT_FP32，DT_FP16，DT_INT16，DT_INT32。</p>
<p id="p6429205613196"><a name="p6429205613196"></a><a name="p6429205613196"></a>不支持空 Tensor，形状支持2-4维，且 shape size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
<tr id="row144815011468"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1189125195515"><a name="p1189125195515"></a><a name="p1189125195515"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1489120525520"><a name="p1489120525520"></a><a name="p1489120525520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p33866304552"><a name="p33866304552"></a><a name="p33866304552"></a>源操作数。</p>
<p id="p1849583112558"><a name="p1849583112558"></a><a name="p1849583112558"></a>支持任意合法的维度索引 ，范围为：-input.dim 到 input.dim - 1。</p>
</td>
</tr>
<tr id="row3750832163314"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1121294213335"><a name="p1121294213335"></a><a name="p1121294213335"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p7212542203319"><a name="p7212542203319"></a><a name="p7212542203319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1021204293318"><a name="p1021204293318"></a><a name="p1021204293318"></a>源操作数。</p>
<p id="p14212174211338"><a name="p14212174211338"></a><a name="p14212174211338"></a>支持的数据类型为：DT_INT32，DT_INT64。</p>
<p id="p141711249112"><a name="p141711249112"></a><a name="p141711249112"></a>不支持空 Tensor，形状支持2-4维，需保证 index 所有轴上的形状大小不超过 input 的对应形状大小，且值为合法索引，即不超过 input 在 dim 轴上的形状大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回输出 Tensor ，输出 Tensor 数据类型与 input 数据类型保持一致；输出 Tensor 的Shape 与 index 的 Shape 相同。

## 约束说明<a name="section753174210543"></a>

1. index 形状的维数与 input 的维数相同，且所有轴上的形状大小不超过 input 的形状大小；值不能超出 input 在 dim 维的形状大小；

2. dim: -input.dim <= dim < input.dim；

3. input.shape 的 dim 轴不可切，要求 viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)，其余维度的形状大小不做限制；

4. tileshape的维度与 result 相同，用于切分 result 和 index，tileshape\[dim\] = viewshape\[dim\]，所有输入和输出的 tileshape 大小总和不能超过UB内存的大小。

## 调用示例<a name="section642mcpsimp"></a>

```
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

结果示例如下：

```
输入数据 x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
输出数据 y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```

