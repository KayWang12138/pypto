# pypto.scatter\_<a name="ZH-CN_TOPIC_0000002484633470"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p124951335517"><a name="p124951335517"></a><a name="p124951335517"></a>x</p>
</td>
</tr>
<tr id="row173191727550"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1031982135514"><a name="p1031982135514"></a><a name="p1031982135514"></a><span id="ph731910215512"><a name="ph731910215512"></a><a name="ph731910215512"></a><term id="zh-cn_topic_0000001312391781_term4363218112215"><a name="zh-cn_topic_0000001312391781_term4363218112215"></a><a name="zh-cn_topic_0000001312391781_term4363218112215"></a>Ascend 310P</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p182521413105513"><a name="p182521413105513"></a><a name="p182521413105513"></a>x</p>
</td>
</tr>
<tr id="row4319162125515"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p9319172115519"><a name="p9319172115519"></a><a name="p9319172115519"></a><span id="ph23191215518"><a name="ph23191215518"></a><a name="ph23191215518"></a><term id="zh-cn_topic_0000001312391781_term71949488213"><a name="zh-cn_topic_0000001312391781_term71949488213"></a><a name="zh-cn_topic_0000001312391781_term71949488213"></a>Ascend 910</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p112553131552"><a name="p112553131552"></a><a name="p112553131552"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将src的值写入input中。写入位置由index张量指定。3维计算公式如下，其他维度以此类推：

![](figures/zh-cn_formulaimage_0000002516835553.png)

## 函数原型<a name="section8786125214915"></a>

```
scatter_(input: Tensor, dim: int, index: Tensor, src: float, *, reduce: str = None) -> Tensor
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
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a><span>待更新的Tensor</span>。</p>
<p id="p6203457185819"><a name="p6203457185819"></a><a name="p6203457185819"></a>支持的数据类型为：DT_FP32/DT_FP16。</p>
<p id="p1120416193145"><a name="p1120416193145"></a><a name="p1120416193145"></a>支持的维度：2-4维</p>
<p id="p14554937181812"><a name="p14554937181812"></a><a name="p14554937181812"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
<tr id="row113226117598"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p10322161115592"><a name="p10322161115592"></a><a name="p10322161115592"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p13432181975917"><a name="p13432181975917"></a><a name="p13432181975917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p202134810105"><a name="p202134810105"></a><a name="p202134810105"></a>指定用于索引的维度，支持input的维度范围内的任意维度。</p>
<p id="p1849583112558"><a name="p1849583112558"></a><a name="p1849583112558"></a>合法的维度索引 ，范围为：-input.dim 到 input.dim - 1。</p>
</td>
</tr>
<tr id="row2582163131918"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p11583183141914"><a name="p11583183141914"></a><a name="p11583183141914"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p155831337191"><a name="p155831337191"></a><a name="p155831337191"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1636319465116"><a name="p1636319465116"></a><a name="p1636319465116"></a><span>input的一组索引</span>。</p>
<p id="p163635464112"><a name="p163635464112"></a><a name="p163635464112"></a>支持的数据类型：INT64。</p>
<p id="p436304641114"><a name="p436304641114"></a><a name="p436304641114"></a>支持的维度：和input保持一致</p>
<p id="p18363104619119"><a name="p18363104619119"></a><a name="p18363104619119"></a>index 的每一维的 shape &lt;= input 的每一维的shape</p>
<p id="p1439641031212"><a name="p1439641031212"></a><a name="p1439641031212"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）</p>
</td>
</tr>
<tr id="row4416102014193"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p15416192021918"><a name="p15416192021918"></a><a name="p15416192021918"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1141642020193"><a name="p1141642020193"></a><a name="p1141642020193"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1990361514114"><a name="p1990361514114"></a><a name="p1990361514114"></a>src<span>是更新的标量值</span>。</p>
<p id="p390341515117"><a name="p390341515117"></a><a name="p390341515117"></a>支持的数据类型为：DT_FP32/DT_FP16。数据类型和 input 保持一致</p>
<p id="p169031915101117"><a name="p169031915101117"></a><a name="p169031915101117"></a>不支持输入INF/NAN。</p>
</td>
</tr>
<tr id="row12830195010149"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p383035021419"><a name="p383035021419"></a><a name="p383035021419"></a>reduce</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p15830105031411"><a name="p15830105031411"></a><a name="p15830105031411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1983045012143"><a name="p1983045012143"></a><a name="p1983045012143"></a><span>要应用的归约操作，支持 </span>'add'<span> 或 </span>'multiply'，不传参时默认为直接替换</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回更新后的 input，为inplace操作。

## 约束说明<a name="section753174210543"></a>

1. broadcast约束：input和index不支持broadcast；

2. input.shape的dim轴不可切，viewshape的维度与input维度相同，要求viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)，其余维度的形状大小不做限制；

3. input.shape的dim轴不可切，tileshape的维度与input维度相同，tileshape\[dim\] \>= viewshape\[dim\]，其余维度的形状大小不做限制，input index 和 result 都会在 UB 中，需满足所有输入和输出的 tileshape 大小总和不能超过UB内存的大小。

## 调用示例<a name="section642mcpsimp"></a>

-   将2维 input 根据2维index更新对应索引的值

    ```
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)
    ```

    结果示例如下：

    ```
    输入数据x:[[0 0 0 0 0],
               [0 0 0 0 0],
               [0 0 0 0 0]]
    输入数据y:[[1 2],
               [0 1]]
    输出数据o:[[2.0 0   0 0 0],
               [2.0 2.0 0 0 0],
               [0   2.0 0 0 0]]
    ```

