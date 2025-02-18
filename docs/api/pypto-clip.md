# pypto.clip<a name="ZH-CN_TOPIC_0000002484633474"></a>

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

对输入 Tensor 进行数据裁剪，裁剪到指定的最小值到最大值范围内，小于最小值的位置替换为最小值，大于最大值的位置替换为最大值，其余值维持不变。该接口非原地操作，不改变输入张量，而是返回一个新的张量作为输出。

## 函数原型<a name="section8786125214915"></a>

```
clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
)-> Tensor
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
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p4131323173018"><a name="p4131323173018"></a><a name="p4131323173018"></a>源操作数。</p>
<p id="p1985220257305"><a name="p1985220257305"></a><a name="p1985220257305"></a>支持的数据类型为：DT_FP32，DT_FP16，DT_INT32, DT_INT16。</p>
<p id="p45217198302"><a name="p45217198302"></a><a name="p45217198302"></a>不支持空Tensor，数据维度大小仅支持2-4维，元素个数不超过 UINT32_MAX。</p>
</td>
</tr>
<tr id="row1840662672216"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p8407172642220"><a name="p8407172642220"></a><a name="p8407172642220"></a>min</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p134081826102211"><a name="p134081826102211"></a><a name="p134081826102211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p10408162618229"><a name="p10408162618229"></a><a name="p10408162618229"></a>标量或张量，可缺省，默认值为 -INF。当作为标量且为 float 类型时会自动转换为 DT_FLOAT32 的 Element，为 int 类型时会自动转换为 DT_INT32 的 Element，其余的类型可以通过 Element 构造。</p>
<p id="p185219549912"><a name="p185219549912"></a><a name="p185219549912"></a>输入为张量时，min  必须满足可以广播到与输入 input一致的形状；</p>
<p id="p16867154718813"><a name="p16867154718813"></a><a name="p16867154718813"></a>支持的数据类型为：DT_FP32，DT_FP16，DT_INT32，DT_INT16。</p>
<p id="p81471449488"><a name="p81471449488"></a><a name="p81471449488"></a>暂不支持 INT32/INT16 模式下以 NaN，INF，-INF 作为输入。</p>
</td>
</tr>
<tr id="row121224712228"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p2013124713226"><a name="p2013124713226"></a><a name="p2013124713226"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1413174732215"><a name="p1413174732215"></a><a name="p1413174732215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1264894021015"><a name="p1264894021015"></a><a name="p1264894021015"></a>标量或张量，可缺省，默认值为 inf。当作为标量且为 float 类型时会自动转换为 DT_FLOAT32 的 Element，为 int 类型时会自动转换为 DT_INT32 的 Element，其余的类型可以通过 Element 构造。</p>
<p id="p15648134011103"><a name="p15648134011103"></a><a name="p15648134011103"></a>输入为张量时，max  必须满足可以广播到与输入 input一致的形状；</p>
<p id="p86484402103"><a name="p86484402103"></a><a name="p86484402103"></a>支持的数据类型为： DT_FP32，DT_FP16，DT_INT32, DT_INT16。</p>
<p id="p176481640171020"><a name="p176481640171020"></a><a name="p176481640171020"></a>暂不支持 INT32/INT16 模式下以 NaN，INF，-INF 作为输入。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

当输入为标量时，输出为：

![](figures/zh-cn_formulaimage_0000002523677905.png)

当输入为张量时，输出为：

![](figures/zh-cn_formulaimage_0000002523637951.png)

输出张量的数据类型和输入 input 相同。

当 min / max 其中一者为 NAN 时，输出结果为 NAN。

当 min \> max 时，输出结果对应位置均为 max 的值。

## 约束说明<a name="section753174210543"></a>

min / max 为张量类型时，其形状大小必须满足可以广播到输入的形状。且各输入的元素个数不超过 UINT32\_MAX，输入的数据类型必须为 DT\_FP32/DT\_FP16/DT\_INT32/DT\_INT16 中的一种，并且必须与输入 input 的数据类型一致。min / max 的类型必须一致，同时为 Element 或同时为 Tensor。

## 调用示例<a name="section642mcpsimp"></a>

```
x = pypto.tensor([2,3], pypto.DT_INT32)
min = pypto.tensor([2,3], pypto.DT_INT32)
max = pypto.tensor([2,3], pypto.DT_INT32)
out = pypto.clip(x,min,max)
```

结果示例如下：

```
输入数据 self: [[-2 1 2], [3 4 5]]
输入数据 min: [[-1 0 2], [0 3 5]] 
输入数据 max: [[1 2  1], [4 4 4]]
输出数据 out: [[-1 1 1], [3 4 4]]
```

示例 2：

```
x = pypto.tensor([2,3], pypto.DT_INT32)
min = 1
max = 3
out = pypto.clip(x,min,max)
```

结果示例如下：

```
输入数据 x: [[0 2 4], [3 4 6]] 
输出数据 out: [[1 2 3], [3 3 3]]
```

