# pypto.index\_add\_<a name="ZH-CN_TOPIC_0000002491612364"></a>

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

将 source 的每一块数据乘以缩放因子_ alpha（默认为1） _加到 input 的相应数据块上，其中索引和数据块方向由 index 张量和 dim指定。

## 函数原型<a name="section8786125214915"></a>

```
index_add_(
    input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1
    ) -> Tensor
```

## 参数说明<a name="section644919345515"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p57634810538"><a name="p57634810538"></a><a name="p57634810538"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p1329911375318"><a name="p1329911375318"></a><a name="p1329911375318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p12845598191"><a name="p12845598191"></a><a name="p12845598191"></a>Tensor类型，目标操作数/源操作数；</p>
<p id="p16444423204"><a name="p16444423204"></a><a name="p16444423204"></a>支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT16，DT_INT32，形状支持2-4维。</p>
</td>
</tr>
<tr id="row17711174412319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p312611191228"><a name="p312611191228"></a><a name="p312611191228"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p712641982210"><a name="p712641982210"></a><a name="p712641982210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p161269197221"><a name="p161269197221"></a><a name="p161269197221"></a>int类型，加法作用到 input 的维度；</p>
<p id="p11261919202218"><a name="p11261919202218"></a><a name="p11261919202218"></a>支持任意不超过 input 维数的值。</p>
</td>
</tr>
<tr id="row1390311445352"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1121294213335"><a name="p1121294213335"></a><a name="p1121294213335"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p7212542203319"><a name="p7212542203319"></a><a name="p7212542203319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p1021204293318"><a name="p1021204293318"></a><a name="p1021204293318"></a>Tensor类型，源操作数，值代表 input 所在 dim 轴的索引；</p>
<p id="p14212174211338"><a name="p14212174211338"></a><a name="p14212174211338"></a>支持的数据类型为：DT_INT32，DT_INT64；</p>
<p id="p141711249112"><a name="p141711249112"></a><a name="p141711249112"></a>不支持空 Tensor，形状只支持1维，索引与 source 的 dim 轴索引一一对应，形状大小与 source 所在 dim 轴的形状大小相同。</p>
</td>
</tr>
<tr id="row144815011468"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p81831467319"><a name="p81831467319"></a><a name="p81831467319"></a>source</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p6183204610311"><a name="p6183204610311"></a><a name="p6183204610311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p33045240345"><a name="p33045240345"></a><a name="p33045240345"></a>Tensor类型，需要加到 input 的源操作数；</p>
<p id="p6183104613315"><a name="p6183104613315"></a><a name="p6183104613315"></a>形状支持2-4维，所在 dim 轴的形状大小与 index 相同，其他维度的形状大小与 input 相同。</p>
</td>
</tr>
<tr id="row347132517399"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p18471625133912"><a name="p18471625133912"></a><a name="p18471625133912"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p6471142517393"><a name="p6471142517393"></a><a name="p6471142517393"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p1655205975318"><a name="p1655205975318"></a><a name="p1655205975318"></a>标量，关键字参数；</p>
<p id="p147396412396"><a name="p147396412396"></a><a name="p147396412396"></a>表示累加时的缩放因子，默认为 1</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

原地操作返回 input

## 约束说明<a name="section753174210543"></a>

1. index 张量必须是整数类型（DT\_INT32 或 DT\_INT64），值不超过 input 在 dim 维度上的形状大小，维数为1，形状大小与 source 所在 dim 轴的形状大小相同；

2. dim 为 int 类型，取值范围  -input.shape.size <= dim < input.shape.size；

3. input 和 source 的数据类型和维数均相同；

4. input.shape和 source.shape的 dim 轴 viewshape 不可切，要求 viewshape\[dim\]\>=max\(input.shape\[dim\], source.shape\[dim\]\)，其余维度的形状大小不做限制；

4. tileshape的维度与 result 相同，用于切分 input 和 source，tileshape\[dim\] = viewshape\[dim\]，所有输入和输出的 tileshape 大小总和不能超过UB内存的大小。

## 调用示例<a name="section642mcpsimp"></a>

```
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
dim = 0
# use alpha
y = pypto.index_add_(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add_(x, dim, index, source)
```

结果示例如下：

```
输入数据 x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
输出数据 y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```

