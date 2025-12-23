# pypto.assemble<a name="ZH-CN_TOPIC_0000002503198071"></a>

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

以offsets指定的out索引位置为基准，将输入张量input赋值到输出张量out的对应区域。

## 函数原型<a name="section19138102360"></a>

```
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None
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
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p135254139556"><a name="p135254139556"></a><a name="p135254139556"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p275718385618"><a name="p275718385618"></a><a name="p275718385618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p12268430182414"><a name="p12268430182414"></a><a name="p12268430182414"></a>源操作数。</p>
<p id="p1872462420360"><a name="p1872462420360"></a><a name="p1872462420360"></a>支持的数据类型为：PyPto支持的数据类型</p>
<p id="p14554937181812"><a name="p14554937181812"></a><a name="p14554937181812"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0146324969_row46369059"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p67301442195511"><a name="p67301442195511"></a><a name="p67301442195511"></a>offsets</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p6282445566"><a name="p6282445566"></a><a name="p6282445566"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p10730184295514"><a name="p10730184295514"></a><a name="p10730184295514"></a>相对于目标输出的偏移。</p>
<p id="p4811184320458"><a name="p4811184320458"></a><a name="p4811184320458"></a>需要保证offsets小于out的Shape。</p>
</td>
</tr>
<tr id="row168414546314"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p1949314584820"><a name="p1949314584820"></a><a name="p1949314584820"></a>out</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p84932534819"><a name="p84932534819"></a><a name="p84932534819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p187667291282"><a name="p187667291282"></a><a name="p187667291282"></a>目的操作数。</p>
<p id="p1076632912283"><a name="p1076632912283"></a><a name="p1076632912283"></a>支持的数据类型为：PyPto支持的数据类型</p>
<p id="p2766929112819"><a name="p2766929112819"></a><a name="p2766929112819"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26662142616"></a>

无返回值，会直接对out进行修改。

## 约束说明<a name="section2581910105618"></a>

无

## 调用示例<a name="section4127133461016"></a>

```
x = pypto.tensor([2, 2], pypto.data_type.DT_FP32) 
out = pypto.tensor([4, 4], pypto.data_type.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out) 
```

结果示例如下：

```
输出数据x: [[1, 1]
            [1, 1]]
输入数据out: [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
输出数据out: [[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
```

