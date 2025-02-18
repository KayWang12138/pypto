# pypto.full<a name="ZH-CN_TOPIC_0000002470278132"></a>

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

创建一个大小为 size、填充值为 fill\_value 的张量。张量的数据类型为dtype。

## 函数原型<a name="section19138102360"></a>

```
full(size: List[int], fill_value: Union[int, float, Element], dtype: DataType, *, valid_shape: Optional[Union[List[int], List[SymbolicScalar]]] = None ) -> Tensor
```

## 参数说明<a name="section75724101161"></a>

<a name="zh-cn_topic_0146324969_table29998725"></a>
<table><thead align="left"><tr id="zh-cn_topic_0146324969_row8953505"><th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0146324969_p54145286"><a name="zh-cn_topic_0146324969_p54145286"></a><a name="zh-cn_topic_0146324969_p54145286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.83%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0146324969_p23692060"><a name="zh-cn_topic_0146324969_p23692060"></a><a name="zh-cn_topic_0146324969_p23692060"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.88%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0146324969_p19480441"><a name="zh-cn_topic_0146324969_p19480441"></a><a name="zh-cn_topic_0146324969_p19480441"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="9.83%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a>源操作数，用于定义输出张量的形状。</p>
<p id="p154988385105"><a name="p154988385105"></a><a name="p154988385105"></a>支持的数据类型为：List[int]</p>
</td>
</tr>
<tr id="row223471701214"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p223431781217"><a name="p223431781217"></a><a name="p223431781217"></a>fill_value</p>
</td>
<td class="cellrowborder" valign="top" width="9.83%" headers="mcps1.1.4.1.2 "><p id="p2023451741214"><a name="p2023451741214"></a><a name="p2023451741214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.1.4.1.3 "><p id="p19234617121215"><a name="p19234617121215"></a><a name="p19234617121215"></a>源操作数，用于填充输出张量的值。</p>
<p id="p1850195211342"><a name="p1850195211342"></a><a name="p1850195211342"></a>支持的数据类型为：int ，float，Element</p>
<p id="p343795110720"><a name="p343795110720"></a><a name="p343795110720"></a>当传入 Element ，支持DT_FP32, DT_INT32。</p>
<p id="p6210192754318"><a name="p6210192754318"></a><a name="p6210192754318"></a>输入需要和 dtype 类型相同，不支持隐式转化。</p>
</td>
</tr>
<tr id="row6418155819368"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p441815582368"><a name="p441815582368"></a><a name="p441815582368"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="9.83%" headers="mcps1.1.4.1.2 "><p id="p104187585367"><a name="p104187585367"></a><a name="p104187585367"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.1.4.1.3 "><p id="p441817581365"><a name="p441817581365"></a><a name="p441817581365"></a>源操作数，用于定义输出张量的类型</p>
<p id="p9429191873718"><a name="p9429191873718"></a><a name="p9429191873718"></a>支持的数据类型为：DT_FP32, DT_INT32。</p>
<p id="p294545744314"><a name="p294545744314"></a><a name="p294545744314"></a>输入需要和 fill_value 类型相同，不支持隐式转化</p>
</td>
</tr>
<tr id="row15353564367"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p936115643619"><a name="p936115643619"></a><a name="p936115643619"></a>valid_shape</p>
</td>
<td class="cellrowborder" valign="top" width="9.83%" headers="mcps1.1.4.1.2 "><p id="p136115617364"><a name="p136115617364"></a><a name="p136115617364"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.1.4.1.3 "><p id="p113610569361"><a name="p113610569361"></a><a name="p113610569361"></a>源操作数，用于定义输出张量的动态shape，关键字参数，用于动态图，静态图可以省略</p>
<p id="p474318578457"><a name="p474318578457"></a><a name="p474318578457"></a>支持的类型为 List[SymbolicScalar]，List[int]</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26662142616"></a>

返回输出Tensor，Tensor的数据类型和dtype相同，Shape为size大小，全部的值为fill\_value。

## 约束说明<a name="section0564179125714"></a>

1.  valid\_shape 用于动态图场景。

    在动态图场景中，若需生成 \[5,5\] 的 Tensor 并设置 ViewShape 为 \[2,2\]，框架会通过 pypto.loop 循环生成 \[2,2\] 分块，并按偏移量拼接。此时若未传入 valid\_shape，代码将默认生成全 \[2,2\] 的张量（如 pypto.full\(\[2,2\], 1, pypto.DT\_INT32\)）。

    然而，当总尺寸 \[5,5\] 无法被分块尺寸 \[2,2\] 整除时，尾块的有效形状（如 \[1,1\]）无法由框架自动推导。例如，最后一行/列可能仅包含 1 个元素，而非完整的 \[2,2\] 分块。此时必须通过 valid\_shape 明确指定尾块的实际有效形状，如下：

    pypto.full\(\[2, 2\], 1, pypto.DT\_INT32, valid\_shape=\[

pypto.min\(2, 5 - 2 \* b\_idx\), pypto.min\(2, 5 - 2 \* s\_idx\)\]\), 其中b\_idx 和 s\_idx 表示循环索引。

## 调用示例<a name="section4127133461016"></a>

```
# Valid shapes use keyword argument
x1 = 1.0 # must be 1.0; implicit conversion is not supported
y1 = pypto.full([2,2], x1, pypto.DT_FP32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

x2 = pypto.Element(pypto.DT_INT32,1)
y2 = pypto.full([2,2], x2, pypto.DT_INT32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

# In static graphs, validshape can be ignored
x3 = pypto.Element(pypto.DT_INT32,1)
y3 = pypto.full([2,2], x3, pypto.DT_INT32)
```

结果示例如下：

```
y1输出数据: [[1.0,1.0], [1.0,1.0]]
y2输出数据: [[1,1],[1,1]]
y3输出数据: [[1,1], [1,1]
```

