# pypto.expand\_clone<a name="ZH-CN_TOPIC_0000002498347400"></a>

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

将输入 Tensor 在唯一等于 1 的轴上广播以匹配 shape，返回真实占内存的新 Tensor。

## 函数原型<a name="section19138102360"></a>

```
expand_clone(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
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
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p379043133217"><a name="p379043133217"></a><a name="p379043133217"></a>源操作数。</p>
<p id="p18790431163219"><a name="p18790431163219"></a><a name="p18790431163219"></a>支持的数据类型为：DT_BF16、DT_FP32、DT_FP16、DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32</p>
<p id="p879011319327"><a name="p879011319327"></a><a name="p879011319327"></a>不支持空Tensor，Shape仅支持2-4维，被广播的轴的Shape大小要为1，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
<tr id="row49032035143315"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p7903133516337"><a name="p7903133516337"></a><a name="p7903133516337"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p1090363511338"><a name="p1090363511338"></a><a name="p1090363511338"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p89031135183311"><a name="p89031135183311"></a><a name="p89031135183311"></a>源操作数。</p>
<p id="p2852102645914"><a name="p2852102645914"></a><a name="p2852102645914"></a>目标形状。</p>
<p id="p1384113655916"><a name="p1384113655916"></a><a name="p1384113655916"></a>Shape Size不大于INT32_MAX；Shape的维度需要与输入的一致，除被广播的轴外其他轴大小须与 input 的 shape对应相等。</p>
</td>
</tr>
<tr id="row32408386561"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p172411938185616"><a name="p172411938185616"></a><a name="p172411938185616"></a>valid_shape</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p12411638195616"><a name="p12411638195616"></a><a name="p12411638195616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p82411138145617"><a name="p82411138145617"></a><a name="p82411138145617"></a>关键字参数。</p>
<p id="p113610569361"><a name="p113610569361"></a><a name="p113610569361"></a>源操作数，用于定义输出张量的动态shape，关键字参数，用于动态图，静态图可以省略。</p>
<p id="p474318578457"><a name="p474318578457"></a><a name="p474318578457"></a>支持的类型为 List[SymbolicScalar]，List[int]</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1319454615394"></a>

返回输出Tensor，Tensor的数据类型和input相同，形状为 shape。

## 约束说明<a name="section9879248193914"></a>

1.  只能一维广播，输入 Tensor 被广播的轴的 形状大小要为1。
2.  关于 valid\_shape 的说明：

    在动态图场景中，假设Tensor input \[a,1\]  扩展到 \[a,5\]，并设置 ViewShape 为 \[a,2\]，框架会通过 pypto.loop 循环生成 \[a,2\] 分块，并按偏移量拼接。此时若未传入 valid\_shape，代码将默认生成全 \[a,2\] 的张量（如 pypto.expand\_clone\(input, \[a,2\]\)）。

    然而，当总尺寸 \[a,5\] 无法被分块尺寸 \[a,2\] 整除时，尾块的有效形状（如 \[a,1\]）无法由框架自动推导。例如，最后一列可能仅包含 1 个元素，而非完整的 \[a,2\] 分块。此时必须通过 valid\_shape 明确指定尾块的实际有效形状，如下：

    pypto.expand\_clone\(input, \[a,2\], valid\_shape = \[a, pypto.min\(2, 5 - 2 \* b\_idx\),\)

    其中b\_idx  表示循环索引。

## 调用示例<a name="section4127133461016"></a>

```
# static graph
a = pypto.tensor([1,8], pypto.DT_INT32)
out1 = pypto.expand_clone(a, [4,8])
# dynamic graph
out2 = pypto.expand_clone(a, [4,8], valid_shape = [pypto.symbolic_scalar(4), pypto.symbolic_scalar(8)])
```

结果示例如下：

```
输入数据a:     [[1, 2, 3, 4, 5, 6, 7, 8]]
输出数据out1:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
输出数据out2:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
```

