# pypto.arange<a name="ZH-CN_TOPIC_0000002437346096"></a>

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

创建长度为![](figures/zh-cn_formulaimage_0000002498843114.png)  的一维张量，包含区间 \[start, end\) 内、以 step 为步长的等差数列。

## 函数原型<a name="section19138102360"></a>

```
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## 参数说明<a name="section75724101161"></a>

<a name="table118012071618"></a>
<table><thead align="left"><tr id="row138013719117"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="p138011779117"><a name="p138011779117"></a><a name="p138011779117"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.1.4.1.2"><p id="p19801117612"><a name="p19801117612"></a><a name="p19801117612"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.1.4.1.3"><p id="p28021710113"><a name="p28021710113"></a><a name="p28021710113"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row98021671918"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p11802272012"><a name="p11802272012"></a><a name="p11802272012"></a>start</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p68020711116"><a name="p68020711116"></a><a name="p68020711116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p198021771412"><a name="p198021771412"></a><a name="p198021771412"></a>源操作数。</p>
<p id="p28026710111"><a name="p28026710111"></a><a name="p28026710111"></a>支持的数据类型为：int32 , float32</p>
<p id="p10716230185415"><a name="p10716230185415"></a><a name="p10716230185415"></a>默认值为 0。</p>
</td>
</tr>
<tr id="row9802171212"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p16802572016"><a name="p16802572016"></a><a name="p16802572016"></a>end</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p980220714116"><a name="p980220714116"></a><a name="p980220714116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p1680297016"><a name="p1680297016"></a><a name="p1680297016"></a>源操作数。</p>
<p id="p168021571114"><a name="p168021571114"></a><a name="p168021571114"></a>支持的数据类型为：int32 ,  float32。</p>
<p id="p4981124435511"><a name="p4981124435511"></a><a name="p4981124435511"></a>该参数不能省略。</p>
</td>
</tr>
<tr id="row18802177510"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p2802271414"><a name="p2802271414"></a><a name="p2802271414"></a>step</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p98021873110"><a name="p98021873110"></a><a name="p98021873110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p128021270113"><a name="p128021270113"></a><a name="p128021270113"></a>源操作数。</p>
<p id="p1180116172075"><a name="p1180116172075"></a><a name="p1180116172075"></a>支持的数据类型为：int32 ,  float32</p>
<p id="p9871205085512"><a name="p9871205085512"></a><a name="p9871205085512"></a>默认值为 1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section13199132819"></a>

返回一维输出Tensor，若输入值存在float数据类型，则输出Tensor数据类型为float，否则为int。

## 约束说明<a name="section1949613215911"></a>

1. step不能为0，作为浮点数，abs\(step\)\>1e-8;

2. \(end-start\)/step需大于0;

3. 如果 start, end, step 均为 int 输入，则三者均不能超出 int32 范围

4. tileshape和输出output维度一致，均为一维，用于切分output

## 调用示例<a name="section6547114511153"></a>

```
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

结果示例如下：

```
输出数据y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
输出数据y2: [1.0, 2.0, 3.0]
输出数据y3: [0, 1, 2, 3]
```

