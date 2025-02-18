# pypto.set\_codegen\_options<a name="ZH-CN_TOPIC_0000002470438090"></a>

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

设置codegen的选项。

## 函数原型<a name="section1814166202715"></a>

```
set_codegen_options(*, support_dynamic_aligned: bool = None, codegen_expression_fusion: bool = None) -> None
```

## 参数说明<a name="section27141942204919"></a>

<a name="table1776912518297"></a>
<table><thead align="left"><tr id="row1876912592919"><th class="cellrowborder" valign="top" width="22.45%" id="mcps1.1.4.1.1"><p id="p20769759295"><a name="p20769759295"></a><a name="p20769759295"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.4.1.2"><p id="p1757193775415"><a name="p1757193775415"></a><a name="p1757193775415"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.03%" id="mcps1.1.4.1.3"><p id="p197075785418"><a name="p197075785418"></a><a name="p197075785418"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row776920512911"><td class="cellrowborder" valign="top" width="22.45%" headers="mcps1.1.4.1.1 "><p id="p472210351681"><a name="p472210351681"></a><a name="p472210351681"></a>support_dynamic_aligned</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p4744319542"><a name="p4744319542"></a><a name="p4744319542"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.03%" headers="mcps1.1.4.1.3 "><p id="p786581882820"><a name="p786581882820"></a><a name="p786581882820"></a><span>含义</span><span>：是否支持动态Shape。</span></p>
<p id="p113481910151614"><a name="p113481910151614"></a><a name="p113481910151614"></a>说明：</p>
<p id="p7211525192015"><a name="p7211525192015"></a><a name="p7211525192015"></a>当值为True，算子生成的设备侧二进制可支持动态Shape对齐场景。</p>
<p id="p2134455102117"><a name="p2134455102117"></a><a name="p2134455102117"></a>当值为False，算子生成的设备侧二进制仅支持处理动态Shape非对齐场景。</p>
<p id="p18641191414322"><a name="p18641191414322"></a><a name="p18641191414322"></a>类型：bool</p>
<p id="p15463132482813"><a name="p15463132482813"></a><a name="p15463132482813"></a><span>取值范围</span><span>：{</span>True<span>, </span>False<span>}</span></p>
<p id="p086510184291"><a name="p086510184291"></a><a name="p086510184291"></a>默认值：False（当算子确认动态Shape，且Shape尾轴均为对齐时，可尝试打开确认是否有性能收益）</p>
<p id="p1781153315296"><a name="p1781153315296"></a><a name="p1781153315296"></a>影响Pass范围：无，仅影响CodeGen模块生成设备侧目标代码</p>
</td>
</tr>
<tr id="row477075112916"><td class="cellrowborder" valign="top" width="22.45%" headers="mcps1.1.4.1.1 "><p id="p998624417810"><a name="p998624417810"></a><a name="p998624417810"></a>codegen_expression_fusion</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p97463165419"><a name="p97463165419"></a><a name="p97463165419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.03%" headers="mcps1.1.4.1.3 "><p id="p119995953119"><a name="p119995953119"></a><a name="p119995953119"></a><span>含义：</span>是否支持在设备侧执行动态表达式计算。</p>
<p id="p14586131115316"><a name="p14586131115316"></a><a name="p14586131115316"></a>说明：</p>
<p id="p1934615613114"><a name="p1934615613114"></a><a name="p1934615613114"></a>当值为True，算子生成的设备侧CCE二进制可支持动态表达式计算</p>
<p id="p13136612421"><a name="p13136612421"></a><a name="p13136612421"></a>当值为False，算子生成的设备侧CCE二进制不支持动态表达式计算</p>
<p id="p15277205318318"><a name="p15277205318318"></a><a name="p15277205318318"></a>类型：bool</p>
<p id="p5477813123119"><a name="p5477813123119"></a><a name="p5477813123119"></a>取值范围: <span>{</span>True<span>, </span>False<span>}</span></p>
<p id="p14429115103112"><a name="p14429115103112"></a><a name="p14429115103112"></a>默认值<span>：</span>False</p>
<p id="p67707572911"><a name="p67707572911"></a><a name="p67707572911"></a>影响Pass范围：无，仅影响Codegen模块生成设备侧CCE二进制代码</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

void：Set方法无返回值。设置操作成功即生效。

## 约束说明<a name="section753174210543"></a>

support\_dynamic\_aligned选项效果后续会通过Pass推导机制进行优化，无需用户手工设置并日落，建议用户谨慎使用。

## 调用示例<a name="section646021318230"></a>

```
pypto.set_codegen_options(support_dynamic_aligned=True, codegen_expression_fusion=True)
```

