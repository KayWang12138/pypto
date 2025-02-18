# pypto.experimental.set\_operation\_config \(赵俊涵\)<a name="ZH-CN_TOPIC_0000002496947482"></a>

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

该接口是编译框架提供的运行时动态配置管理功能的核心部分，它将原本静态地写在配置文件tile\_fwk\_config.json中的参数转变为动态、可编程的指令。

## 函数原型<a name="section1814166202715"></a>

```
set_operation_config(*, force_combine_axis: bool)
```

## 参数说明<a name="section27141942204919"></a>

<a name="table1776912518297"></a>
<table><thead align="left"><tr id="row1876912592919"><th class="cellrowborder" valign="top" width="22.470000000000002%" id="mcps1.1.4.1.1"><p id="p1778818307581"><a name="p1778818307581"></a><a name="p1778818307581"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.4.1.2"><p id="p47881330125814"><a name="p47881330125814"></a><a name="p47881330125814"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.00999999999999%" id="mcps1.1.4.1.3"><p id="p20788153075815"><a name="p20788153075815"></a><a name="p20788153075815"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row776920512911"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p1476912514292"><a name="p1476912514292"></a><a name="p1476912514292"></a>force_combine_axis</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p1029132613580"><a name="p1029132613580"></a><a name="p1029132613580"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p786581882820"><a name="p786581882820"></a><a name="p786581882820"></a><span>含义</span><span>：在代码生成阶段进行合轴优化。</span></p>
<p id="p991914515584"><a name="p991914515584"></a><a name="p991914515584"></a>说明：<span>例如 Reduce 输出为 (32, 1) 这类场景，为实现数据连续搬运，将其合轴为 (1, 32)，以支撑后续 ElementWise 算子随路 Broadcast 计算的优化。</span></p>
<p id="p768611365517"><a name="p768611365517"></a><a name="p768611365517"></a>类型<strong id="b3299143752310"><a name="b3299143752310"></a><a name="b3299143752310"></a>：</strong>bool</p>
<p id="p15463132482813"><a name="p15463132482813"></a><a name="p15463132482813"></a><span>取值范围</span><span>：{</span>True<span>, </span>False<span>}</span></p>
<p id="p086510184291"><a name="p086510184291"></a><a name="p086510184291"></a>默认值：False</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15631291597"></a>

void：Set方法无返回值。设置操作成功即生效。

## 约束说明<a name="section163179296"></a>

-   类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
-   作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例<a name="section646021318230"></a>

```
pypto.set_operation_config(force_combine_axis=True)
```

