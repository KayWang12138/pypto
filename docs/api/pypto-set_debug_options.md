# pypto.set\_debug\_options<a name="ZH-CN_TOPIC_0000002497946066"></a>

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

设置debug的选项。

## 函数原型<a name="section1814166202715"></a>

```
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  ) -> None
```

## 参数说明<a name="section27141942204919"></a>

<a name="table332075643414"></a>
<table><thead align="left"><tr id="row173201156103418"><th class="cellrowborder" valign="top" width="22.439999999999998%" id="mcps1.1.4.1.1"><p id="p3552162123616"><a name="p3552162123616"></a><a name="p3552162123616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.51%" id="mcps1.1.4.1.2"><p id="p85521928363"><a name="p85521928363"></a><a name="p85521928363"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.05%" id="mcps1.1.4.1.3"><p id="p255272123616"><a name="p255272123616"></a><a name="p255272123616"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row15320125613420"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p388155714610"><a name="p388155714610"></a><a name="p388155714610"></a>compile_debug_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p586575610352"><a name="p586575610352"></a><a name="p586575610352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.05%" headers="mcps1.1.4.1.3 "><p id="p78711459714"><a name="p78711459714"></a><a name="p78711459714"></a>含义：设置编译阶段调试模式</p>
<p id="p1787110516718"><a name="p1787110516718"></a><a name="p1787110516718"></a>说明：0：代表默认不使能编译阶段调试模式；</p>
<p id="p1872555717"><a name="p1872555717"></a><a name="p1872555717"></a>1：代表图使能编译阶段调试模式，一键开启图编译相关配置，当前进包括计算图；</p>
<p id="p18721657710"><a name="p18721657710"></a><a name="p18721657710"></a>类型：int</p>
<p id="p10872175574"><a name="p10872175574"></a><a name="p10872175574"></a><span>取值范围：</span>0 或 1</p>
<p id="p3872351172"><a name="p3872351172"></a><a name="p3872351172"></a>默认值：0</p>
<p id="p108725510719"><a name="p108725510719"></a><a name="p108725510719"></a>影响Pass范围：NA</p>
</td>
</tr>
<tr id="row332055610344"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p2890201018711"><a name="p2890201018711"></a><a name="p2890201018711"></a>runtime_debug_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p88652563356"><a name="p88652563356"></a><a name="p88652563356"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.05%" headers="mcps1.1.4.1.3 "><p id="p15301116574"><a name="p15301116574"></a><a name="p15301116574"></a>含义：设置执行阶段调试模式</p>
<p id="p193011161774"><a name="p193011161774"></a><a name="p193011161774"></a>说明：0：代表默认不使能执行阶段调试模式；</p>
<p id="p9301916074"><a name="p9301916074"></a><a name="p9301916074"></a>1：代表使能执行阶段调试模式，一键开启图执行相关配置，当前仅包括泳道图；</p>
<p id="p143017161478"><a name="p143017161478"></a><a name="p143017161478"></a>类型：int</p>
<p id="p15301101619711"><a name="p15301101619711"></a><a name="p15301101619711"></a><span>取值范围：</span>0 或 1</p>
<p id="p43012162710"><a name="p43012162710"></a><a name="p43012162710"></a>默认值：0</p>
<p id="p1830111620712"><a name="p1830111620712"></a><a name="p1830111620712"></a>影响Pass范围：NA</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

void：Set方法无返回值。设置操作成功即生效。

## 约束说明<a name="section753174210543"></a>

无。

## 调用示例<a name="section646021318230"></a>

```
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
```

