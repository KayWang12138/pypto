# pypto.Tensor.get\_cache\_policy<a name="ZH-CN_TOPIC_0000002473170296"></a>

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

## 功能说明<a name="section698519423241"></a>

获取某个缓存策略是否被启用。

## 函数原型<a name="section15297165392417"></a>

```
get_cache_policy(self, policy: CachePolicy) -> bool
```

## 参数说明<a name="section32851234132519"></a>

<a name="table770022613375"></a>
<table><thead align="left"><tr id="row187011726173718"><th class="cellrowborder" valign="top" width="15.541554155415543%" id="mcps1.1.4.1.1"><p id="p1701122663712"><a name="p1701122663712"></a><a name="p1701122663712"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.731173117311732%" id="mcps1.1.4.1.2"><p id="p47011268375"><a name="p47011268375"></a><a name="p47011268375"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.72727272727272%" id="mcps1.1.4.1.3"><p id="p107011926163713"><a name="p107011926163713"></a><a name="p107011926163713"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row7701226183719"><td class="cellrowborder" valign="top" width="15.541554155415543%" headers="mcps1.1.4.1.1 "><p id="p1701172673715"><a name="p1701172673715"></a><a name="p1701172673715"></a>policy</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.1.4.1.2 "><p id="p370112610376"><a name="p370112610376"></a><a name="p370112610376"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.72727272727272%" headers="mcps1.1.4.1.3 "><p id="p6701162633717"><a name="p6701162633717"></a><a name="p6701162633717"></a>缓存策略类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section20685185616257"></a>

Cache策略是否被启用。

## 约束说明<a name="section58751342506"></a>

无。

## 调用示例<a name="section111321836263"></a>

```
t = pypto.tensor((16, 16), pypto.DT_FP32)
out = t.get_cache_policy(pypto.CachePolicy.PREFETCH)
```

结果示例如下：

```
输出数据out: False
```

