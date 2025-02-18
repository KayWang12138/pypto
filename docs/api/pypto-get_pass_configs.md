# pypto.get\_pass\_configs<a name="ZH-CN_TOPIC_0000002520262377"></a>

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

获取指定Pass的全部配置信息。

## 函数原型<a name="section1814166202715"></a>

```
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## 参数说明<a name="section27141942204919"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p137871810171817"><a name="p137871810171817"></a><a name="p137871810171817"></a>strategy</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p16787110131818"><a name="p16787110131818"></a><a name="p16787110131818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p117871310161812"><a name="p117871310161812"></a><a name="p117871310161812"></a>Pass 策略名称，如"PVC2_OOO"</p>
</td>
</tr>
<tr id="row6253333118"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p57776143184"><a name="p57776143184"></a><a name="p57776143184"></a>identifier</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p577721410180"><a name="p577721410180"></a><a name="p577721410180"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p5777101431819"><a name="p5777101431819"></a><a name="p5777101431819"></a>Pass 名称，如"ExpandFunction"</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

PassConfigs 对象，该对象含有以下属性，属性只读：

<a name="table62551622182914"></a>
<table><thead align="left"><tr id="row425592211296"><th class="cellrowborder" valign="top" width="33.300000000000004%" id="mcps1.1.3.1.1"><p id="p2255152272912"><a name="p2255152272912"></a><a name="p2255152272912"></a>属性</p>
</th>
<th class="cellrowborder" valign="top" width="66.7%" id="mcps1.1.3.1.2"><p id="p112558227292"><a name="p112558227292"></a><a name="p112558227292"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row13255132214292"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p207692488296"><a name="p207692488296"></a><a name="p207692488296"></a>printFunction</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p325522211292"><a name="p325522211292"></a><a name="p325522211292"></a>dump计算图ir</p>
</td>
</tr>
<tr id="row625552216293"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p16255132220298"><a name="p16255132220298"></a><a name="p16255132220298"></a>dumpGraph</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p1125532262916"><a name="p1125532262916"></a><a name="p1125532262916"></a>dump 计算图</p>
</td>
</tr>
<tr id="row271732173017"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p2733283016"><a name="p2733283016"></a><a name="p2733283016"></a>dumpPassTimeCost</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p97143211306"><a name="p97143211306"></a><a name="p97143211306"></a>dump Pass耗时</p>
</td>
</tr>
<tr id="row51045502307"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p1010465053015"><a name="p1010465053015"></a><a name="p1010465053015"></a>preCheck</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p11104115083019"><a name="p11104115083019"></a><a name="p11104115083019"></a>Pass执行前进行校验</p>
</td>
</tr>
<tr id="row1321775314300"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p15217155363010"><a name="p15217155363010"></a><a name="p15217155363010"></a>postCheck</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p16217175316309"><a name="p16217175316309"></a><a name="p16217175316309"></a>Pass执行后进行校验</p>
</td>
</tr>
<tr id="row181501293116"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p1515013253115"><a name="p1515013253115"></a><a name="p1515013253115"></a>disablePass</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p0969445123119"><a name="p0969445123119"></a><a name="p0969445123119"></a>不执行当前Pass</p>
</td>
</tr>
<tr id="row76631048313"><td class="cellrowborder" valign="top" width="33.300000000000004%" headers="mcps1.1.3.1.1 "><p id="p76631546312"><a name="p76631546312"></a><a name="p76631546312"></a>healthCheck</p>
</td>
<td class="cellrowborder" valign="top" width="66.7%" headers="mcps1.1.3.1.2 "><p id="p11663144123113"><a name="p11663144123113"></a><a name="p11663144123113"></a>执行健康检查并生成报告</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section753174210543"></a>

无

## 调用示例<a name="section646021318230"></a>

```
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```

