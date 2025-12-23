# pypto.pass\_verify\_save<a name="ZH-CN_TOPIC_0000002532012867"></a>

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

在精度调试 Verify 特性使能时，使用该接口保存指定 Tensor 模拟计算的结果到数据文件。

## 函数原型<a name="section1814166202715"></a>

```
pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None
```

## 参数说明<a name="section27141942204919"></a>

<a name="table332075643414"></a>
<table><thead align="left"><tr id="row173201156103418"><th class="cellrowborder" valign="top" width="20.75%" id="mcps1.1.5.1.1"><p id="p3552162123616"><a name="p3552162123616"></a><a name="p3552162123616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="6.950000000000001%" id="mcps1.1.5.1.2"><p id="p85521928363"><a name="p85521928363"></a><a name="p85521928363"></a>Python 类型</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.5.1.3"><p id="p25261435213"><a name="p25261435213"></a><a name="p25261435213"></a>框架默认值</p>
</th>
<th class="cellrowborder" valign="top" width="64.78%" id="mcps1.1.5.1.4"><p id="p255272123616"><a name="p255272123616"></a><a name="p255272123616"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row15320125613420"><td class="cellrowborder" valign="top" width="20.75%" headers="mcps1.1.5.1.1 "><p id="p10780849134914"><a name="p10780849134914"></a><a name="p10780849134914"></a>*values</p>
</td>
<td class="cellrowborder" valign="top" width="6.950000000000001%" headers="mcps1.1.5.1.2 "><p id="p586575610352"><a name="p586575610352"></a><a name="p586575610352"></a>List[Union[pypto.Tensor, pypto.SymbolicScalar, OTHER_PRINTABLE]]</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.3 "><p id="p152654105210"><a name="p152654105210"></a><a name="p152654105210"></a>None</p>
</td>
<td class="cellrowborder" valign="top" width="64.78%" headers="mcps1.1.5.1.4 "><p id="p354484515316"><a name="p354484515316"></a><a name="p354484515316"></a>总体使能开关，决定所有 *pass_verify_* 选项、接口是否有效</p>
</td>
</tr>
<tr id="row332055610344"><td class="cellrowborder" valign="top" width="20.75%" headers="mcps1.1.5.1.1 "><p id="p111319374195"><a name="p111319374195"></a><a name="p111319374195"></a>cond</p>
</td>
<td class="cellrowborder" valign="top" width="6.950000000000001%" headers="mcps1.1.5.1.2 "><p id="p88652563356"><a name="p88652563356"></a><a name="p88652563356"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.3 "><p id="p9526144195211"><a name="p9526144195211"></a><a name="p9526144195211"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="64.78%" headers="mcps1.1.5.1.4 "><p id="p36709333545"><a name="p36709333545"></a><a name="p36709333545"></a>配置是否需要将模拟计算的 Tensor 数据存盘</p>
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
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```

