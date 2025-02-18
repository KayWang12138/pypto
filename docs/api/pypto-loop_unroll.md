# pypto.loop\_unroll<a name="ZH-CN_TOPIC_0000002515464451"></a>

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

pypto.loop\_unroll是一个支持循环展开的循环迭代器函数，功能与pypto.loop类似，增加了unroll\_list参数支持多个展开方式。

## 函数原型<a name="section1814166202715"></a>

```
loop_unroll(*args, **kwargs) -> Iterator[Tuple[SymInt, int]]
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
<tbody><tr id="row8721417018"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p6831418020"><a name="p6831418020"></a><a name="p6831418020"></a>*args</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p138161420018"><a name="p138161420018"></a><a name="p138161420018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p182144011"><a name="p182144011"></a><a name="p182144011"></a>三个可选参数，分别为循环起始值（start），循环结束值（stop），循环步长（step），有以下三种写法：</p>
<p id="p929314019381"><a name="p929314019381"></a><a name="p929314019381"></a>单参数形式：stop(SymInt)，起始值默认为0，步长默认为1。等价于：loop_unroll(0, stop, 1)</p>
<p id="p11734715144018"><a name="p11734715144018"></a><a name="p11734715144018"></a>双参数形式：start(SymInt)，stop(SymInt)，等价于loop_unroll(start, stop, 1)</p>
<p id="p520932015403"><a name="p520932015403"></a><a name="p520932015403"></a>三参数形式：start (SymInt)，stop(SymInt)，step(SymInt)，等价于loop_unroll(start, stop, step)</p>
</td>
</tr>
<tr id="row542411593443"><td class="cellrowborder" rowspan="4" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1442514595447"><a name="p1442514595447"></a><a name="p1442514595447"></a>**kwargs</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1742535974414"><a name="p1742535974414"></a><a name="p1742535974414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p06152115916"><a name="p06152115916"></a><a name="p06152115916"></a>name(str)：循环标识名称，默认生成f"loop_{loop_idx}"。</p>
</td>
</tr>
<tr id="row141811247173816"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p18182194723818"><a name="p18182194723818"></a><a name="p18182194723818"></a>idx_name(str):  循环索引变量的名称，默认生成f"loop_idx_{loop_idx}"。</p>
</td>
</tr>
<tr id="row128646446382"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p186554411383"><a name="p186554411383"></a><a name="p186554411383"></a>unroll_list(Set[int]):  需要展开unroll的循环层数集合，默认为空集合。loop会提供等于该集合长度的几种展开方式，展开次数为n时，循环步长会变成step*n，每次迭代会执行n次循环体。每种展开次数会生成不同的代码路径。</p>
</td>
</tr>
<tr id="row5258194103818"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p9259144113383"><a name="p9259144113383"></a><a name="p9259144113383"></a>submit_before_loop(bool):  是否在循环开始前提交计算，默认为False。开启后会在循环开启前强制提交当前累积的计算任务到AICore执行。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回一个迭代器，每次迭代产生一个元组（idx, unroll\_factor\)，idx表示当前循环的索引值，unroll\_factor标识当前选择的展开方式。

## 约束说明<a name="section753174210543"></a>

-   展开因子列表会被排序并去重，且总是包含 1
-   展开因子按从大到小排序
-   每个展开因子会生成一个子循环

## 调用示例<a name="section646021318230"></a>

```
for _ in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx", unroll_list=[1, 2, 4]):
   ...
```

