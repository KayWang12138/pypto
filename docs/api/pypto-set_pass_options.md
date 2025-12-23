# pypto.set\_pass\_options<a name="ZH-CN_TOPIC_0000002503357975"></a>

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

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型<a name="section8786125214915"></a>

```
set_pass_options(*,
                     pg_skip_partition: Optional[bool] = None,
                     pg_upper_bound: Optional[int] = None,
                     pg_lower_bound: Optional[int] = None,
                     pg_parallel_lower_bound: Optional[int] = None,
                     mg_vec_parallel_lb: Optional[int] = None,
                     cube_nbuffer_mode: Optional[int] = None,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_mode: Optional[int] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_mode: Optional[int] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     mg_copyin_upper_bound: Optional[int] = None,
                     sg_set_scope: Optional[int] = None,
                     )
```

## 参数说明<a name="section644919345515"></a>

<a name="table1776912518297"></a>
<table><thead align="left"><tr id="row1876912592919"><th class="cellrowborder" valign="top" width="22.470000000000002%" id="mcps1.1.4.1.1"><p id="p1977919501843"><a name="p1977919501843"></a><a name="p1977919501843"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.4.1.2"><p id="p177790504416"><a name="p177790504416"></a><a name="p177790504416"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.00999999999999%" id="mcps1.1.4.1.3"><p id="p17791750045"><a name="p17791750045"></a><a name="p17791750045"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row776920512911"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p1476912514292"><a name="p1476912514292"></a><a name="p1476912514292"></a>pg_skip_partition</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p497412439414"><a name="p497412439414"></a><a name="p497412439414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p786581882820"><a name="p786581882820"></a><a name="p786581882820"></a>含义：是否跳过子图切分过程。</p>
<p id="p10523744162916"><a name="p10523744162916"></a><a name="p10523744162916"></a>说明：当值为True时，将完整的计算图作为单一子图，不进行切分。当值为False时，进行子图切分。</p>
<p id="p18641191414322"><a name="p18641191414322"></a><a name="p18641191414322"></a>类型：bool</p>
<p id="p15463132482813"><a name="p15463132482813"></a><a name="p15463132482813"></a>取值范围：{True, False}</p>
<p id="p086510184291"><a name="p086510184291"></a><a name="p086510184291"></a>默认值：False</p>
<p id="p1781153315296"><a name="p1781153315296"></a><a name="p1781153315296"></a>影响Pass范围： GraphPartition</p>
</td>
</tr>
<tr id="row477075112916"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p3556115082915"><a name="p3556115082915"></a><a name="p3556115082915"></a>pg_upper_bound</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p7974443148"><a name="p7974443148"></a><a name="p7974443148"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p119995953119"><a name="p119995953119"></a><a name="p119995953119"></a><span>含义：合图参数，用于配置子图大小上界。</span></p>
<p id="p14586131115316"><a name="p14586131115316"></a><a name="p14586131115316"></a>说明：<span>当子图大小达到上界不允许与其他子图合并。</span></p>
<p id="p15277205318318"><a name="p15277205318318"></a><a name="p15277205318318"></a>类型：<span>int</span></p>
<p id="p5477813123119"><a name="p5477813123119"></a><a name="p5477813123119"></a>取值范围<span>：0~2147483647</span></p>
<p id="p14429115103112"><a name="p14429115103112"></a><a name="p14429115103112"></a>默认值<span>：10000</span></p>
<p id="p67707572911"><a name="p67707572911"></a><a name="p67707572911"></a>影响Pass范围：<span> GraphPartition</span></p>
</td>
</tr>
<tr id="row2770145172914"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p11556650192915"><a name="p11556650192915"></a><a name="p11556650192915"></a>pg_lower_bound</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p14974184315418"><a name="p14974184315418"></a><a name="p14974184315418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p186506284314"><a name="p186506284314"></a><a name="p186506284314"></a>含义：合图参数，用于配置子图大小下界。</p>
<p id="p6893385312"><a name="p6893385312"></a><a name="p6893385312"></a>说明：当子图大小小于下界时尝试与其他子图合并。</p>
<p id="p13862105123115"><a name="p13862105123115"></a><a name="p13862105123115"></a>类型：int</p>
<p id="p1517024023114"><a name="p1517024023114"></a><a name="p1517024023114"></a>取值范围：0~2147483647</p>
<p id="p1444517422312"><a name="p1444517422312"></a><a name="p1444517422312"></a>默认值：512</p>
<p id="p1777019582917"><a name="p1777019582917"></a><a name="p1777019582917"></a>影响Pass范围： GraphPartition</p>
</td>
</tr>
<tr id="row3770952294"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p1855675010295"><a name="p1855675010295"></a><a name="p1855675010295"></a><span>pg_parallel_lower_bound</span></p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p1897454318415"><a name="p1897454318415"></a><a name="p1897454318415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p3821205383218"><a name="p3821205383218"></a><a name="p3821205383218"></a><span>含义：合图参数，用于配置相同结构子图的最小并行度。</span></p>
<p id="p166275610327"><a name="p166275610327"></a><a name="p166275610327"></a>说明：<span>当某个相同结构的子图数小于该值时不做合并。</span></p>
<p id="p43051258103214"><a name="p43051258103214"></a><a name="p43051258103214"></a>类型：<span>int</span></p>
<p id="p14875185915327"><a name="p14875185915327"></a><a name="p14875185915327"></a>取值范围<span>：0~2147483647</span></p>
<p id="p957071123319"><a name="p957071123319"></a><a name="p957071123319"></a>默认值<span>：20</span></p>
<p id="p1577017518291"><a name="p1577017518291"></a><a name="p1577017518291"></a>影响Pass范围：<span> GraphPartition</span></p>
</td>
</tr>
<tr id="row11021959121512"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p410315915158"><a name="p410315915158"></a><a name="p410315915158"></a>sg_set_scope</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p9103195911159"><a name="p9103195911159"></a><a name="p9103195911159"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p1697510300168"><a name="p1697510300168"></a><a name="p1697510300168"></a>含义：手动控制合图参数。</p>
<p id="p172121738161615"><a name="p172121738161615"></a><a name="p172121738161615"></a>说明：将operation赋予特定的scopeId，若相邻的operation具有相同的非-1的scopeId，则会被强制合并在一个子图之中，并且这个子图不会与其他子图合并。</p>
<p id="p851914118164"><a name="p851914118164"></a><a name="p851914118164"></a>类型：int</p>
<p id="p8347184418164"><a name="p8347184418164"></a><a name="p8347184418164"></a>取值范围：-1~2147483647</p>
<p id="p17144647181612"><a name="p17144647181612"></a><a name="p17144647181612"></a>默认值：-1</p>
<p id="p2103145901515"><a name="p2103145901515"></a><a name="p2103145901515"></a>影响Pass范围：GraphPartition</p>
</td>
</tr>
<tr id="row777010519298"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p1477025152919"><a name="p1477025152919"></a><a name="p1477025152919"></a><span>mg_vec_parallel_lb</span></p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p4974124319417"><a name="p4974124319417"></a><a name="p4974124319417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p12121915153319"><a name="p12121915153319"></a><a name="p12121915153319"></a>含义：合图参数，用于配置相同结构AIV子图的最小并行度。</p>
<p id="p1813414177339"><a name="p1813414177339"></a><a name="p1813414177339"></a>说明：当某个相同结构的子图数小于该值时不做合并。</p>
<p id="p19556102473316"><a name="p19556102473316"></a><a name="p19556102473316"></a>类型：int</p>
<p id="p12620206338"><a name="p12620206338"></a><a name="p12620206338"></a>取值范围<span>：0~2147483647</span></p>
<p id="p166522219336"><a name="p166522219336"></a><a name="p166522219336"></a>默认值：48</p>
<p id="p18770155162918"><a name="p18770155162918"></a><a name="p18770155162918"></a>影响Pass范围：NBufferMerge</p>
</td>
</tr>
<tr id="row17701552918"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p9556195042912"><a name="p9556195042912"></a><a name="p9556195042912"></a><span>vec_nbuffer_mode</span></p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p49741543344"><a name="p49741543344"></a><a name="p49741543344"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p259084212337"><a name="p259084212337"></a><a name="p259084212337"></a><span>含义：合图参数，用于配置相同结构AIV子图合并策略。</span></p>
<p id="p9988154463319"><a name="p9988154463319"></a><a name="p9988154463319"></a>说明：<span>该参数适用于结构相同的AIV子图合并，避免同一结构子图数过大并增大核内流水调度可能性。</span></p>
<p id="p104081275342"><a name="p104081275342"></a><a name="p104081275342"></a>类型：<span>int</span></p>
<p id="p19838141813228"><a name="p19838141813228"></a><a name="p19838141813228"></a>取值：</p>
<a name="ul116531427122211"></a><a name="ul116531427122211"></a><ul id="ul116531427122211"><li>0<span>：不使能相同结构子图间合并逻辑。</span></li><li>1<span>：使能相同结构子图间合并，合并逻辑为依据sgVecParallelNum自适应计算每个结构的合并数。</span></li><li>2<span>：所有结构相同子图都按用户设置VecNBufferMap来做子图间的合并。</span></li></ul>
<p id="p11174851133318"><a name="p11174851133318"></a><a name="p11174851133318"></a>默认值<span>：1</span></p>
<p id="p0770195152915"><a name="p0770195152915"></a><a name="p0770195152915"></a>影响Pass范围：<span> NBufferMerge</span></p>
</td>
</tr>
<tr id="row14995333173413"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p599563312344"><a name="p599563312344"></a><a name="p599563312344"></a>vec_nbuffer_setting</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p697419431347"><a name="p697419431347"></a><a name="p697419431347"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p131321443133413"><a name="p131321443133413"></a><a name="p131321443133413"></a>含义：合图参数，用于配置相同结构AIV子图的合并数量。</p>
<p id="p1973014413410"><a name="p1973014413410"></a><a name="p1973014413410"></a>说明：该参数适用于结构相同的AIV子图合并。</p>
<p id="p190503215220"><a name="p190503215220"></a><a name="p190503215220"></a>类型： dict[int, int]</p>
<p id="p77471249133412"><a name="p77471249133412"></a><a name="p77471249133412"></a>使用条件：</p>
<p id="p10811956103411"><a name="p10811956103411"></a><a name="p10811956103411"></a>CubenBufferMode = 0/1, VecNBufferSetting 设置为nullMap。</p>
<p id="p1585611591346"><a name="p1585611591346"></a><a name="p1585611591346"></a>CubenBufferMode = 2，用户手动设置VecNBufferSetting 。</p>
<p id="p1312844183513"><a name="p1312844183513"></a><a name="p1312844183513"></a>默认值：nullMap</p>
<p id="p119959338349"><a name="p119959338349"></a><a name="p119959338349"></a>影响Pass范围： NBufferMerge</p>
</td>
</tr>
<tr id="row02581215113514"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p17639342282"><a name="p17639342282"></a><a name="p17639342282"></a>cube_l1_reuse_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p109745435410"><a name="p109745435410"></a><a name="p109745435410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p9775122816354"><a name="p9775122816354"></a><a name="p9775122816354"></a><span>含义：合图参数，用于配置结构相同且重复搬运同一GM数据的子图合并策略。</span></p>
<p id="p17523630143518"><a name="p17523630143518"></a><a name="p17523630143518"></a>说明：<span>该参数适用于含有CUBE计算的子图，避免同一数据被重复搬运次数过多。</span></p>
<p id="p67411186119"><a name="p67411186119"></a><a name="p67411186119"></a>类型：<span>int</span>0<span>：不使能结构相同且存在重复搬运子图间合并逻辑。</span>&gt;0<span>：所有结构都按用户设置的值来做子图间的合并。</span></p>
<p id="p921910319250"><a name="p921910319250"></a><a name="p921910319250"></a>取值：0~2147483647</p>
<p id="p9549123623518"><a name="p9549123623518"></a><a name="p9549123623518"></a>默认值<span>：0</span></p>
<p id="p325851533511"><a name="p325851533511"></a><a name="p325851533511"></a>影响Pass范围：<span> L1ReuseMerge</span></p>
</td>
</tr>
<tr id="row2053113280426"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p17531928184218"><a name="p17531928184218"></a><a name="p17531928184218"></a>cube_l1_reuse_setting</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p497424318413"><a name="p497424318413"></a><a name="p497424318413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p12824458425"><a name="p12824458425"></a><a name="p12824458425"></a>含义：合图参数，用于配置结构相同且重复搬运同一GM数据的子图合并数量。</p>
<p id="p1952414654212"><a name="p1952414654212"></a><a name="p1952414654212"></a>说明：该参数适用于含有CUBE计算的子图合并</p>
<p id="p1623449174214"><a name="p1623449174214"></a><a name="p1623449174214"></a>类型： dict[int, int]</p>
<p id="p19421195115429"><a name="p19421195115429"></a><a name="p19421195115429"></a>默认值：nullMap</p>
<p id="p1053182816423"><a name="p1053182816423"></a><a name="p1053182816423"></a>影响Pass范围：L1ReuseMerge</p>
</td>
</tr>
<tr id="row15769164718357"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p27708477356"><a name="p27708477356"></a><a name="p27708477356"></a>cube_nbuffer_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p159743439416"><a name="p159743439416"></a><a name="p159743439416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p11953458133515"><a name="p11953458133515"></a><a name="p11953458133515"></a><span>含义：合图参数，用于配置相同结构AIC子图合并策略</span>。</p>
<p id="p1222710173618"><a name="p1222710173618"></a><a name="p1222710173618"></a>说明：<span>该参数适用于结构相同的AIC子图合并，避免同一结构子图数过大并增大核内流水调度可能性。</span></p>
<p id="p998916523614"><a name="p998916523614"></a><a name="p998916523614"></a>类型：<span>int</span></p>
<p id="p115191620164717"><a name="p115191620164717"></a><a name="p115191620164717"></a>取值：</p>
<a name="ul9519142018471"></a><a name="ul9519142018471"></a><ul id="ul9519142018471"><li>0<span>：不使能相同结构子图间合并逻辑。但用户设置cube_nbuffer_setting时</span>仍然<span>按用户设置的cube_nbuffer_setting来做子图间的合并。</span></li><li>1<span>：使能相同结构子图间合并，合并逻辑为依据cube核数自适应计算每个结构的合并数。</span></li><li>2<span>：所有结构相同子图都按用户设置cube_nbuffer_setting来做子图间的合并。</span></li></ul>
<p id="p65199207477"><a name="p65199207477"></a><a name="p65199207477"></a>默认值<span>：0</span></p>
<p id="p8519172012476"><a name="p8519172012476"></a><a name="p8519172012476"></a>影响Pass范围：<span> L1ReuseMerge</span></p>
</td>
</tr>
<tr id="row18809113183619"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p6556155022920"><a name="p6556155022920"></a><a name="p6556155022920"></a>cube_nbuffer_setting</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p89744431145"><a name="p89744431145"></a><a name="p89744431145"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p1592192912367"><a name="p1592192912367"></a><a name="p1592192912367"></a>含义：合图参数，用于配置相同结构AIC子图的合并数量。</p>
<p id="p733933218362"><a name="p733933218362"></a><a name="p733933218362"></a>说明：该参数适用于结构相同的AIC子图合并。</p>
<p id="p168903412367"><a name="p168903412367"></a><a name="p168903412367"></a>类型： dict[int, int]</p>
<p id="p147434264521"><a name="p147434264521"></a><a name="p147434264521"></a>取值：</p>
<a name="ul1169993785219"></a><a name="ul1169993785219"></a><ul id="ul1169993785219"><li><span>{-1, N}：key为-1时，value值N表示</span>结构相同的AIC子图的合并数量默认值为N</li></ul>
<p id="p1778637133619"><a name="p1778637133619"></a><a name="p1778637133619"></a>默认值：nullMap</p>
<p id="p11810201320362"><a name="p11810201320362"></a><a name="p11810201320362"></a>影响Pass范围： L1ReuseMerge</p>
</td>
</tr>
<tr id="row787713417364"><td class="cellrowborder" valign="top" width="22.470000000000002%" headers="mcps1.1.4.1.1 "><p id="p11888651911"><a name="p11888651911"></a><a name="p11888651911"></a>mg_copyin_upper_bound</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p7974164314413"><a name="p7974164314413"></a><a name="p7974164314413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p68401552153618"><a name="p68401552153618"></a><a name="p68401552153618"></a><span>含义：合图参数，用于配置合图大小。</span></p>
<p id="p1525254193619"><a name="p1525254193619"></a><a name="p1525254193619"></a>说明：<span>该参数控制子图内搬运数据总量上界。当子图内数据搬运量大于该值则不再合并。</span></p>
<p id="p174165783612"><a name="p174165783612"></a><a name="p174165783612"></a>类型：<span>int</span></p>
<p id="p94494592366"><a name="p94494592366"></a><a name="p94494592366"></a>取值范围<span>：0~2147483647</span></p>
<p id="p635111233712"><a name="p635111233712"></a><a name="p635111233712"></a>默认值<span>：1024 * 1024</span></p>
<p id="p1487719416364"><a name="p1487719416364"></a><a name="p1487719416364"></a>影响Pass范围：<span> L1ReuseMerge</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="section753174210543"></a>

-   设置时机：必须在图编译开始前调用。
-   类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
-   作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例<a name="section642mcpsimp"></a>

```
   pypto.set_pass_options(pg_skip_partition=False,
                       pg_upper_bound=10000,
                       pg_lower_bound=512,
                       pg_parallel_lower_bound=24,
                       mg_vec_parallel_lb=48,
                       vec_nbuffer_mode=1,
                       vec_nbuffer_setting={},
                       cube_l1_reuse_mode=0,
                       cube_l1_reuse_setting={},
                       cube_nbuffer_mode=0,
                       cube_nbuffer_setting={},
                       mg_copyin_upper_bound=1024 * 1024)
```

