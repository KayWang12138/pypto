# pypto.experimental.gather\_in\_l1<a name="ZH-CN_TOPIC_0000002520724489"></a>

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

## 功能说明<a name="section112531620115513"></a>

该接口为定制接口，约束较多。不保证稳定性。

从GM上的Tensor离散搬运指定行的数据，同时每行搬运前size个数据至L1。

## 函数原型<a name="section13445759105712"></a>

```
gather_in_l1(src: Tensor, indices: Tensor, blockTable: Tensor, blockSize: int,
                 size: int, is_b_matrix: bool, is_trans: bool):
```

参数说明

<a name="zh-cn_topic_0146324969_table29998725"></a>
<table><thead align="left"><tr id="zh-cn_topic_0146324969_row8953505"><th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0146324969_p54145286"><a name="zh-cn_topic_0146324969_p54145286"></a><a name="zh-cn_topic_0146324969_p54145286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0146324969_p23692060"><a name="zh-cn_topic_0146324969_p23692060"></a><a name="zh-cn_topic_0146324969_p23692060"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.7%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0146324969_p19480441"><a name="zh-cn_topic_0146324969_p19480441"></a><a name="zh-cn_topic_0146324969_p19480441"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p94336309250"><a name="p94336309250"></a><a name="p94336309250"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p11066451345"><a name="p11066451345"></a><a name="p11066451345"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a>源操作数。</p>
<p id="p1872462420360"><a name="p1872462420360"></a><a name="p1872462420360"></a>支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8</p>
<p id="p65652202218"><a name="p65652202218"></a><a name="p65652202218"></a>不支持空Tensor，支持两维。</p>
</td>
</tr>
<tr id="row148718148574"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p138721410572"><a name="p138721410572"></a><a name="p138721410572"></a>indices</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p1787014105716"><a name="p1787014105716"></a><a name="p1787014105716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p687151410574"><a name="p687151410574"></a><a name="p687151410574"></a>源操作数的行偏移</p>
<p id="p143281141171719"><a name="p143281141171719"></a><a name="p143281141171719"></a>支持的数据类型为：DT_INT32、DT_INT64</p>
<p id="p1082013228"><a name="p1082013228"></a><a name="p1082013228"></a>不支持空Tensor，支持两维。</p>
<p id="p1725416441574"><a name="p1725416441574"></a><a name="p1725416441574"></a>shape形状为[1,n]。</p>
</td>
</tr>
<tr id="row1853016014125"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p1692916591218"><a name="p1692916591218"></a><a name="p1692916591218"></a>blockTable</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p1692935131212"><a name="p1692935131212"></a><a name="p1692935131212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p192965141219"><a name="p192965141219"></a><a name="p192965141219"></a>源操作数</p>
<p id="p174621923171816"><a name="p174621923171816"></a><a name="p174621923171816"></a>支持的数据类型为DT_INT32，</p>
<p id="p6543203615185"><a name="p6543203615185"></a><a name="p6543203615185"></a>不支持空Tensor，支持两维。</p>
<p id="p1877019541628"><a name="p1877019541628"></a><a name="p1877019541628"></a>在实际使用中表示为 Page Attention  中的页表，形状为[1,block_table_size]，其中block_table_size表示页表的长度</p>
</td>
</tr>
<tr id="row1445118371211"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p828941818412"><a name="p828941818412"></a><a name="p828941818412"></a>blockSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p129295516124"><a name="p129295516124"></a><a name="p129295516124"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p129291659125"><a name="p129291659125"></a><a name="p129291659125"></a>源操作数</p>
<p id="p1515521881818"><a name="p1515521881818"></a><a name="p1515521881818"></a>int 类型</p>
<p id="p1924643219415"><a name="p1924643219415"></a><a name="p1924643219415"></a>表示 Page Attention 中一个块可以放多少个token</p>
</td>
</tr>
<tr id="row1860017145572"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p76001714155716"><a name="p76001714155716"></a><a name="p76001714155716"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p66001214105720"><a name="p66001214105720"></a><a name="p66001214105720"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p76001214115710"><a name="p76001214115710"></a><a name="p76001214115710"></a>每行搬运的数据数</p>
<p id="p317293316352"><a name="p317293316352"></a><a name="p317293316352"></a>数据数要小于源操作数的列数</p>
</td>
</tr>
<tr id="row19289141818412"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p485113771812"><a name="p485113771812"></a><a name="p485113771812"></a>is_b_matrix</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p1328941816413"><a name="p1328941816413"></a><a name="p1328941816413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p142891818544"><a name="p142891818544"></a><a name="p142891818544"></a>搬运后的结果，即输出Tensor是否作为matmul的B矩阵</p>
</td>
</tr>
<tr id="row573213018149"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p1125894120182"><a name="p1125894120182"></a><a name="p1125894120182"></a>is_trans</p>
</td>
<td class="cellrowborder" valign="top" width="11.01%" headers="mcps1.1.4.1.2 "><p id="p1573213061416"><a name="p1573213061416"></a><a name="p1573213061416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.7%" headers="mcps1.1.4.1.3 "><p id="p13781135195"><a name="p13781135195"></a><a name="p13781135195"></a>搬运后的结果，即输出Tensor是否转置</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1334651291213"></a>

返回输出Tensor

## 调用示例<a name="section4127133461016"></a>

```
src = pypto.tensor([16, 32], pypto.DT_FP32, "tensor_src")
offset = pypto.tensor([1, 32], pypto.DT_INT32, "tensor_offset")
out = pypto.experimental.gather_in_l1(src , offset, 20, false, false)
```

