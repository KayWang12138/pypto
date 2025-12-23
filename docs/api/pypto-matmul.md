# pypto.matmul<a name="ZH-CN_TOPIC_0000002470278122"></a>

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

实现input 、mat2矩阵的矩阵乘运算，计算公式为：out = input @ mat2

-   input 、mat2为源操作数，input 为左矩阵；mat2为右矩阵
-   out 为目的操作数，存放矩阵乘结果的矩阵

## 函数原型<a name="section19138102360"></a>

```
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## 参数说明<a name="section75724101161"></a>

<a name="zh-cn_topic_0146324969_table29998725"></a>
<table><thead align="left"><tr id="zh-cn_topic_0146324969_row8953505"><th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0146324969_p54145286"><a name="zh-cn_topic_0146324969_p54145286"></a><a name="zh-cn_topic_0146324969_p54145286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0146324969_p23692060"><a name="zh-cn_topic_0146324969_p23692060"></a><a name="zh-cn_topic_0146324969_p23692060"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.87%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0146324969_p19480441"><a name="zh-cn_topic_0146324969_p19480441"></a><a name="zh-cn_topic_0146324969_p19480441"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0146324969_row41106249"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p135254139556"><a name="p135254139556"></a><a name="p135254139556"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p275718385618"><a name="p275718385618"></a><a name="p275718385618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p1535483411719"><a name="p1535483411719"></a><a name="p1535483411719"></a>表示输入左矩阵。不支持输入空Tensor。</p>
<p id="p435413416175"><a name="p435413416175"></a><a name="p435413416175"></a>支持的数据类型为：DT_INT8, DT_FP16, DT_BF16，DT_FP32，且左右矩阵数据类型需保持一致。</p>
<p id="p128352465228"><a name="p128352465228"></a><a name="p128352465228"></a>支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。</p>
<p id="p7354534131718"><a name="p7354534131718"></a><a name="p7354534131718"></a>支持的Format为：TILEOP_ND, TILEOP_NZ。</p>
<p id="p321545011245"><a name="p321545011245"></a><a name="p321545011245"></a>当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。</p>
<p id="p2173195242413"><a name="p2173195242413"></a><a name="p2173195242413"></a>当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。</p>
<p id="p15387029167"><a name="p15387029167"></a><a name="p15387029167"></a>内轴外轴：当输入矩阵input非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵input转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M；</p>
<p id="p38393116259"><a name="p38393116259"></a><a name="p38393116259"></a>在使用pypto.view接口的场景，应保证传入View的shape维度也满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐</p>
<p id="p13354103461712"><a name="p13354103461712"></a><a name="p13354103461712"></a>当矩阵维度为3维或者4维时，不支持pypto.view场景。</p>
</td>
</tr>
<tr id="zh-cn_topic_0146324969_row46369059"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p67301442195511"><a name="p67301442195511"></a><a name="p67301442195511"></a>mat2</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p6282445566"><a name="p6282445566"></a><a name="p6282445566"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p18479142753"><a name="p18479142753"></a><a name="p18479142753"></a>表示输入右矩阵。不支持输入空Tensor。</p>
<p id="p249002918435"><a name="p249002918435"></a><a name="p249002918435"></a>支持的数据类型为：DT_INT8, DT_FP16, DT_BF16，DT_FP32，且左右矩阵数据类型需保持一致。</p>
<p id="p156511784235"><a name="p156511784235"></a><a name="p156511784235"></a>支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。</p>
<p id="p1598210219206"><a name="p1598210219206"></a><a name="p1598210219206"></a>支持的Format为：TILEOP_ND, TILEOP_NZ。</p>
<p id="p9929339256"><a name="p9929339256"></a><a name="p9929339256"></a>当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。</p>
<p id="p43211823123410"><a name="p43211823123410"></a><a name="p43211823123410"></a>当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。</p>
<p id="p5532737192"><a name="p5532737192"></a><a name="p5532737192"></a>内轴外轴：当输入矩阵mat2非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat2转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K；</p>
<p id="p56532494505"><a name="p56532494505"></a><a name="p56532494505"></a>在使用pypto.view接口的场景，应保证传入View的shape维度也满足内轴32字节对齐（其中当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。</p>
<p id="p11478163717368"><a name="p11478163717368"></a><a name="p11478163717368"></a>当输出矩阵Format为TILEOP_NZ，且输入矩阵mat2转置时，输入矩阵mat2对应数据排布为[N, K]，此时外轴为N，当输入矩阵数据类型为DT_FP16、DT_BF16、DT_INT8时N轴需要满足16元素对齐，数据类型为DT_FP32时N轴要满足8元素对齐。</p>
<p id="p17598162220241"><a name="p17598162220241"></a><a name="p17598162220241"></a>当矩阵维度为3维或者4维时，不支持pypto.view场景</p>
</td>
</tr>
<tr id="row1745413528152"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p1045565251510"><a name="p1045565251510"></a><a name="p1045565251510"></a>out_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p1945545219159"><a name="p1945545219159"></a><a name="p1945545219159"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p420614111528"><a name="p420614111528"></a><a name="p420614111528"></a>表示输出矩阵数据类型，支持DT_FP32，DT_FP16，DT_BF16, DT_INT32。</p>
<a name="ul735211265515"></a><a name="ul735211265515"></a><ul id="ul735211265515"><li>输入矩阵数据类型为DT_FP16时，out_dtype可选DT_FP32, DT_FP16。</li><li>输入矩阵数据类型为DT_BF16时，out_dtype可选DT_FP32, DT_BF16。</li><li>输入矩阵数据类型为DT_INT8时，out_dtype可选DT_INT32。</li><li>输入矩阵数据类型为DT_FP32时，out_dtype可选DT_FP32。</li></ul>
</td>
</tr>
<tr id="row3756191017161"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p117568108165"><a name="p117568108165"></a><a name="p117568108165"></a>a_trans</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p075671091615"><a name="p075671091615"></a><a name="p075671091615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p92951028194213"><a name="p92951028194213"></a><a name="p92951028194213"></a>参数a_trans表示输入左矩阵是否转置，默认为False。</p>
</td>
</tr>
<tr id="row63741118141617"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p637431831618"><a name="p637431831618"></a><a name="p637431831618"></a>b_trans</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p10374191871611"><a name="p10374191871611"></a><a name="p10374191871611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p82958280425"><a name="p82958280425"></a><a name="p82958280425"></a>参数b_trans表示输入右矩阵是否转置，默认为False。</p>
</td>
</tr>
<tr id="row3205258162"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p112092511615"><a name="p112092511615"></a><a name="p112092511615"></a>c_matrix_nz</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p920125151610"><a name="p920125151610"></a><a name="p920125151610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p2296428104214"><a name="p2296428104214"></a><a name="p2296428104214"></a>参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False。</p>
</td>
</tr>
<tr id="row43281639112610"><td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.1 "><p id="p16328123952612"><a name="p16328123952612"></a><a name="p16328123952612"></a>extend_params</p>
</td>
<td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.2 "><p id="p1332883914265"><a name="p1332883914265"></a><a name="p1332883914265"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.87%" headers="mcps1.1.4.1.3 "><p id="p8328133962615"><a name="p8328133962615"></a><a name="p8328133962615"></a>支持bias及fixpipe的反量化功能，数据类型为字典格式。</p>
<a name="ul178736590307"></a><a name="ul178736590307"></a><ul id="ul178736590307"><li>bias功能<p id="p17154171311314"><a name="p17154171311314"></a><a name="p17154171311314"></a>参数：bias_tensor</p>
<p id="p11493122013111"><a name="p11493122013111"></a><a name="p11493122013111"></a>类型：Tensor</p>
<p id="p112315753211"><a name="p112315753211"></a><a name="p112315753211"></a>功能说明：</p>
<a name="ul066532314330"></a><a name="ul066532314330"></a><ul id="ul066532314330"><li>输入AB矩阵数据类型为DT_FP16时，Bias矩阵数据类型可选DT_FP16和DT_FP32。</li><li>输入AB矩阵数据类型为DT_FP32时，Bias矩阵数据类型只能为DT_FP32。</li><li>输入AB矩阵数据类型为DT_INT8时，Bias矩阵数据类型只能为DT_INT32。</li><li>bias_tensor只支持ND格式。</li><li>bias_tensor的第一维度应置1，且N维度需要与mat2矩阵的N维度相等。</li><li>Bias不支持多核切K功能。</li><li>仅支持矩阵维度为2维场景。</li></ul>
</li><li>fixpipe反量化功能：只支持INT8输入，FP16输出，不支持多核切K功能，scale和scale_tensor只能传一个。<p id="p20204181373717"><a name="p20204181373717"></a><a name="p20204181373717"></a>参数：scale</p>
<p id="p20497161113510"><a name="p20497161113510"></a><a name="p20497161113510"></a>类型：float</p>
<p id="p19204161312372"><a name="p19204161312372"></a><a name="p19204161312372"></a>功能说明：</p>
<a name="ul14462153711430"></a><a name="ul14462153711430"></a><ul id="ul14462153711430"><li>输入为float类型，取1为符号位 + 8位指数位 + 10位尾数位参与运算。</li></ul>
<p id="p13498943203711"><a name="p13498943203711"></a><a name="p13498943203711"></a>参数：relu_type</p>
<p id="p54261727173510"><a name="p54261727173510"></a><a name="p54261727173510"></a>类型：ReLuType</p>
<p id="p249864317371"><a name="p249864317371"></a><a name="p249864317371"></a>功能说明：</p>
<a name="ul12498134383720"></a><a name="ul12498134383720"></a><ul id="ul12498134383720"><li>支持RELU和NO_RELU两种模式。</li><li>仅支持矩阵维度为2维场景。</li></ul>
<p id="p142636485375"><a name="p142636485375"></a><a name="p142636485375"></a>参数：scale_tensor</p>
<p id="p2126175143510"><a name="p2126175143510"></a><a name="p2126175143510"></a>类型：Tensor</p>
<p id="p7263048103714"><a name="p7263048103714"></a><a name="p7263048103714"></a>功能说明：</p>
<a name="ul136356114472"></a><a name="ul136356114472"></a><ul id="ul136356114472"><li>scale_tensor输入固定为uint64_t 的Tensor。计算时会转换uint64_t为float类型的低32位bit后，取1为符号位 + 8位指数位 + 10位尾数位参与运算。</li><li>scale_tensor的第一维度必须置1，且N维度需要与mat2矩阵的N维度相等。</li><li>scale_tensor只支持ND格式。</li><li>仅支持矩阵维度为2维场景。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26662142616"></a>

返回值为out 矩阵（Tensor）。

## 约束说明<a name="section2581910105618"></a>

-   调用matmul接口前需要通过pypto.set\_cube\_tile\_shapes设置M、N、K轴上的切分大小
-   当矩阵维度为3维或者4维时，需要调用pypto.set\_vec\_tile\_shapes接口设置vector的TileShape切分，如未设置，接口内部会设置2维的vec\_tile\_shape，其值为128，128。
-   调用matmul接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始shape的m,k,n值。
-   调用matmul接口的输入矩阵维度为3维/4维并且数据格式为NZ格式时，需要调用pypto.set\_matrix\_size接口设置输入到matmul的原始shape的m,k,n值。

## 调用示例<a name="section4127133461016"></a>

```
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)
   
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pto.tensor((16, 32), pto.DT_INT8, "tensor_a")
 b = pto.tensor((32, 64), pto.DT_INT8, "tensor_b")
 extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
 pto.matmul(a, b, pto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pto.tensor((16, 32), pto.DT_INT8, "tensor_a")
 b = pto.tensor((32, 64), pto.DT_INT8, "tensor_b")
 scale_tensor = pto.tensor((1, 64), pto.DT_UINT64, "tensor_scale")
 extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
 pto.matmul(a, b, pto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)
```

