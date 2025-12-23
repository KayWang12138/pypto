# pypto.scatter\_update<a name="ZH-CN_TOPIC_0000002470278118"></a>

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

功能1：将4维src根据2维索引index更新到4维 input 上，计算公式如下：

![](figures/zh-cn_formulaimage_0000002471398200.png)

功能2：将2维src根据2维index更新到2维 input 上，计算公式如下：

![](figures/zh-cn_formulaimage_0000002471238212.png)

## 函数原型<a name="section8786125214915"></a>

```
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## 参数说明<a name="section644919345515"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.470000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p57634810538"><a name="p57634810538"></a><a name="p57634810538"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="7.470000000000001%" headers="mcps1.1.4.1.2 "><p id="p1329911375318"><a name="p1329911375318"></a><a name="p1329911375318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.99%" headers="mcps1.1.4.1.3 "><p id="p652854125310"><a name="p652854125310"></a><a name="p652854125310"></a><span>待更新的Tensor</span>。</p>
<p id="p6203457185819"><a name="p6203457185819"></a><a name="p6203457185819"></a>支持的数据类型为：DT_FP32/DT_FP16/DT_BF16/INT32/INT16。</p>
<p id="p1120416193145"><a name="p1120416193145"></a><a name="p1120416193145"></a>支持的维度：2维，4维</p>
<p id="p14554937181812"><a name="p14554937181812"></a><a name="p14554937181812"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
<tr id="row197211619118"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p15416192021918"><a name="p15416192021918"></a><a name="p15416192021918"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="7.470000000000001%" headers="mcps1.1.4.1.2 "><p id="p1141642020193"><a name="p1141642020193"></a><a name="p1141642020193"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.99%" headers="mcps1.1.4.1.3 "><p id="p64161120111914"><a name="p64161120111914"></a><a name="p64161120111914"></a>请保持默认值-2。</p>
</td>
</tr>
<tr id="row113226117598"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p10322161115592"><a name="p10322161115592"></a><a name="p10322161115592"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="7.470000000000001%" headers="mcps1.1.4.1.2 "><p id="p13432181975917"><a name="p13432181975917"></a><a name="p13432181975917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.99%" headers="mcps1.1.4.1.3 "><p id="p551114212491"><a name="p551114212491"></a><a name="p551114212491"></a><span>input的一组索引</span>。</p>
<p id="p7220106135015"><a name="p7220106135015"></a><a name="p7220106135015"></a>支持的数据类型：INT64/INT32/INT16。</p>
<p id="p458575941216"><a name="p458575941216"></a><a name="p458575941216"></a>支持的维度：<span>2维</span></p>
</td>
</tr>
<tr id="row2582163131918"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p11583183141914"><a name="p11583183141914"></a><a name="p11583183141914"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="7.470000000000001%" headers="mcps1.1.4.1.2 "><p id="p155831337191"><a name="p155831337191"></a><a name="p155831337191"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.99%" headers="mcps1.1.4.1.3 "><p id="p6876151311547"><a name="p6876151311547"></a><a name="p6876151311547"></a>src<span>是一组更新值</span>。</p>
<p id="p10876181312547"><a name="p10876181312547"></a><a name="p10876181312547"></a>支持的数据类型为：DT_FP32/DT_FP16/DT_BF16/INT32/INT16。数据类型和input保持一致</p>
<p id="p189184244137"><a name="p189184244137"></a><a name="p189184244137"></a>支持的维度：2维，4维</p>
<p id="p1464961991315"><a name="p1464961991315"></a><a name="p1464961991315"></a>不支持空Tensor，且Shape Size不大于2147483647（即INT32_MAX）。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回更新后的 input，为inplace操作。

## 约束说明<a name="section753174210543"></a>

broadcast 约束：不支持 broadcast。

tileshape 约束：尾轴不做 tileshape 切分，四维场景下第三轴不做切分，tileshape 设置针对 src 生效，2维场景下0维大小为s1的约数或者倍数，如src Shape为\[12, 64\]且index Shape为\[3, 4\]，则 tileshape 为 tileshape\[0, 64\]，其中 tileshape0 =1、2、4、8、12、24等。

## 调用示例<a name="section642mcpsimp"></a>

-   将2维 src 根据2维index更新到2维input上

    ```
    x = pypto.tensor([8, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_FP32)
    o = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```
    输入数据x:[[0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0]]
    输入数据y:[[1 2],
               [4 5]]
    输入数据z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    输出数据o:[[0 0 0],
               [1 2 3],
               [4 5 6],
               [0 0 0],
               [7 8 9],
               [10 11 12],
               [0 0 0],
               [0 0 0]])
    ```

-   将4维src根据2维索引index更新到4维input上

    ```
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_FP32)
    o = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```
    输入数据x:[[
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ]]
    输入数据y:[[1 8],
               [4 10]]
    输入数据z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    输出数据o:[[
                 [[0 0 0]],
                 [[1 2 3]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[7 8 9]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[4 5 6]],
                 [[0 0 0]],
                 [[10 11 12]],
                 [[0 0 0]],
               ]]
    ```

