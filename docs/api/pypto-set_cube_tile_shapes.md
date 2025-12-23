# pypto.set\_cube\_tile\_shapes<a name="ZH-CN_TOPIC_0000002470278102"></a>

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

设置cube计算中的TileShape大小。

## 函数原型<a name="section1814166202715"></a>

```
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], set_l1_tile: bool = False, enable_split_k: bool = False) -> None
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1817713411812"><a name="p1817713411812"></a><a name="p1817713411812"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1329911375318"><a name="p1329911375318"></a><a name="p1329911375318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1891715019415"><a name="p1891715019415"></a><a name="p1891715019415"></a>m维度在L0和L1上的<span>TileShape</span>（切片形状）的切分大小，分别对应mL0和mL1的切分大小</p>
</td>
</tr>
<tr id="row181554794720"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p18816154712474"><a name="p18816154712474"></a><a name="p18816154712474"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p19816164719473"><a name="p19816164719473"></a><a name="p19816164719473"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1043412520487"><a name="p1043412520487"></a><a name="p1043412520487"></a>k维度在L0和L1上的<span>TileShape</span>（切片形状）的切分大小，分别对应kL0和kL1的切分大小</p>
</td>
</tr>
<tr id="row56231733194814"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p186235334480"><a name="p186235334480"></a><a name="p186235334480"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p66231533134811"><a name="p66231533134811"></a><a name="p66231533134811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p16959115210482"><a name="p16959115210482"></a><a name="p16959115210482"></a>n维度在L0和L1上的<span>TileShape</span>（切片形状）的切分大小，分别对应nL0和nL1的切分大小</p>
</td>
</tr>
<tr id="row183721836174817"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1637343694815"><a name="p1637343694815"></a><a name="p1637343694815"></a>set_l1_tile</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p19373103694810"><a name="p19373103694810"></a><a name="p19373103694810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p037313619488"><a name="p037313619488"></a><a name="p037313619488"></a>设置True表示使能matmul的L1大包搬运功能，False表示未使能L1的大包搬运，默认为False</p>
</td>
</tr>
<tr id="row397243904715"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1997333919474"><a name="p1997333919474"></a><a name="p1997333919474"></a>enable_split_k</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1397333918471"><a name="p1397333918471"></a><a name="p1397333918471"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1597314390473"><a name="p1597314390473"></a><a name="p1597314390473"></a>设置True表示使能matmul的多核切K功能，False表示未使能多核切K，默认为False</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

void

## 约束说明<a name="section753174210543"></a>

TileShape需要满足以下约束条件：

-   对齐约束：
    -   要求kL0、kL1、nL0、nL1均满足32字节对齐。例如，输入矩阵的数据类型为DT\_FP16时，kL0 \* sizeof\(DT\_FP16\) % 32 == 0。
    -   A矩阵在format为ND且转置场景时（即数据排布为\(K, M\)），要求mL0满足32字节对齐。
    -   A、B矩阵在format为NZ场景时，要求外轴切分大小满足16元素对齐，内轴切分大小满足32字节对齐。例如，在A矩阵非转置场景，外轴为M、内轴为K，要求mL0、mL1满足16元素对齐，kL0、kL1满足32字节对齐。
    -   0 < mL0 <= mL1 且 mL1 % mL0 == 0
    -   0 < kL0 <= kL1 且 kL1 % kL0 == 0
    -   0 < nL0 <= nL1 且 nL1 % nL0 == 0

-   buffer空间约束：
    -   L0A、L0B、L0C空间约束：
        -   当输入dtype为DT\_FP16、DT\_BF16或DT\_FP32

            CeilAlign\(mL0,16\)\* CeilAlign\(kL0,16\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,16\) \* CeilAlign\(kL0,16\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,16\)\* CeilAlign\(nL0,16\)\* sizeof\(cDtype\) <= L0C\_size

            其中aDtype、bDtype为输入dtype，cDtype为DT\_FP32

        -   当输入dtype为DT\_INT8

            CeilAlign\(mL0,32\)\* CeilAlign\(kL0,32\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,32\) \* CeilAlign\(kL0,32\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,32\)\* CeilAlign\(nL0,32\)\* sizeof\(cDtype\) <= L0C\_size

            其中aDtype、bDtype为输入dtype，cDtype为DT\_INT32

    -   L1空间约束：

        -   输入dtype为DT\_FP16或DT\_BF16或DT\_FP32

            CeilAlign\(mL1,16\)\* CeilAlign\(kL1,16\) \* sizeof\(aDtype\) + CeilAlign\(nL1,16\) \* CeilAlign\(kL1,16\) \* sizeof\(bDtype\) <= L1\_size

            其中aDtype、bDtype为输入dtype

        -   输入dtype为DT\_INT8

            CeilAlign\(mL1,32\)\* CeilAlign\(kL1,32\) \* sizeof\(aDtype\) + CeilAlign\(nL1,32\) \* CeilAlign\(kL1,32\) \* sizeof\(bDtype\) <= L1\_size

            其中aDtype、bDtype为输入dtype

        其中，CeilAlign（元素对齐）基本实现为：

        ```
        def ceil_align(value, align) {  return ((value + align - 1) // align) * align;}
        ```

Bias场景约束条件：

-   Bias空间约束：
    -   BTBuffer空间大小为1kb，且bias数据到达BTBuffer全部转为fp32，需满足以下约束:

        nL0 \* 4 <= BTBuffer\_size

FixPipe场景约束条件：

-   FixBuffer空间约束：
    -   FixBuffer空间大小为2kb，且scaleTensor数据为uint64\_t，需满足以下约束:

        nL0 \* 8 <= FixBuffer\_size

Output需要满足以下约束条件：

-   格式约束：当输出为NZ格式时，需要满足内轴（N轴）32字节对齐

当输入矩阵维度为3维或4维时，set\_l1\_tile、enable\_split\_k参数仅支持False，即不支持使能L1大包搬运及多核切K功能。

多核切K场景支持数据类型：

-   输入矩阵数据类型为DT\_FP16时，out\_dtype可选DT\_FP32。
-   输入矩阵数据类型为DT\_BF16时，out\_dtype可选DT\_FP32。
-   输入矩阵数据类型为DT\_INT8时，out\_dtype可选DT\_INT32。
-   输入矩阵数据类型为DT\_FP32时，out\_dtype可选DT\_FP32。

## 调用示例<a name="section646021318230"></a>

```
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], False, True)
```

