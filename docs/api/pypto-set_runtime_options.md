# pypto.set\_runtime\_options<a name="ZH-CN_TOPIC_0000002503198049"></a>

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

设置runtime的选项。

## 函数原型<a name="section1814166202715"></a>

```
set_runtime_options(*,
                    device_sched_mode : int = None,
                    stitch_function_inner_memory: int = None,
                    stitch_function_outcast_memory : int = None,
                    stitch_function_num_initial : int = None,
                    stitch_function_num_step : int = None,
                    stitch_function_size : int = None,
                    cfgcache_device_task_num : int = None,
                    cfgcache_root_task_num : int = None,
                    cfgcache_leaf_task_num : int = None,
                    run_mode : int = None,
                    ) -> None
```

## 参数说明<a name="section27141942204919"></a>

<a name="table332075643414"></a>
<table><thead align="left"><tr id="row173201156103418"><th class="cellrowborder" valign="top" width="22.439999999999998%" id="mcps1.1.4.1.1"><p id="p3552162123616"><a name="p3552162123616"></a><a name="p3552162123616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.4.1.2"><p id="p85521928363"><a name="p85521928363"></a><a name="p85521928363"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.04%" id="mcps1.1.4.1.3"><p id="p255272123616"><a name="p255272123616"></a><a name="p255272123616"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row15320125613420"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p183205563348"><a name="p183205563348"></a><a name="p183205563348"></a>device_sched_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p586575610352"><a name="p586575610352"></a><a name="p586575610352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p17320105603418"><a name="p17320105603418"></a><a name="p17320105603418"></a><span>含义</span><span>：</span>设置计算子图的调度模式</p>
<p id="p15995126105811"><a name="p15995126105811"></a><a name="p15995126105811"></a>说明：0：代表默认调度模式；</p>
<p id="p94296117589"><a name="p94296117589"></a><a name="p94296117589"></a>1：代表L2cache亲和调度模式，选择最新依赖ready的子图优先下发，达到复用L2cache的效果；</p>
<p id="p89321454115215"><a name="p89321454115215"></a><a name="p89321454115215"></a>2：公平调度模式，aicpu上多线程调度管理多个aicore的时候，下发子图会尽量控制在多线程间的公平性，此模式会带来额外的调度管理开销；</p>
<p id="p1749212243812"><a name="p1749212243812"></a><a name="p1749212243812"></a>3：代表同时开启L2cache亲和调度模式以及公平调度模式；</p>
<p id="p17320155643416"><a name="p17320155643416"></a><a name="p17320155643416"></a>类型：int</p>
<p id="p43201566345"><a name="p43201566345"></a><a name="p43201566345"></a><span>取值范围</span><span>：</span>0 或 1 或 2 或 3</p>
<p id="p532005616340"><a name="p532005616340"></a><a name="p532005616340"></a>默认值：0</p>
<p id="p63205564344"><a name="p63205564344"></a><a name="p63205564344"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row332055610344"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p832010569347"><a name="p832010569347"></a><a name="p832010569347"></a>stitch_function_inner_memory</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p88652563356"><a name="p88652563356"></a><a name="p88652563356"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p17320256153410"><a name="p17320256153410"></a><a name="p17320256153410"></a><span>含义：</span>控制用于分配root function的non-outcast内存池大小的参数，内存池大小为max_root_nonoutcast_workspace *STITCH_FUNCTION_INNER_MEMORY 。</p>
<p id="p432018562341"><a name="p432018562341"></a><a name="p432018562341"></a>说明：该数值越小，root function间越容易因workspace重叠导致互相产生依赖，导致无法并行；反之，该数值越大，通常stitch batch内并行度越高。</p>
<p id="p932035612343"><a name="p932035612343"></a><a name="p932035612343"></a>类型：int</p>
<p id="p93201556143412"><a name="p93201556143412"></a><a name="p93201556143412"></a>取值范围：1~128，当前版本取&gt;128的值不再有正面效果</p>
<p id="p532075615347"><a name="p532075615347"></a><a name="p532075615347"></a>默认值<span>：10</span></p>
<p id="p1432045616347"><a name="p1432045616347"></a><a name="p1432045616347"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row1832055612346"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p332012568346"><a name="p332012568346"></a><a name="p332012568346"></a>stitch_function_outcast_memory</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p19865165643510"><a name="p19865165643510"></a><a name="p19865165643510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p143201156153414"><a name="p143201156153414"></a><a name="p143201156153414"></a><span>含义：</span>编译算子时候用来评估算子在运行时需要的workspace内存的大小</p>
<p id="p143209567341"><a name="p143209567341"></a><a name="p143209567341"></a>说明：设置的值代表该workspace允许将多少loop的计算图动态的stitch到一起并行下发处理，设置的值越大代表评估使用的workspace内存越大</p>
<p id="p832016564347"><a name="p832016564347"></a><a name="p832016564347"></a>类型：int</p>
<p id="p143202565345"><a name="p143202565345"></a><a name="p143202565345"></a>取值范围:1 ~ 128</p>
<p id="p193201756103420"><a name="p193201756103420"></a><a name="p193201756103420"></a>默认值<span>：50</span></p>
<p id="p2320175612342"><a name="p2320175612342"></a><a name="p2320175612342"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row1832075693413"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p1976115327394"><a name="p1976115327394"></a><a name="p1976115327394"></a>stitch_function_num_initial</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p1786514562353"><a name="p1786514562353"></a><a name="p1786514562353"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p1574018401714"><a name="p1574018401714"></a><a name="p1574018401714"></a><span>含义：machine运行时ctrlflow aicpu里控制首个提交给schedule aicpu处理的stitch task的计算任务量</span></p>
<p id="p1432015568345"><a name="p1432015568345"></a><a name="p1432015568345"></a>说明：设置的值代表第一个stitch task里处理的loop个数，通过此值来控制头开销的大小，让ctrlflow aicpu和schedule aicpu计算尽快overlap起来</p>
<p id="p203206568340"><a name="p203206568340"></a><a name="p203206568340"></a>类型：int</p>
<p id="p1232065620340"><a name="p1232065620340"></a><a name="p1232065620340"></a>取值范围:1 ~ 128</p>
<p id="p83201156113414"><a name="p83201156113414"></a><a name="p83201156113414"></a>默认值<span>：30</span></p>
<p id="p132075663412"><a name="p132075663412"></a><a name="p132075663412"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row33205569346"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p9320135616344"><a name="p9320135616344"></a><a name="p9320135616344"></a>stitch_function_num_step</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p15865195616359"><a name="p15865195616359"></a><a name="p15865195616359"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p15320756143412"><a name="p15320756143412"></a><a name="p15320756143412"></a><span>含义：</span>machine运行时ctrlflow aicpu里控制非首次stitch task的处理loop的计算量</p>
<p id="p4320256153417"><a name="p4320256153417"></a><a name="p4320256153417"></a>说明：为了后续stitch task处理计算量平滑增加，可以通过设置此配置项进行控制。如设置为n，则每次stitch task里处理的loop次数分别base+n， base+2n 。。。</p>
<p id="p1932045620342"><a name="p1932045620342"></a><a name="p1932045620342"></a>类型：int</p>
<p id="p932012564341"><a name="p932012564341"></a><a name="p932012564341"></a>取值范围:1 ~ 128</p>
<p id="p83201556193410"><a name="p83201556193410"></a><a name="p83201556193410"></a>默认值<span>：30</span></p>
<p id="p123201956143417"><a name="p123201956143417"></a><a name="p123201956143417"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row7865129141110"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p1086513951113"><a name="p1086513951113"></a><a name="p1086513951113"></a>stitch_function_size</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p98651856183519"><a name="p98651856183519"></a><a name="p98651856183519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p144812376200"><a name="p144812376200"></a><a name="p144812376200"></a><span>含义：</span>machine运行时ctrlflow aicpu里控制stitch task处理最大Callop计算量</p>
<p id="p1448173722015"><a name="p1448173722015"></a><a name="p1448173722015"></a>说明：为了保障stitch task处理单次loop时的性能，需通过设置该配置项进行控制，该配置项设置的过大会带来额外的性能和内存开销，需根据算子最大Callop数量调整该配置项。若Callop数量超过该配置会报错提示：ASSERT FAILED：</p>
<p id="p1983173913481"><a name="p1983173913481"></a><a name="p1983173913481"></a>CallOpSize&lt;=CallOpmaxSize."loopFunction:&lt;function name&gt; ,CallopSize:&lt;当前Callop数量&gt;，CallOpmaxSize：&lt;配置项大小&gt;"</p>
<p id="p2448237162017"><a name="p2448237162017"></a><a name="p2448237162017"></a>类型：int</p>
<p id="p164487376208"><a name="p164487376208"></a><a name="p164487376208"></a>取值范围:1 ~ 65535</p>
<p id="p1544833762013"><a name="p1544833762013"></a><a name="p1544833762013"></a>默认值<span>：20000</span></p>
<p id="p1844853711203"><a name="p1844853711203"></a><a name="p1844853711203"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row068815515711"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p10672104292018"><a name="p10672104292018"></a><a name="p10672104292018"></a>cfgcache_device_task_num</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p2068825565718"><a name="p2068825565718"></a><a name="p2068825565718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p3688125513576"><a name="p3688125513576"></a><a name="p3688125513576"></a>含义：machine在运行时对控制流的缓存，配置用于估算缓存的 device task 的个数</p>
<p id="p9463463284"><a name="p9463463284"></a><a name="p9463463284"></a>说明：当该值为0 的时候，表示不做控制流缓存。该值仅影响控制流缓存的大小的估算。控制流缓存的大小实际限制了缓存的量。</p>
<p id="p2040322111253"><a name="p2040322111253"></a><a name="p2040322111253"></a>类型：int</p>
<p id="p674319258252"><a name="p674319258252"></a><a name="p674319258252"></a>取值范围：0~102400</p>
<p id="p112062032172519"><a name="p112062032172519"></a><a name="p112062032172519"></a>默认值：0</p>
<p id="p177461335192812"><a name="p177461335192812"></a><a name="p177461335192812"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row16147210175812"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p76897312217"><a name="p76897312217"></a><a name="p76897312217"></a>cfgcache_root_task_num</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p314781018588"><a name="p314781018588"></a><a name="p314781018588"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p2685101816304"><a name="p2685101816304"></a><a name="p2685101816304"></a>含义：machine在运行时对控制流的缓存，配置用于估算缓存的 root task 的个数</p>
<p id="p468511813303"><a name="p468511813303"></a><a name="p468511813303"></a>说明：该值仅影响控制流缓存的大小的估算。控制流缓存的大小实际限制了缓存的量。</p>
<p id="p1685191893017"><a name="p1685191893017"></a><a name="p1685191893017"></a>类型：int</p>
<p id="p1668511873018"><a name="p1668511873018"></a><a name="p1668511873018"></a>取值范围：0~102400</p>
<p id="p76851118103015"><a name="p76851118103015"></a><a name="p76851118103015"></a>默认值：0</p>
<p id="p106851018193014"><a name="p106851018193014"></a><a name="p106851018193014"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row1091920145816"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p176881838102111"><a name="p176881838102111"></a><a name="p176881838102111"></a>cfgcache_leaf_task_num</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p1992012116588"><a name="p1992012116588"></a><a name="p1992012116588"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 "><p id="p18776119173017"><a name="p18776119173017"></a><a name="p18776119173017"></a>含义：machine在运行时对控制流的缓存，配置用于估算缓存的 leaf task 的个数</p>
<p id="p15776201973015"><a name="p15776201973015"></a><a name="p15776201973015"></a>说明：该值仅影响控制流缓存的大小的估算。控制流缓存的大小实际限制了缓存的量。</p>
<p id="p18776111918302"><a name="p18776111918302"></a><a name="p18776111918302"></a>类型：int</p>
<p id="p3776131933016"><a name="p3776131933016"></a><a name="p3776131933016"></a>取值范围：0~102400</p>
<p id="p19776619163011"><a name="p19776619163011"></a><a name="p19776619163011"></a>默认值：0</p>
<p id="p577601919301"><a name="p577601919301"></a><a name="p577601919301"></a>影响pass范围：NA</p>
</td>
</tr>
<tr id="row912035622117"><td class="cellrowborder" valign="top" width="22.439999999999998%" headers="mcps1.1.4.1.1 "><p id="p1312155662110"><a name="p1312155662110"></a><a name="p1312155662110"></a>run_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.4.1.2 "><p id="p135201626162213"><a name="p135201626162213"></a><a name="p135201626162213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.04%" headers="mcps1.1.4.1.3 ">&nbsp;&nbsp;</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

void：Set方法无返回值。设置操作成功即生效。

## 约束说明<a name="section753174210543"></a>

无。

## 调用示例<a name="section646021318230"></a>

```
pypto.set_runtime_options(device_sched_mode=1,
                          stitch_function_inner_memory=10,
                          stitch_function_outcast_memory=60,
                          stitch_function_num_initial=30,
                          stitch_function_num_step=20)
```

