# pypto.Tensor构造函数<a name="ZH-CN_TOPIC_0000002505295629"></a>

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

创建Tensor对象。Tensor创建时为未初始化的随机值。

## 函数原型<a name="section1814166202715"></a>

```
__init__(self, 
         shape=None, 
         dtype: Union[DataType, None] = None,
         name: str = "", 
         format: TileOpFormat = TileOpFormat.TILEOP_ND,
         data_ptr: Optional[int] = None, 
         device=None, 
         ori_shape=None
)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p231475933411"><a name="p231475933411"></a><a name="p231475933411"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1631319595342"><a name="p1631319595342"></a><a name="p1631319595342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p731345933418"><a name="p731345933418"></a><a name="p731345933418"></a>Tensor的形状，可以是以下类型：</p>
<a name="ul19286332184014"></a><a name="ul19286332184014"></a><ul id="ul19286332184014"><li>None：创建空Tensor</li><li>List[int]：整数列表，指定各维度的大小</li><li>List[Union[int, SymbolicScalar]]：包含整数或符号标量的列表，用于动态形状</li></ul>
</td>
</tr>
<tr id="row6253333118"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p12312165913346"><a name="p12312165913346"></a><a name="p12312165913346"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p2311959113416"><a name="p2311959113416"></a><a name="p2311959113416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p1531135953414"><a name="p1531135953414"></a><a name="p1531135953414"></a>Tensor的数据类型。</p>
</td>
</tr>
<tr id="row360415107522"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p173101859133413"><a name="p173101859133413"></a><a name="p173101859133413"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p1731005910343"><a name="p1731005910343"></a><a name="p1731005910343"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p33094599341"><a name="p33094599341"></a><a name="p33094599341"></a>Tensor的名称。</p>
</td>
</tr>
<tr id="row1923214159546"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p12308105910344"><a name="p12308105910344"></a><a name="p12308105910344"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p5308125910347"><a name="p5308125910347"></a><a name="p5308125910347"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p123071359173418"><a name="p123071359173418"></a><a name="p123071359173418"></a>Tensor的格式，可选值包括：</p>
<a name="ul3613155318439"></a><a name="ul3613155318439"></a><ul id="ul3613155318439"><li>TileOpFormat.TILEOP_ND(默认)</li><li>TileOpFormat.TILEOP_NZ</li></ul>
</td>
</tr>
<tr id="row1937504018380"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p183761140163813"><a name="p183761140163813"></a><a name="p183761140163813"></a>data_ptr</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p73761840193816"><a name="p73761840193816"></a><a name="p73761840193816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p337654073815"><a name="p337654073815"></a><a name="p337654073815"></a>数据指针，用于指定外部数据的内存地址，默认为None</p>
</td>
</tr>
<tr id="row1452611497383"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p20526194903814"><a name="p20526194903814"></a><a name="p20526194903814"></a>device</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p0526104910385"><a name="p0526104910385"></a><a name="p0526104910385"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p4526249103811"><a name="p4526249103811"></a><a name="p4526249103811"></a>设备信息，默认为None</p>
</td>
</tr>
<tr id="row18738195163817"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p47389515385"><a name="p47389515385"></a><a name="p47389515385"></a>ori_shape</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p973825123817"><a name="p973825123817"></a><a name="p973825123817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.1.4.1.3 "><p id="p37388512380"><a name="p37388512380"></a><a name="p37388512380"></a>原始形状，用于保存张量的原始形状信息，默认为None</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

Tensor对象。

## 约束说明<a name="section753174210543"></a>

无。

## 调用示例<a name="section646021318230"></a>

```
# 创建空张量 
empty_tensor = pypto.Tensor()  

# 创建指定形状和数据类型的张量 
tensor1 = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32) 
tensor2 = pypto.Tensor(shape=[8, 16, 32], dtype=pypto.DT_INT32)  

# 创建带名称的张量 
named_tensor = pypto.Tensor(shape=(4, 4),      
                            dtype=pypto.DT_FP32,      
                            name="input_tensor" )  

# 创建指定格式的张量 
sparse_tensor = pypto.Tensor(shape=(4, 32),      
                             dtype=pypto.DT_FP32,       
                             format=pypto.TileOpFormat.TILEOP_NZ )  

# 创建动态形状张量（使用符号标量） 
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8] 
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,      
                              dtype=pypto.DT_FP32 )  

# 使用 pypto.tensor 便捷函数创建（推荐方式） 
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32) 
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```

