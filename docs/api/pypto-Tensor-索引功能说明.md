# pypto.Tensor 索引功能说明<a name="ZH-CN_TOPIC_0000002490456756"></a>

Tensor（张量）索引是核心的张量操作之一，用于从张量中筛选、提取或修改特定位置的元素。通过索引操作，开发者可精准获取张量中的部分数据（如单个元素、子张量、特定维度数据），或对指定位置元素进行赋值修改。

## 一、\_\_getitem\_\_<a name="section1322102618017"></a>

## 功能说明<a name="section1194520820284"></a>

通过索引或切片的方式从张量中获取Tensor，该方法支持多种索引模式，提供了灵活且直观的数据访问方式。

## 函数原型<a name="section12823182962820"></a>

```
def __getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## 参数说明<a name="section32851234132519"></a>

<a name="table770022613375"></a>
<table><thead align="left"><tr id="row187011726173718"><th class="cellrowborder" valign="top" width="15.541554155415543%" id="mcps1.1.4.1.1"><p id="p1701122663712"><a name="p1701122663712"></a><a name="p1701122663712"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.731173117311732%" id="mcps1.1.4.1.2"><p id="p47011268375"><a name="p47011268375"></a><a name="p47011268375"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.72727272727272%" id="mcps1.1.4.1.3"><p id="p107011926163713"><a name="p107011926163713"></a><a name="p107011926163713"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row7701226183719"><td class="cellrowborder" valign="top" width="15.541554155415543%" headers="mcps1.1.4.1.1 "><p id="p1701172673715"><a name="p1701172673715"></a><a name="p1701172673715"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.1.4.1.2 "><p id="p370112610376"><a name="p370112610376"></a><a name="p370112610376"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.72727272727272%" headers="mcps1.1.4.1.3 "><p id="p129021270333"><a name="p129021270333"></a><a name="p129021270333"></a>Tensor索引，用于获取Tensor对应位置的数据。</p>
<p id="p1718773153511"><a name="p1718773153511"></a><a name="p1718773153511"></a>支持类型：</p>
<p id="p28308286240"><a name="p28308286240"></a><a name="p28308286240"></a>int 或 SymbolicScalar（符号标量）: 单个整数索引。</p>
<p id="p68301628112410"><a name="p68301628112410"></a><a name="p68301628112410"></a>slice: 切片对象。</p>
<p id="p8894441163210"><a name="p8894441163210"></a><a name="p8894441163210"></a>tuple: 多维索引的组合，类型包括：int 或 SymbolicScalar，slice，Ellipsis(...)。</p>
</td>
</tr>
<tr id="row149411934192218"><td class="cellrowborder" valign="top" width="15.541554155415543%" headers="mcps1.1.4.1.1 "><p id="p694193415227"><a name="p694193415227"></a><a name="p694193415227"></a>valid_shape</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.1.4.1.2 "><p id="p1794117345229"><a name="p1794117345229"></a><a name="p1794117345229"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.72727272727272%" headers="mcps1.1.4.1.3 "><p id="p14942134192218"><a name="p14942134192218"></a><a name="p14942134192218"></a><span>表示输出Tensor有效数据的大小</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section20685185616257"></a>

返回对应索引位置的Tensor数据。

## 约束说明<a name="section753174210543"></a>

1.对于slice（切片对象，格式为 start:end:step），当前功能暂时不支持step设置, 值默认固定为1。

未支持示例：a\[1:2:2, :\] 。

2.当前功能暂时不支持bool类型索引。

未支持示例：a\[True, False, True, False\] 。

3.当前功能暂时不支持Tensor类型索引。

未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)。

## 使用示例<a name="section150449134214"></a>

1.全切片（slice）

使用切片获取张量的子区域

```
a = pypto.tensor([4, 4], pypto.DT_FP32)
b = a[:2, :2] #等价于view(a, [2, 2], [0, 0])
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输出数据b: [[1, 2],
            [5, 6]]
```

2.混合索引和切片

结合整数索引和切片，可以降低维度并提取特定行或列。

```
a = pypto.tensor([4, 4], pypto.DT_FP32)
b = a[1, 1:3] #等价于先view(a, [1, 2], [1, 1])，再reshape为 [2]
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输出数据b: [6, 7]
```

3. 负数索引

支持 Python 风格的负索引，从末尾开始计数。

```
a = pypto.tensor([4, 4], pypto.DT_FP32)
b = a[-1, -3:-1] #等价于s[3, 1:3]
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输出数据b: [14, 15]
```

4. 省略号（ ...）

使用 \`...\` 自动填充中间的所有维度，简化多维索引。

```
a = pypto.tensor([4, 4], pypto.DT_FP32)
b = a[..., 1:3] #等价于s[:, 1:3]
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输出数据b: [[2, 3],
            [6, 7],
            [10, 11],
            [14, 15]]
```

5. 单元素访问

整数索引，取出Tensor的单个元素（仅支持 DT\_INT32 类型）。

```
a = pypto.tensor([4, 4], pypto.DT_INT32)
b = a[0, 0] #返回SymbolicScalar
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输出数据b: 1
```

6. Gather 操作

当key为slice，key.start为int，key.stop为Tensor时\(a\[start:stop\]\)，执行gather操作。

```
a = pypto.tensor([4, 4], pypto.DT_FP32)
index = pypto.tensor([1, 4], pypto.DT_INT32)
b = a[0:index] #调用gather(a, 0, index)
```

结果示例如下：

```
输入数据a: [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
输入数据index: [[0, 1, 2, 3]]
输出数据b: [[1, 6, 11, 16]]
```

## 二、\_\_setitem\_\_<a name="section111143812465"></a>

## 功能说明<a name="section952219911918"></a>

通过索引或切片的方式向Tensor的指定位置赋值。

## 函数原型<a name="section926013261390"></a>

```
def __setitem__(self, key, value)
```

## 参数说明<a name="section0107174012911"></a>

<a name="table237713547910"></a>
<table><thead align="left"><tr id="row1937711541396"><th class="cellrowborder" valign="top" width="15.541554155415543%" id="mcps1.1.4.1.1"><p id="p737745412915"><a name="p737745412915"></a><a name="p737745412915"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.731173117311732%" id="mcps1.1.4.1.2"><p id="p15377185419915"><a name="p15377185419915"></a><a name="p15377185419915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.72727272727272%" id="mcps1.1.4.1.3"><p id="p11377135411911"><a name="p11377135411911"></a><a name="p11377135411911"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row133778547914"><td class="cellrowborder" valign="top" width="15.541554155415543%" headers="mcps1.1.4.1.1 "><p id="p63772541692"><a name="p63772541692"></a><a name="p63772541692"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.1.4.1.2 "><p id="p14377195414910"><a name="p14377195414910"></a><a name="p14377195414910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.72727272727272%" headers="mcps1.1.4.1.3 "><p id="p9377254699"><a name="p9377254699"></a><a name="p9377254699"></a>Tensor索引，用于获取Tensor对应位置的数据。</p>
<p id="p392925115355"><a name="p392925115355"></a><a name="p392925115355"></a>支持类型：</p>
<p id="p2377115416912"><a name="p2377115416912"></a><a name="p2377115416912"></a>int 或 SymbolicScalar（符号标量）: 单个整数索引。</p>
<p id="p19377185419914"><a name="p19377185419914"></a><a name="p19377185419914"></a>slice: 切片对象。</p>
<p id="p637755416915"><a name="p637755416915"></a><a name="p637755416915"></a>tuple: 多维索引的组合，类型包括：int 或 SymbolicScalar，slice，Ellipsis(...)。</p>
</td>
</tr>
<tr id="row737712541291"><td class="cellrowborder" valign="top" width="15.541554155415543%" headers="mcps1.1.4.1.1 "><p id="p2037765410912"><a name="p2037765410912"></a><a name="p2037765410912"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.1.4.1.2 "><p id="p437718543914"><a name="p437718543914"></a><a name="p437718543914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.72727272727272%" headers="mcps1.1.4.1.3 "><p id="p2377195412913"><a name="p2377195412913"></a><a name="p2377195412913"></a>需要设置的值,<span> 类型支持Tensor或标量（float/int）</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1060318388139"></a>

返回对应位置赋值后的Tensor。

## 约束说明<a name="section1034511326572"></a>

1.对于slice（切片对象，格式为 start:end:step），当前功能暂时不支持step设置, 值默认固定为1。

未支持示例：a\[1:2:2, :\] 。

2.当前功能暂时不支持bool类型索引。

未支持示例：a\[True, False, True, False\] 。

3.当前功能暂时不支持Tensor类型索引。

未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)。

## 使用示例<a name="section942820164910"></a>

1. 全切片（slice）

使用切片将一个小张量组装到大张量的指定位置。

```
a = pypto.Tensor([4, 4], pypto.DT_FP32)
b = pypto.Tensor([2, 2], pypto.DT_FP32)
a[0:, 0:] = b #等价于assemble(b, (0, 0), a)
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输入数据b: [[10, 10]
            [10, 10]]
输出数据a: [[10, 10, 0, 0],
            [10, 10, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
```

2. 混合索引和切片

结合整数索引和切片，可以对特定行或列进行操作。

```
a = pypto.Tensor([4, 4], pypto.DT_FP32)
b = pypto.Tensor([2], pypto.DT_FP32)
a[0, 1:3] = b #b被reshape为(1, 2),等价于pypto.assemble(b, (0, 1), a)
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输入数据b: [10, 10]
输出数据a: [[0, 10, 10, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
```

3. 负索引

支持 Python 风格的负索引，从末尾开始计数。

```
a = pypto.Tensor([4, 4], pypto.DT_FP32)
b = pypto.Tensor([2], pypto.DT_FP32)
a[-1, -3:-1] = b #等价于a[3, 1:3]
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输入数据b: [10, 10]
输出数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 10, 10, 0]]
```

4. 省略号（ ...）

使用 ... 可以自动填充中间维度。

```
a = pypto.Tensor([4, 4], pypto.DT_FP32)
b = pypto.Tensor([2, 2], pypto.DT_FP32)
a[..., 2:4] = b #等价于a[0:2, 2:4]
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输入数据b: [[10, 10]
            [10, 10]]
输出数据a: [[0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
```

5. 单元素赋值

整数索引，对单个元素赋值（仅支持 DT\_INT32 类型）。

```
a = pypto.Tensor([4, 4], pypto.DT_INT32)
a[2, 3] = 5 #调用SetTensorData
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输出数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 5],
            [0, 0, 0, 0]]
```

6. Scatter 操作

当key为slice，key.start为int， key.stop 为 Tensor 时\(a\[start:stop\]\)，执行 scatter 操作。

```
a = pypto.Tensor([4, 4], pypto.DT_FP32)
indices = pypto.Tensor([1, 4], pypto.DT_INT32) # 索引张量
values = pypto.Tensor([1, 4], pypto.DT_FP32)
# 在维度0上进行scatter
a[0:indices] = values #调用pypto.scatter(a, 0, indices, values)
```

结果示例如下：

```
输入数据a: [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
输入数据indices：[[0, 1, 2, 3]]
输入数据values：[[10, 10, 10, 10]]
输出数据a: [[10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]]
```

