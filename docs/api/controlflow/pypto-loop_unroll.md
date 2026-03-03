# pypto.loop\_unroll

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

pypto.loop\_unroll是一个支持循环展开的循环迭代器函数，功能与pypto.loop类似，增加了unroll\_list参数支持多个展开方式。在动态shape下，框架会根据unroll_list选择合适的档位或组合，避免冗余计算，以实现高性能。

### 核心特性
- 多档位展开；unroll\_list支持配置多个展开因子，框架会静态构建每个展开因子对应的Tensor Graph，并在执行时选择最优的路径执行
- 智能展开算法；展开因子按照从大到小排序并且去重，且总是会包含因子1作为保底策略
- 动态档位匹配；执行时，根据输入shape的大小，自动匹配最优的档位；当动态轴的shape值为127，unroll\_list={32,16,8,4,2,1}时，会以此匹配32、32、32、16,、8、4、2、1档位，优先选择大的档位

### 适用场景

loop_unroll适用于动态shape 且shape范围广的场景；当算子中某个动态轴需要泛华支持1-64k的shape范围时，指定单一的动态轴切分大小难以满足要求。切分过大时，小shape场景会引入较多实际计算大小为0的空计算任务，增加耗时；切分过小时，大shape场景下循环次数过多，影响整体性能；引入loop_unroll后，大小shape场景都可以选择合适的档位进行组合，无空计算仍无，且循环次数可控，可获得较好的性能。

### 性能优势
- 增大调优空间；多档位展开后可以得到更大的循环体，配合合适的tile_shape，可以使得任务间得到更高的并行度，给框架的切图、调度等算法更大调优空间；同时不同的展开因子可以配置不同的调优参数，可以灵活调优
- 减少冗余计算；不论大小shape场景，都可以选择合适的档位进行组合，不会引入空任务，无多余计算耗时
- 减少调度开销；增加展开的档位可降低循环次数，减少调度开销。


## 函数原型

```python
loop_unroll(*args, **kwargs) -> Iterator[Tuple[SymInt, int]]
```

## 参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| *args             | 输入      | 三个可选参数，分别为循环起始值（start），循环结束值（stop），循环步长（step），有以下三种写法：<br> - 单参数形式：stop(SymInt)，起始值默认为0，步长默认为1。等价于：loop_unroll(0, stop, 1)<br> - 双参数形式：start(SymInt)，stop(SymInt)，等价于loop_unroll(start, stop, 1)<br> - 三参数形式：start (SymInt)，stop(SymInt)，step(SymInt)，等价于loop_unroll(start, stop, step) |
| **kwargs          | 输入      | - name(str)：循环标识名称，默认生成f"loop_{loop_idx}"。<br> - idx_name(str):  循环索引变量的名称，默认生成f"loop_idx_{loop_idx}"。<br> - unroll_list(List[int]):  需要展开unroll的循环层数集合，默认为空集合。loop会提供等于该集合长度的几种展开方式，展开次数为n时，循环步长会变成step*n，每次迭代会执行n次循环体。每种展开次数会生成不同的代码路径。<br> - submit_before_loop(bool):  是否在循环开始前提交计算，默认为False。开启后会在循环开启前强制提交当前累积的计算任务到AICore执行。 |

## 返回值说明

返回一个迭代器，每次迭代产生一个元组（idx, unroll\_factor\)

-  idx：当前循环的索引值
-  unroll\_factor：当前选择的展开档位

## 约束说明

-   展开因子列表会被排序并去重，且总是包含 1
-   展开因子按从大到小排序
-   每个展开因子会生成一个子循环

## 调用示例

```python

'''
input：A， shape:[-1, 64]
output：B， shape:[-1, 64]

-1：表示动态shape
'''
.....
for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[1, 2, 4, 128], name="A", idx_name='b'):
   ### 支持在不同的展开档位设置不同调优参数
   if k <= 4:
      pypto.set_vec_tile_shapes(4, 64)
   else :
      pypto.set_vec_tile_shapes(128, 64)

   tile_a = A[b * 64:(b + k) * 64, :]  #
   tile_a = tile_a + 2
   B[b * 64:, :] = tile_a
```
