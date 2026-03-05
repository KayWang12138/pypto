# pypto.set\_runtime\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置runtime的选项。

**概念说明**：
- **Stitch**：将多个执行任务缝合成一个大任务统一下发，减少同步开销，提升调度性能。
- **Loop**：控制流中的循环结构，stitch会将多个loop的计算图动态组合到一起并行下发处理。

## 函数原型

```python
set_runtime_options(*,
                    device_sched_mode : int = None,
                    stitch_function_inner_memory: int = None,
                    stitch_function_outcast_memory : int = None,
                    stitch_function_num_initial : int = None,
                    stitch_function_max_num : int = None,
                    stitch_function_num_step : int = None,
                    stitch_function_size : int = None,
                    stitch_cfgcache_size: int = None,
                    run_mode : int = None,
                    valid_shape_optimize : int = None,
                    ) -> None
```

## 参数说明


| 参数名                         | 输入/输出 | 说明                                                         |
| ------------------------------ | --------- | ------------------------------------------------------------ |
| device_sched_mode              | 输入      | 含义：设置计算子图的调度模式 <br> 说明：<br> 0：FIFO共享队列模式，所有ready子图放入单一共享队列，多线程抢占式下发，严格先入先出。适用于对缓存命中率和线程公平性无特殊要求的场景，调度开销较低（默认值）。<br> 1：L2Cache亲和调度模式，优先选择"最新依赖ready"的子图下发，倾向最近激活链路，提升L2 cache复用率。适用于数据重用度高、时间局部性强的网络/子图，但可能出现部分线程/子图饥饿。<br> 2：公平调度模式，强调多线程调度多个aicore时的线程公平性，尽量均衡分配任务。适用于多线程调度存在"饿死"风险、需要较好公平性的场景，但会引入额外管理逻辑，调度开销略增。<br> 3：公平+L2Cache亲和混合模式，同时开启L2Cache亲和与公平调度，在保证一定公平性的前提下尽可能复用L2 cache。适用于希望兼顾缓存命中率与线程公平性的场景，但调度开销最高。<br> 优化建议：若对L2 cache命中率及调度线程间的公平性无特殊需求，推荐使用默认调度模式（即0）。 <br> 类型：int <br> 取值范围：0 或 1 或 2 或 3 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_inner_memory   | 输入      | 含义：控制root function中间计算结果的内存池大小的参数，内存池大小为max_root_nonoutcast_workspace（单个rootfunction的最大非outcast内存）。 <br> 说明：该数值越小，root function间越容易因workspace重叠导致互相产生依赖，导致无法并行；反之，该数值越大，通常stitch batch内并行度越高。<br> 注意：即将废弃，不建议再配置此项，请使用stitch_function_max_num替代。 <br> 类型：int <br> 取值范围：1~2147483647，当前版本最大有效值是 1024 * max_unroll_times，max_unroll_times 是算子代码里最大多分档档位。 <br> 默认值：128 <br> 影响pass范围：NA |
| stitch_function_outcast_memory | 输入      | 含义：控制stitch构建的devicetask中间计算结果（devitask内部rootfunction的outcast）的内存池大小的参数，内存池大小为maxOutcastWorkspace\*stitch_function_outcast_memory <br> 说明：设置的值代表该workspace允许将多少loop的计算图动态的stitch到一起并行下发处理，设置的值越大代表评估使用的workspace内存越大。<br> 注意：即将废弃，不建议再配置此项，请使用stitch_function_max_num替代。 <br> 类型：int <br> 取值范围:1 ~ 2147483647，当前版本最大有效值是 1024 * max_unroll_times，max_unroll_times 是算子代码里最大多分档档位。 <br> 默认值：128 <br> 影响pass范围：NA |
| stitch_function_num_initial    | 输入      | 含义：machine运行时ctrlflow aicpu里控制首个提交给schedule aicpu处理的device task的计算任务量 <br> 说明：设置的值代表第一个stitch task里处理的loop个数，通过此值来控制device machine启动头开销的大小，让ctrlflow aicpu和schedule aicpu计算尽快overlap起来。<br> 注意：即将废弃，不建议再配置此项，请使用stitch_function_max_num替代。 <br> 类型：int <br> 取值范围:1 ~ 1024 <br> 默认值：128 <br> 影响pass范围：NA |
| stitch_function_max_num        | 输入      | 含义：machine运行时ctrlflow aicpu里控制每次提交给schedule aicpu处理的最大device task的计算任务量 <br> 说明：设置的值代表每一个stitch task里处理的最大loop个数，该数值越大，通常stitch batch内并行度越高，相应的workspace内存使用也越大。数值越大，stitch内的任务并行度越高，同步开销越小，调度性能通常越好；但同时会带来控制流逻辑生成耗时增加和workspace内存占用增加的代价。<br> 优化建议：在内存使用允许的情况下，可通过适当增大该配置降低调度开销；但数值过大会带来控制流任务生成开销，因此建议逐步增大数据进行性能对比，找到当前硬件/模型的性能拐点。<br> 注意：此项配置会替代掉stitch_function_inner_memory、stitch_function_outcast_memory和stitch_function_num_initial三项。 <br> 类型：int <br> 取值范围:1 ~ 1024 <br> 默认值：128 <br> 影响pass范围：NA |
| stitch_function_num_step       | 输入      | 含义：machine运行时ctrlflow aicpu里控制非首次device task的计算任务量 <br> 说明：为了后续stitch task处理计算量平滑增加，可以通过设置此配置项进行控制。如设置为n，则每次stitch task里处理的loop次数分别base+n， base+2n 。。。 <br> 类型：int <br> 取值范围:0 ~ 1024 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_size           | 输入      | 含义：machine运行时ctrlflow aicpu里控制stitch生成的device task处理最大Callop计算量 <br> 说明：为了保障stitch task处理单次loop时的性能，需通过设置该配置项进行控制，该配置项设置的过大会带来额外的性能和内存开销，需根据算子最大Callop数量调整该配置项。若Callop数量超过该配置会报错提示：ASSERT FAILED：CallOpSize&lt;=CallOpmaxSize."loopFunction:&lt;function name&gt; ,CallopSize:&lt;当前Callop数量&gt;，CallOpmaxSize：&lt;配置项大小&gt;" <br> 类型：int <br> 取值范围:1 ~ 65535 <br> 默认值：20000 <br> 影响pass范围：NA |
| stitch_cfgcache_size           | 输入      | 含义：指定生成控制流缓存的大小，单位是字节 <br>说明：如果该值是0，则表示不使能控制流缓存。由于控制流缓存是按照任务大小来缓存，如果设置比较小，例如小于一个任务，那么无法缓存。<br>类型：int<br>取值范围：0~100000000<br>默认值：0<br>影响pass范围：NA |
| run_mode                       | 输入      | 含义：设置计算子图的执行设备 <br> 说明：<br> 0：表示在NPU上执行 <br> 1：表示在模拟器上执行 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：根据是否设置cann的环境变量来决定。如果设置了环境变量，则在NPU上执行；否则在模拟器上执行 <br> 影响pass范围：NA |
| valid_shape_optimize           | 输入      | 含义：动态shape场景，validshape编译优化选项，打开该选项后，动态轴的Loop循环中，主块（shape与validshape相等）采用静态shape编译，尾块采用动态shape编译 <br> 说明：<br> 0：默认值，表示关闭validshape编译优化选项，所有Loop循环均采用动态shape进行编译 <br> 1：表示打开validshape编译优化选项 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：0 <br> 影响pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
pypto.set_runtime_options(device_sched_mode=1,
                          stitch_function_inner_memory=128,
                          stitch_function_outcast_memory=128,
                          stitch_function_num_initial=128,
                          stitch_function_num_step=20)
@pypto.frontend.jit(
        runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_function_num_initial": 128,
        "device_sched_mode": 1
        }
)
```


## 参数调优案例

### device_sched_mode 调度模式对比（以 `examples/models/glm_v4_5/glm_attention.py` 为例）

- `"device_sched_mode": 0`  
  所有 ready 子图放入单一共享队列，多线程抢占式下发，严格先入先出，适合作为默认配置。  
  ![泳道图 - mode 0](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/39044413/67510cb51c5a4799862f0719fd01860a.png)

- `"device_sched_mode": 1`  
  优先选择“最新依赖 ready”的子图下发，更有利于 L2 cache 复用，适用于数据重用度高、时间局部性强的场景。  
  ![泳道图 - mode 1](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/39044639/ac53d5a4881b4bf7b9805aec1a595fd4.png)

- `"device_sched_mode": 2`  
  强调多线程调度多个 aicore 时的线程公平性，可避免部分线程“饿死”，适用于对公平性有要求的场景。  
  ![泳道图 - mode 2](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/39044832/6ec2356a1d9f4479b55e54dea825b0d9.png)

- `"device_sched_mode": 3`  
  同时开启公平与 L2Cache 亲和调度，兼顾缓存命中率与公平性，但调度开销最高。  
  ![泳道图 - mode 3](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/39045282/20bccca9192f4875a304dc40ff40dc5d.png)

综合建议：若对 L2 cache 命中率及调度线程间的公平性无特殊需求，推荐优先使用默认模式 `device_sched_mode=0`。

### stitch_function_max_num 调优案例（以 `examples/models/glm_v4_5/glm_attention.py` 为例）

该参数控制一次 stitch 能处理的最大 loop 数量，会同时影响调度开销、控制流生成耗时以及 workspace 内存占用。

- `runtime_options={"stitch_function_max_num": 1}`  
  device task 间非常离散，每个 task 跑完都要同步，调度开销大，性能较差。  
  算子端到端耗时 ≈ **1590 μs**。  
  ![泳道图 - max_num 1](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38985782/a8623875590f47edb7b028671f4e231a.png)  
  ![profiling - max_num 1](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38998057/aee22e7921d444439d7f379bd2a9c057.png)

- `runtime_options={"stitch_function_max_num": 128}`（默认值）  
  task 间相比 `max_num=1` 明显更紧凑，调度与同步开销大幅降低。  
  算子执行端到端耗时 ≈ **180 μs**，若包含冷启动与 stitch 控制流生成，总 profiling 耗时约 **803 μs**。  
  ![泳道图 - max_num 128](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38985789/94cc199e361a477eb62e9d402f4ebec2.png)  
  ![profiling - max_num 128](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38987157/a361f318a64f4e05b77aab49a913a8ef.png)

- `runtime_options={"stitch_function_max_num": 512}`  
  泳道图进一步收紧，算子执行端到端耗时降至约 **150 μs**，为该案例下算子侧最优。  
  但 stitch 控制流生成开销明显增大，profiling 总体耗时反而回升至约 **977 μs**。  
  ![泳道图 - max_num 512](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38985797/318ddeab13ec4fb7b67503d56226469c.png)  
  ![profiling - max_num 512](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026020610093254/38987490/9e31a9dcb5744626b835291e7f7ecfcc.png)

调优建议：在内存资源允许的前提下，可逐步增大 `stitch_function_max_num`，结合端到端耗时与 profiling 结果，寻找在性能收益与控制流开销之间的最佳平衡点。


