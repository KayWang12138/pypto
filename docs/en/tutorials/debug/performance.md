# Performance Optimization

## Introduction

During the development of fused operators, operator performance optimization is typically extremely challenging. This chapter aims to introduce how to use the PyPTO Toolkit visualization tool to help developers write high-performance fused operators without needing to deeply understand hardware implementation details.

## Overall Workflow

After completing precision debugging, developers can use PyPTO Toolkit to view the swimlane graph for the corresponding operator. The swimlane graph visually displays the actual scheduling and execution process of the computation graph, clearly showing task execution order and timing information. Based on this, developers can obtain the initial performance of the operator they have written, also referred to as out-of-the-box performance.

By observing the swimlane graph, developers can identify performance bottlenecks in the current operator implementation. After identifying bottlenecks, they can adjust tiling configurations and adopt different computation graph compilation strategies to obtain the swimlane graph under the new implementation, make further adjustments, and gradually improve operator performance — implementing a Man-In-The-Loop (human-in-the-loop) optimization workflow.

## Performance Optimization Tools

### Collecting Swimlane Graph Data

1.  Enable the performance data collection feature by configuring the graph execution debug switch via the debug_options parameter of the @pypto.frontend.jit decorator.

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2.  Run the test case.

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3.  Generate the swimlane graph JSON file.

    The file merged\_swimlane.json is generated in the output/output\_timestamp directory under the current working directory. This file is the swimlane graph data file.

### Viewing Swimlane Graph Data

1.  View the swimlane graph using the PyPTO Toolkit plugin.

    Right-click the JSON file and select "Open with PyPTO Toolkit" from the pop-up menu, as shown below.

    **Figure 1**  Viewing the Swimlane Graph
    ![](../figures/view_swimlane_graph.png "Viewing the Swimlane Graph")

    The graph shows task execution order and timing information, helping developers analyze performance bottlenecks.

## Out-of-the-Box Performance Optimization

The initial operator performance is most closely related to how loops are written and how TileShape is set. This chapter introduces how to use the relevant interfaces to achieve good out-of-the-box performance directly during the initial operator development process. Refer to the implemented operators in the models folder of the PyPTO repository when developing new operators.

### Choosing the Correct Loop Style

Because subgraphs from different root functions cannot be merged, and subgraph merging is the key means by which PyPTO optimizes performance, the core principle of loop optimization is: **increase the size of each root function and reduce their total number**.

#### Use Python for Loops for Static Axes

The pypto.loop method expands each axis iteration into different root functions. Therefore, loops over static axes should use Python's for loop, avoiding PyPTO's loop.

   ```python
   # Recommended: use Python for on static axes
   for i in range(batch_size):
       result[i] = process(data[i])

   # Avoid: use PyPTO loop on static axes
   for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
       result[i] = process(data[i])
   ```

For the optimization effect, refer to section 3.3.1 of the [GDR operator case](./performance_case_GDR.md).

#### Use PyPTO loop with Properly Configured view for Dynamic Axes

When an operator has dynamic shapes, the dimension range of dynamic axes is often wide and requires loop processing. In this case, pay attention to the view parameter configuration: the selected shape range must not be too small, otherwise it will limit the subsequent TileShape configuration range, causing the amount of computation per loop iteration to be too small, increasing the number of loop iterations, and potentially causing additional redundant transfers for some operations. For example, in matmul operations, a too-small TileShape causes many root functions to transfer the same left or right matrix. An example with a view TileShape of 128 is as follows:

   ```python
   # Recommended: use loop + unroll for dynamic axes
   bsz, h = x.shape
   b = 128
   b_loop = (bsz + b - 1) // b
   for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
       b_valid = (bsz - b_idx * b).min(b)
       x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
       # Matmul
       pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
       y = pypto.matmul(x_view, W)
   ```

#### Merge Loops Where Possible

Check whether there are loop blocks in the operator code that can be merged, and merge them to increase the root function size. For example, the following two loops can be merged, which increases the likelihood of merging Operation1 and Operation2 operations and reduces redundant transfers of y.

```
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
       out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
       out_2 = Operation2(x2[b_idx, :], y)
```

#### Use [loop_unroll](../../api/controlflow/pypto-loop_unroll.md) When the Dynamic Axis Range Is Wide

When an operator uses dynamic shapes with a wide shape range, consider using loop_unroll instead of the loop interface. The loop_unroll feature is similar to loop but adds support for the unroll_list parameter with multiple expansion modes. For example, when a dynamic axis in an operator needs to generalize and support a shape range of 1 to 64k, specifying a single dynamic axis partition size is difficult to satisfy. When the partition is too large, small-shape scenarios introduce many computation tasks whose actual computation size is 0, increasing latency. When the partition is too small, large-shape scenarios have too many loop iterations, affecting overall performance. With loop_unroll, regardless of the shape size, the framework selects an appropriate level or combination based on the unroll_list parameter, avoiding redundant computation, keeping the number of loop iterations controllable, and thus achieving better performance.

Pay attention to the following when using this feature:

- As the number of levels configured in the unroll_list parameter increases, the amount of work that needs to be processed during compilation doubles, causing compilation time to increase. Therefore, when initially writing an operator, it is recommended to use a shorter unroll_list, such as [64, 16, 4].
- When using dynamic level selection, note that different levels may require separately selected appropriate TileShapes.
- In multi-layer loop nesting scenarios, only the innermost loop_unroll can successfully use the unroll_list parameter.

Reference example:

```python

'''
input: A, shape: [-1, 64]
output: B, shape: [-1, 64]

-1: indicates dynamic shape
'''
.....
for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[64, 16, 4], name="A", idx_name='b'):
   ### Supports setting different optimization parameters for different expansion levels
   if k <= 16:
      pypto.set_vec_tile_shapes(16, 64)
   else :
      pypto.set_vec_tile_shapes(64, 64)

   tile_a = A[b * 64:(b + k) * 64, :]  #
   tile_a = tile_a + 2
   B[b * 64:, :] = tile_a
```

For the optimization effect, refer to section 3.3.3 of the [GDR operator case](./performance_case_GDR.md).

### Setting Reasonable Initial TileShape Values

The basic principles and usage constraints for TileShape configuration are described in the [Tiling Configuration](../development/tiling.md) chapter. The tile size directly determines the number of tasks after operator tiling, which in turn determines the number of cores used and the number of computation rounds during actual execution. At the same time, the tile size theoretically determines the arithmetic intensity of the operator. Therefore, the key to optimizing performance is to optimize the tiling configuration.

In general, a larger tile size leads to higher arithmetic intensity and makes it easier for computation to reach the Compute Bound, thereby fully leveraging the NPU's compute power. This is because tiling inevitably introduces redundant transfers — more tiling means more redundant transfers and lower arithmetic intensity. On the other hand, tile size is also constrained by the capacity of the on-chip multi-level caches (L1, L0, or UB) and cannot be increased indefinitely.

#### Initial Tiling Configuration for Matmul

For matrix computation scenarios, using DT_BF16 or DT_FP16 data types for both the A and B matrices as an example, the recommended tiling configuration that satisfies buffer space constraints is:
```python
# Recommended TileShape for Cube-related computations; choose the closest configuration based on actual M, K, N sizes:
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

Advantages of the above tiling configurations:
- Can achieve high arithmetic intensity while satisfying L0 Buffer constraints. Since tile sizes must satisfy the alignment requirements of the fractal format, and the effect of tile size on write bandwidth must also be considered, combinations of 128-256 are generally chosen.
- When further performing depth optimization using graph fusion interfaces, there is an opportunity to enable Double Buffer and enable pipeline parallelism.

#### Initial Tiling Configuration for Vector

For vector computation scenarios, the appropriate TileShape should be determined based on the Operations and the chip's UB size.

- First, the TileShape specification constraints for specific Operations must be satisfied. For example, scatter update requires that the tail-axis TileShape is the same as the shape (i.e., no tiling on the tail axis). For specific limitations of each Operation, refer to the API description in the PyPTO documentation.

- Second, the input and output tensors of the Operation must be allocatable in UB memory, so TileShape must not be too large. At the same time, since small subgraphs and small data blocks for transfers can degrade performance, TileShape must not be too small either. Taking Atlas A3 Training Series as an example, the UB cache capacity is 192 KB. Therefore, a suitable initial TileShape satisfies both the Operation requirements and keeps the data block size between 16 and 64 KB, with the tail axis aligned to 32 B.

- Additionally, for reduction-type computations (reduce operations such as sum, max, min, etc.), try to avoid tiling along the reduction axis. For example, for an RMSNorm with input shape (56, 1024), the TileShape of the last dimension should be set to 1024. The upper half of the figure below shows an example of a swimlane graph for RMSNorm with tiling on the reduce axis — outputs from multiple subgraphs need to be reduced in the same subgraph, leading to GM transfers and scheduling overhead. The lower half shows an example without tiling on the reduce axis; here, upstream and downstream subgraphs are merged, with no GM transfers and no scheduling overhead.

![Example of tiling on the reduce axis](../figures/perf_reduce.png)

Considering all the above, the following setting can be used as an initial starting point during the initial operator development stage, with further optimization based on swimlane graph data.

```python
# Recommended TileShape for Vector-related computations:
pypto.set_vec_tile_shapes(64, 512)
```

For the actual optimization effect of TileShape tuning, refer to section 3.2 of the [GDR operator case](./performance_case_GDR.md).

It should be noted that the above tiling configurations are not fixed; users need to make comprehensive decisions based on the computation scenario (considering input shape, dtype, format, etc.) and the hardware platform.

### Other Notes

- Check whether input matrices, especially large weight matrices, can be stored in NZ format in advance. Data in NZ format is transferred to L1 with higher bandwidth.

- When there is a transpose before or after matmul, try swapping the left and right matrices and using transpose configurations for both. Since the N axis is on the tail axis, when the M axis is large and the N axis is small, this approach can also be tried to give the left and right matrices a larger tail axis and improve transfer bandwidth.

- Check whether there are redundant transfers caused by unreasonable data operations, such as replacing concat with assemble or trying to configure `inplace = True` for reshape.

## Deep Performance Optimization

Further optimizing operator performance requires a man-in-loop approach: obtain and analyze the current operator's performance data, adjust performance configuration parameters in a targeted manner, and iteratively tune toward the optimal performance. Operator performance data can be obtained through swimlane graphs. The collection and analysis of swimlane graphs is an important part of the operator optimization process, and the optimization process in this chapter needs to be carried out in conjunction with swimlane graphs.

### Stitch Optimization

[Stitch](../appendix/glossary.md) configuration determines how many root functions are dispatched for scheduling simultaneously — this parameter controls the maximum number of loops that one stitch can process at a time, and it affects both scheduling overhead, control flow generation time, and workspace memory usage. A larger Stitch setting allows tasks to be fully parallelized, which generally leads to better performance. When a large number of gaps appear in the swimlane graph, this may be caused by a Stitch value that is too small.

The current Stitch configuration is primarily determined by the [stitch_function_max_num](../../api/config/pypto-jit.md#runtime_options_detail) parameter, which is configured in the jit decorator. A reference configuration is as follows:

```python
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128}
    )
```

Taking the [glm_attention.py](../../../models/glm_v4_5/glm_attention.py) operator as an example, compare the effect of this value:

- When this value is too small (e.g., set to 1), each task requires synchronization, the scheduling overhead is large, and performance is poor. The operator kernel latency is 1230 us, and the end-to-end latency (scheduling + execution) is 1590 us. The corresponding swimlane graph is as follows:

![Swimlane graph - Stitch-1](../figures/stitchnum11.png)
![Trace graph - Stitch-1](../figures/stitchnum12.png)

- When this value is increased to 128, the swimlane graph is noticeably more compact, and scheduling and synchronization overhead decreases significantly. The operator kernel latency is 180 us, and the end-to-end latency is 803 us. The corresponding swimlane graph is as follows:

![Swimlane graph - Stitch-128](../figures/stitchnum21.png)
![Trace graph - Stitch-128](../figures/stitchnum22.png)

- When this value is further increased to 512, the swimlane graph is even more compact, but scheduling latency increases noticeably. The operator kernel latency is 150 us, and the end-to-end latency is 977 us. The corresponding swimlane graph is as follows:

![Swimlane graph - Stitch-512](../figures/stitchnum31.png)
![Trace graph - Stitch-512](../figures/stitchnum32.png)

As can be seen, as the Stitch configuration increases, the operator kernel latency continues to decrease. However, a larger Stitch configuration is not always better:

- Excessively large values cause scheduling latency to increase noticeably, leading to diminishing returns in end-to-end latency.

- A larger Stitch setting also increases workspace usage. With many tasks running in parallel, the L2 cache hit rate may be lower.

Optimization recommendation: When memory resources allow, gradually increase the Stitch configuration and adjust the `stitch_function_max_num` parameter in combination with swimlane graph and end-to-end total latency data. Find the optimal balance between performance gains and control flow overhead to reduce total latency.

For the actual effect, refer to section 3.1 of the [GDR operator case](./performance_case_GDR.md).

### TileShape Optimization

#### Matmul TileShape Optimization

Further optimizing the matmul TileShape requires fully considering the impact on arithmetic intensity and bandwidth. For details, refer to the [Matmul High-Performance Programming](./matmul_performance_guide.md) chapter.

This section focuses on two optimization techniques: **reducing redundant loads** and **K-axis core splitting**. These correspond to the `enable_split_k` configuration parameter of the `set_cube_tile_shapes` interface. Users can derive and select an appropriate switch configuration strategy based on the principles described above, or directly test and verify using swimlane graph data and configure the optimal option. The two parameters are decoupled from each other. A reference configuration is as follows:

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

#### Vector TileShape Optimization

In addition to the principles described in previous sections, Vector TileShape configuration also requires attention to the following points in conjunction with swimlane graphs:

- The TileShape of downstream Vector Operations should use the output TileShape of upstream Operations as much as possible. For example, when a Transpose is followed by an Add Operation, if the former's TileShape is set to (64, 128), the latter's TileShape should preferably be set to (128, 64). When the TileShapes of upstream and downstream Operations are aligned, they have a simple one-to-one dependency, and passes will typically automatically merge them into a single subgraph, achieving the optimization effect of a fused operator. If they are not merged, use the sg_set_scope or graph-cutting knobs described earlier to merge them. When the TileShapes of upstream and downstream Operations are not aligned, many-to-many dependencies may occur, preventing normal graph merging. As shown in the figure below, each color represents a type of subgraph. When the preceding and following Sqrt and Cast Operations use the same TileShape, two parallel fused subgraphs can be produced. Conversely, when the Sqrt and Cast Operations use different TileShapes, the upstream and downstream subgraphs have a three-to-two dependency, and parallel fused subgraphs cannot be obtained.

![alt text](../figures/perf_tilesize.png)

- Adjust TileShape based on subgraph sizes and parallelism in the swimlane graph. Under the target optimization scenario, when the number of parallel cores for a certain part of the swimlane graph is small (e.g., fewer than half the Vector cores are used), try reducing the TileShape of the Operation at that location. Conversely, when certain subgraphs in the swimlane graph have short durations and the proportion of scheduling overhead is high, try increasing the TileShape of the Operation at that location. Note that these adjustments should avoid the tail axis and reduce axis, as the TileShape of these axes should conform to the optimization principles described earlier. Also, TileShape optimization may cause OoO errors due to graph merging, in which case the related Operation should be found, its TileShape reduced, and then the third optimization above applied again.

- Adjust the TileShapes of adjacent Cube and Vector Operations to simplify the dependencies between Cube subgraphs and Vector subgraphs, and try to avoid many-to-many dependencies.

### Graph Fusion Optimization

Before optimizing graph fusion–related knobs, TileShape optimization must be completed first, because an inappropriate TileShape will cause complex dependency relationships and theoretically prevent obtaining subgraph tasks that are both multi-core parallel and fuse multiple Operations.

Graph fusion refers to the process of merging multiple logically independent Operations in the computation graph into a single logical subgraph, which then generates a single physical compute kernel. Deep learning model computation graphs are often composed of a large number of fine-grained Operations. In the traditional per-Operation execution mode, each Operation independently triggers one kernel launch, and intermediate results are written back to global memory (GM) after computation is complete. This execution method introduces significant kernel launch overhead and redundant memory accesses on real hardware, making it difficult to fully leverage the compute units' full capabilities. Graph fusion optimization aggregates Operations logically, enabling multiple Operations to execute cooperatively within the same kernel. Intermediate computation results are kept in on-chip high-speed caches for direct reading by downstream Operations, eliminating redundant GM reads and writes, significantly improving the compute-to-memory access ratio, and improving overall execution efficiency.

In the PyPTO programming model, developers build computation graphs through Tensors and Tensor Operations. The graph fusion process is automatically completed by optimization passes inside the compiler, without the need for developers to manually write fused Operation code. Graph fusion passes analyze and rewrite the computation graph while ensuring correctness of computation results, partitioning and reorganizing the original computation graph into subgraphs better suited for execution on the target hardware. PyPTO's graph fusion optimization is mainly divided into depth-direction fusion and breadth-direction fusion, targeting different performance bottleneck scenarios.

#### Depth-Direction Graph Fusion

Depth-direction graph fusion is based on producer-consumer relationships in the computation graph. It fuses consecutive Operations along the data dependency path. By eliminating intermediate result write-back operations, this approach directly optimizes the data flow path, allowing operator chains that were previously bandwidth-limited to complete computation within a single kernel.
![](../figures/pypto.set_pass_options_1.png)

The PyPTO framework has already implemented automatic graph fusion in the depth direction. For extreme performance optimization scenarios, you can manually specify a graph fusion plan to assign operations to specific computation tasks, thereby changing the latency of each task and achieving load balancing. This is implemented by configuring the sg_set_scope parameter of the [set_pass_options](../../api/config/pypto-set_pass_options.md) interface.

Fusion targets typically come from dependency relationships between Operations. For example, when the amount of data transferred between two upstream-downstream Operations is large, they should be merged to reduce transfer latency. Or, when multiple Operations become multiple parallel connected branches after tile tiling, these Operations should be merged. Fusion targets can also come from inherent experience with specific types of operators. For example, for an IFA operator tiled along the batch, s2, and g axes, V1 and V2 should generally each be one subgraph task.

Currently, this capability is primarily considered for use in consecutive Vector computation processes. Merging Matmul Operations with Vector Operations into the same graph is not yet supported. When using this feature, analysis and adjustments should be made in combination with swimlane graph information. For specific usage, refer to the case: [glm_attention.py](../../../models/glm_v4_5/glm_attention.py).

#### Breadth-Direction Graph Fusion
Breadth-direction graph fusion targets Operations at the same level in the computation graph that can be executed in parallel. By merging multiple parallel Operations into the same kernel for execution, it increases the computation scale of a single kernel execution. During the intra-core instruction scheduling phase, multi-branch fusion can more fully fill the hardware pipeline, achieving better multi-pipe concurrency. At the memory access level, by consolidating same-source memory accesses, multiple repeated GM-to-L1 transfers are integrated into a single load, saving memory bandwidth while effectively amortizing the kernel launch overhead, ultimately improving the overall throughput of the hardware compute units.
![](../figures/pypto.set_pass_options_2.png)

For Matmul and Vector computation, PyPTO provides different breadth-direction graph fusion optimization interfaces, which are introduced in detail in the following sections.

**Matmul Breadth-Direction Graph Fusion**

In the matmul computation scenario, subgraph merging is configured through the [set_pass_options](../../api/config/pypto-set_pass_options.md) interface. The main strategies available are `L1Reuse` and `CubeNBuffer`. Both are used to merge Cube subgraphs in the breadth direction. L1Reuse merges subgraphs with redundant L1 transfers, while CubeNBuffer merges isomorphic subgraphs. L1Reuse can reduce L1 transfer volume, and CubeNBuffer can hide transfer and computation latency between matmul operations on different branches. Both can reduce subgraph scheduling overhead. However, given that data transfer is the performance bottleneck in most matmul scenarios, L1Reuse strategy should be prioritized to reduce data transfer volume.

In practice, the L1Reuse strategy is enabled by default and automatically computes and configures the number of subgraph merges. For extreme performance optimization scenarios, you can manually configure it with the `cube_l1_reuse_setting` parameter. Typically consider values of 2, 4, 8, etc., and configure the optimal value based on actual swimlane graph test data. Reference configuration:

```python
# Globally set to 2
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)

# On top of global auto-configuration, set isomorphic subgraph id=0 to 8
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {0: 8}}
)

# On top of global setting of 2, set isomorphic subgraph id=0 to 8
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}
)
```

CubeNBuffer is for scenarios where L1Reuse cannot be enabled. Such scenarios are less common; the main cases are:
1. There are no repeated L1 transfers between Cube subgraphs. For example, when the shapes of the left and right matrices of a BatchMatmul are (128, 64, 64) and (128, 64, 64) respectively, after the pass produces 128 isomorphic Cube subgraphs with left and right matrix shapes of (64, 64) and (64, 64), there are no repeated L1 transfers between them. Another example is the MM2 in an FA operator — there are no repeated L1 transfers between MM2 subgraphs for different S2 blocks.
2. The K axis is very long. Without K-axis splitting, L1Reuse requires that an entire row of the left matrix or an entire column of the right matrix remain resident in L1. L1 cache capacity is limited. Therefore, when the K axis is long and K has not been split, L1Reuse cannot be used.

In such cases, the `cube_nbuffer_setting` parameter can be configured, with further optimization based on actual swimlane graph test data. Refer to the case [mla_prolog_quant_impl.py](../../../models/deepseek_v32_exp/mla_prolog_quant_impl.py).

**Vector Breadth-Direction Graph Fusion**

In the vector computation scenario, breadth-direction graph fusion is configured through the `vec_nbuffer_setting` parameter of the [set_pass_options](../../api/config/pypto-set_pass_options.md) interface. Note that the preceding optimization steps — adjusting the tiling and merging of upstream and downstream subgraphs to a reasonably good state — should be completed first before using vecNBuffer for breadth-direction merging. When there are groups of isomorphic subgraphs in the swimlane graph with a large number of small subgraphs (duration below 10 us), this feature should be used to reduce scheduling overhead and kernel launch overhead.

The configuration method for the `vec_nbuffer_setting` parameter is similar to `cube_nbuffer_setting`. Refer to the case [sparse_flash_attention_quant_impl.py](../../../models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py).

### Scheduling Strategy Optimization

The inter-core pipeline of a PyPTO operator is determined by the AICPU's scheduling of subgraphs, based on inter-subgraph dependency relationships and inter-core task scheduling strategies. You can try changing the scheduling strategy to achieve better operator performance. When dependencies between upstream and downstream subgraphs are relatively simple, or when the L2 cache hit rate of input tensors for downstream subgraphs is important, L2-affinity scheduling is recommended. The configuration is as follows:

```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

When configuring, comprehensively consider the effects of L2 reuse and load balancing. The optimal configuration strategy differs for different scenarios; specific analysis should be performed in conjunction with swimlane graphs.

### Other Optimization Methods

- For special shapes, try using Vector operations to pre-process the input matrix to convert it to a more standard shape. Taking a matmul with left and right matrix shapes of (884736, 16) and (16, 16) respectively as an example: if only L1Reuse is used, optimization can only reach 500 us. Using the following approach — pre-constructing a new right matrix c of shape (64, 64) by placing four identical right matrices on the diagonal, then performing matmul with the reshaped left matrix — the operator performance improves dramatically to 40 us.

```python
def matmul_kernel(a, b, out):
    # Construct c
    pypto.set_vec_tile_shapes(64, 64)
    d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
    c1 = pypto.concat([b, d, d, d], 1)
    c2 = pypto.concat([d, b, d, d], 1)
    c3 = pypto.concat([d, d, b, d], 1)
    c4 = pypto.concat([d, d, d, b], 1)
    c = pypto.concat([c1, c2, c3, c4], 0)
    # Reshape a
    a = pypto.reshape(a, [221184, 64])
    # Matmul
    pypto.set_pass_options(Cube_l1_reuse_setting={-1:9})
    pypto.set_cube_tile_shapes([512, 512], [64, 64], [64, 64], True)
    e = pypto.matmul(a, c, pypto.DT_BF16)
    e = pypto.reshape(e, [884736, 16])
    pypto.assemble(e, [0, 0], out)
```

- Add redundant computation to avoid redundant dependencies and transfers. The following is an example from [glm_moe_fusion.py](../../../models/glm_v4_5/glm_moe_fusion.py) in the PyPTO repository. By duplicating e_score_bias_2d tile_batch times before performing the cast operation, each copy's cast is fused with the other operations for the corresponding batch. This avoids a one-to-many subgraph dependency between the e_score_bias_2d cast operation and the subsequent computations for each batch, reducing scheduling overhead, and also avoids transfers of the cast result.
```
e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
for tmp_idx in range(tile_batch):
       pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)
```

- Try to avoid processing tensors with small tail-axis lengths. When a large TileShape cannot prevent the tail axis of an Operation's input tensor from being small, use data operation Operations such as concat, transpose, or reshape to increase the tail axis size.

- Use [set_cache_policy](../../api/tensor/pypto-Tensor-set_cache_policy.md) to set a reasonable L2 CacheMode. For global memory data that is accessed only once, set its access state to not enter the L2 cache.

- When operator performance is still poor after the above optimizations, consider whether the TileOperation implementation itself is suboptimal. Construct a test case for a single Operation and compare its performance with the Ascend C small operator performance. After confirming poor performance, check whether a better instruction is not being used.
