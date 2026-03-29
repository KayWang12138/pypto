# GatedDeltaRule Operator Performance Optimization Case
## 1. Task and Goal
Transformer, with its powerful attention mechanism, has demonstrated excellent sequence modeling capabilities and achieved breakthrough progress in the field of large language models. However, the computational complexity of the self-attention module grows quadratically with sequence length, posing enormous compute challenges during training and inference. To address this problem, researchers have proposed linear Transformer alternatives that replace kernelized dot-product attention with traditional softmax attention, reformulating it as a linear RNN with matrix states, significantly reducing compute demands during both training and inference.
The paper "Gated Delta Networks: Improving Mamba2 with Delta Rule" proposes the Gated DeltaNet architecture. By combining gating mechanisms and the Delta update rule, it improves the performance of linear Transformers on long-sequence modeling and information retrieval tasks.
![Illustration](../figures/gdr_1.1.png)
<p align="center">
Figure 1-1 Gated DeltaNet Model Architecture
</p>

## 2. Operator Framework
```py
@pypto.jit
def chunk_gated_delta_rule():
	# parallel loop over b
    for b_idx in pypto.loop(B):
		# parallel loop over n
        for nv_idx in pypto.loop(Nv):
		    # serial loop over s
            for s_idx in pypto.loop(0, S, L):
                # view
				...

                # compute
                # pre-recurrent_loop logic that can execute in parallel
				...
                # recurrent_loop: computation logic must be serial because output state is updated chunk by chunk
				...

                # assemble
				...
```

## 3. Optimization Workflow
### 3.1 PyPTO Config Optimization
PyPTO provides various Config settings to allow users to perform in-depth custom optimization for their specific fused operator implementations, such as adjusting pass graph fusion strategies, runtime task scheduling strategies, and so on. Among these, the settings most relevant to GDN operator performance optimization are `stitch_function_inner_memory`, `stitch_function_outcast_memory`, and `stitch_function_num_initial` in runtime_options. Dynamic stitch is a key technology in PyPTO's MPMD architecture. Through the AICPU dynamically determining the executed control flow based on runtime inputs and dynamically computing dependencies, it stitches and combines tasks for dispatch. The `stitch_function_inner_memory` setting controls the size of the memory pool for intermediate computation results within a root function; a larger value generally leads to higher parallelism within a stitch batch. The `stitch_function_outcast_memory` setting controls the size of the memory pool for intermediate computation results (the outcast of root functions inside the devitask) within a DeviceTask constructed by stitch. The value represents how many loops' worth of computation graphs the workspace allows to be dynamically stitched together for parallel dispatch; a larger value means a larger estimated workspace memory usage. The `stitch_function_num_initial` setting represents the amount of computation tasks in the first DeviceTask submitted to the Schedule AICPU for processing by the CtrlFlow AICPU in the machine's runtime. This value controls the size of the device machine startup overhead, allowing the CtrlFlow AICPU and Schedule AICPU computations to overlap sooner. For generalization across scenarios with small data volumes (efficient memory use) and appropriate startup overhead, the default values of the three settings are 10, 50, and 30 respectively. In the GDN operator computation scenario, these parameters need to be configured as large as possible — 128, 128, and 128 — to maximize the computation parallelism within a single stitch.
The following swimlane graphs are shown for a data scenario of B=2, T=8192, H=4, D=128. As shown in Figures 3-1 and 3-2, they show the swimlane graph with stitch_function_num_initial set to 32 and 128, respectively. A larger value allows tasks to be more fully parallelized within a stitch, yielding better performance, but workspace usage also increases, requiring a trade-off.
![Illustration](../figures/gdr_3.1.png)
<p align="center">
Figure 3-1 Swimlane graph with stitch_function_num_initial set to 32
</p>

![Illustration](../figures/gdr_3.2.png)
<p align="center">
Figure 3-2 Swimlane graph with stitch_function_num_initial set to 128
</p>

### 3.2 Ascend-Affine Tile Shape Optimization
In PyPTO, all computations are based on Tiles (hardware-aware data blocks), fully leveraging the hardware's parallel compute capabilities and memory hierarchy. Tiles can be stored in AI Core private caches (such as UB, L1), significantly improving data access efficiency. For each operation, the tile_shape size can be flexibly set to optimize compute load balancing, memory bandwidth utilization, and maximize hardware resource usage efficiency, fully utilizing UB capacity and aligning with Ascend architecture characteristics. Triton operators developed for GPU typically set the block size to 64, where the Tile block shape participating in computation is [64, 128]. When the dtype is FP32, the data size is 32 KB. Since the Ascend NPU's UB is typically 192 KB, a suitable Tile block size ranges from 16 to 64 KB. After adjusting the chunk size to 128, the data volume increases to 64 KB, which better aligns with the Ascend operator development strategy of transferring larger data blocks in a single operation. This greatly reduces performance overhead from data transfers and fully leverages the hardware's parallel compute capabilities. In addition, due to the chunk-based iterative update algorithm characteristics of the GDN operator, processing data with a larger chunk size at once can reduce the number of serial iterative update cycles, alleviating the performance issue where serial computation logic under PyPTO's current MPMD scheduling strategy is difficult to fully parallelize across all cores.
As shown in Figure 3-3, compared to Figure 3-2, this shows the swimlane graph after optimizing chunk_size from 64 to 128 and TileShape from [64, 128] to [128, 128]. The Tile block data size better aligns with the Ascend NPU hardware storage size, reducing the number of loop iterations, the number of root functions, and the number of tasks. With the same number of tasks bound for stitch dispatch (stitch_function_num_initial=128), the number of stitches is reduced. In essence, this reduces transfer overhead and improves compute efficiency.
![Illustration](../figures/gdr_3.3.png)
<p align="center">
Figure 3-3 Swimlane graph with chunk_size=128 and TileShape=[128, 128]
</p>

### 3.3 PyPTO Loop Style and Graph Fusion/Tiling Expansion Optimization
#### 3.3.1 Converting Dynamic Loop to Static Loop
PyPTO adopts the PTO (Parallel Tensor/Tile Operation) programming paradigm, with a tile-based programming model as its core design philosophy. Through multi-level computation graph representation, it compiles AI models built by users via APIs from high-level Tensor computation graphs step by step into hardware instructions, ultimately generating code that can be efficiently executed on the target platform. The device side automatically schedules execution in MPMD (Multiple Program Multiple Data) fashion. Understanding PyPTO's loop and graph fusion/tiling logic is a deeper understanding of its frontend representation. First, the loop in PyPTO is mainly for handling dynamic shapes. In full-model scenarios, parameters such as batch_size and seq_length are often dynamic, while tile sizes are typically fixed. The PyPTO framework pre-marks dynamic axes and translates them into CCE code using expressions composed of SymbolicScalar during compilation. The Machine layer then resolves the concrete sizes of these expressions, dynamically handling on-board execution under different shapes. This approach greatly reduces the repetitive time for compilation and graph construction, allowing different dynamic shape inputs to share the same frontend representation and graph construction IR. In the GDN operator, the matrix inversion module performs row-by-row iterative computation updates on Tile blocks. In scenarios without tail blocks (tail blocks can also be padded), the number of row-by-row loop iterations is fixed. PyPTO's loop iteration count consumes root functions — i.e., the number of tasks that a single stitch can bind for dispatch. Therefore, in terms of programming paradigm, the inversion module is better suited for using Python's static loop style, which sacrifices some compilation performance but greatly reduces the runtime overhead caused by the large number of repetitive computations of the inversion module when using dynamic loops.
```py
# row_num is a static value
# Original dynamic loop style
for i in pypto.loop(2, row_num, 1):
    # Row-by-row iterative inversion update

# Modified to static loop style
for i in range(2, row_num, 1):
    # Row-by-row iterative inversion update
```

#### 3.3.2 Graph Fusion/Tiling Optimization
PyPTO's computation graph consists of Tensor data nodes and Operation nodes. Through layer-by-layer pass optimization, the Tensor Graph defined by the user's frontend is ultimately converted into an Execution Graph and then translated into executable CCE code. The Execute Graph integrates computation subgraph information, including their dependency relationships and scheduling information. It defines the specific Operation combinations of Tiles on AIC/AIV. Different graph fusion and tiling optimization strategies have a large impact on performance. Passes typically provide generalized optimization strategies to help users obtain reasonably good performance. However, if precise control over the computation flow within Tile blocks is needed, the DFX capabilities of the computation graph should be used to visually inspect whether the result matches expectations. As shown in Figure 3-4, this is the initial out-of-the-box swimlane graph for B=1, T=128, H=1. It is observed that there are many very short-duration fragmented blocks as well as anomalously long-duration execution graphs. After comparing the computation graph's computation flow with the frontend code, it is found that the PyPTO framework itself has certain graph fusion and tiling functional issues — for example, the inversion module was not correctly fused and tiled according to the frontend representation. After discussion with framework developers, the problem was resolved. As shown in Figure 3-5, this is the swimlane graph after manually adjusting the graph fusion and tiling for B=1, T=128, H=1. At this point, the computation logic graph is highly consistent with the frontend representation. For example, the inversion module consists of 8 parallel row-by-row update isomorphic subgraphs of 16*16 Tile blocks.
![Illustration](../figures/gdr_3.4.png)
<p align="center">
Figure 3-4 Initial out-of-the-box swimlane graph for B=1, T=128, H=1
</p>

![Illustration](../figures/gdr_3.5.png)
<p align="center">
Figure 3-5 Swimlane graph after manually adjusting graph fusion and tiling for B=1, T=128, H=1
</p>

#### 3.3.3 unroll_list Optimization
After understanding runtime concepts such as root function and stitch, we can observe from the swimlane graph that the GDN operator's algorithm consists of parallel computation logic combined with serial computation logic. Due to the current stitch strategy and scheduling strategy of the PyPTO framework, the chunk-by-chunk state update logic is difficult to fully utilize all cores when S is large. PyPTO provides the unroll_list capability, which merges the innermost loop graph. When the expansion count is n, the loop step size becomes step*n, and each iteration executes n iterations of the loop body, thereby improving the parallel compute capability of the outer BN. In essence, this also reduces the number of loop iterations, the number of root functions, and the number of tasks. With the same number of tasks bound for stitch dispatch (stitch_function_num_initial=128), the number of stitches is reduced. As shown in Figure 3-6, compared to Figure 3-3, this shows the swimlane graph when setting unroll_list=[16] in Loop S.

```py
# Original
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx"):

# Set unroll_list=[16]
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16]):
```

![Illustration](../figures/gdr_3.6.png)
<p align="center">
Figure 3-6 Swimlane graph with unroll_list=[16]
</p>

### 3.4 Deep Performance Optimization Based on DFX
As shown in Figure 3-8, in the initial out-of-the-box implementation of GDN, the inversion of [128, 128] Tile blocks was implemented by first inverting 8 [16, 16] blocks separately. This provides good parallel compute capability for smaller shape scenarios. However, as shown in Figure 3-9, for larger data volumes (when AIV cores are already fully utilized), the compute efficiency is not high. Using the DFX capabilities provided by the computation graph and swimlane graph, we proposed a tail-axis merging optimization plan: merging the 8 [16, 16] blocks by their tail axis into 2 [16, 64] blocks for computation, to reduce transfer overhead and improve compute efficiency. However, the piercing benefit obtained did not meet expectations. As shown in Figure 3-7, performance only improved from 8*22 us to 2*62 us, which did not meet the expected benefit. After inspecting the computation flow in the computation graph, it was found that there were significant copy-in and copy-out overheads, as shown by the green and red nodes in Figure 3-8.
![Illustration](../figures/gdr_3.7.png)
<p align="center">
Figure 3-7 Swimlane graph of the initial tail-axis merge optimization piercing plan for inversion
</p>

![Illustration](../figures/gdr_3.8.png)
<p align="center">
Figure 3-8 Computation graph of the initial tail-axis merge optimization piercing plan for inversion (green nodes represent copy-in, red nodes represent copy-out)
</p>

By comparing the computation flow in the computation graph with the frontend code, it was found that the cause was the matrix being iteratively updated row by row while being frequently transferred in and out of UB. However, the expectation was that it could remain resident in memory. This enabled further in-depth optimization. As shown in Figure 3-9, the optimized computation graph shows data copy-in only on the leftmost side of the computation flow and data copy-out only on the right side, which matches expectations.
![Illustration](../figures/gdr_3.9.png)
<p align="center">
Figure 3-9 Computation graph of the tail-axis merge optimization piercing plan for inversion
</p>

Finally, the tail-axis merge approach was further optimized to concat into a single [16, 128] block for row-by-row inversion. The resulting final swimlane graph of the tail-axis merge optimization piercing plan for inversion is shown in Figure 3-10. The inversion module latency improved from 8*28 us to 49 us.
![Illustration](../figures/gdr_3.10.png)
<p align="center">
Figure 3-10 Final swimlane graph of the tail-axis merge optimization piercing plan for inversion
</p>
