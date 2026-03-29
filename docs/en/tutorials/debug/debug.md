# Functional Debug

## Compilation and Execution Workflow

After developers complete the definition of Tensors and build a complete computation flow through a series of basic Tensor operations, the system generates a graph structure consisting of interleaved Tensor and Operation nodes — the computation graph. This computation graph goes through the PyPTO compilation optimization pipeline to complete the compilation process from the original computation graph to an executable graph, ultimately generating executable code that runs on Ascend hardware to perform actual computation tasks.

### Graph Compilation Workflow

The following figure illustrates the complete computation graph compilation process. The Tensor Graph, Tile Graph, and Block Graph stages undergo optimization through multiple passes. The Execute Graph stage then integrates graph information and generates the final hardware execution graph.

For the specific list of passes, refer to the file: framework/src/passes/pass\_mgr/pass\_manager.cpp.

**Figure 1**  Computation Graph Compilation Workflow
![](../figures/computation_graph_compilation_process.png "Computation Graph Compilation Workflow")

The graphs generated at each compilation stage are the Tensor Graph, Tile Graph, Block Graph, and Execute Graph. These graphs are key artifacts of the compilation process, representing the complete compilation pipeline from abstract computation description to hardware execution for a PyPTO program.

-   **Tensor Graph**: The Tensor Graph consists of Tensor and Operation nodes and describes the computation flow defined by the user. This graph does not involve tile expansion or memory hierarchy semantics; it serves only as a high-level representation of the computation logic. Optimizations based on the Tensor Graph focus mainly on hardware-independent general graph optimization techniques, such as redundant node elimination and constant folding.
-   **Tile Graph**: The Tile Graph consists of Tiles and TileOps. The Tensor Graph is expanded according to TileShape, decomposing Tensors into Tiles and Operations into TileOps. The Tile Graph automatically infers the storage location of Tiles based on TileOp information and the memory hierarchy of the target hardware, and inserts memory transfer nodes where necessary to ensure correct data transfer between different memory levels.
-   **Block Graph**: The Block Graph partitions the Tile Graph into multiple subgraphs, each of which can be scheduled to run on a single AI Core. The Block Graph is used for hardware-related optimizations, including instruction scheduling, on-chip memory allocation, and synchronization operation insertion, thereby improving hardware execution efficiency.
-   **Execute Graph**: The Execute Graph is the final product of the compilation pipeline. It integrates all optimization results and precisely describes the dependencies between the Block Graphs, used for scheduling and execution by the device scheduler.

Using the PyPTO Toolkit visualization tool, developers can intuitively inspect the computation graph structure and understand the node information in the computation graph, helping them debug operator functionality more conveniently.

### Graph Execution Workflow

The following figure illustrates the complete execution process from resource preparation and task submission to computation.

-   Resource preparation phase: Based on the execution resource information described in the Execute Graph, request global resources such as workspace memory and Streams from the execution hardware.
-   Task parameter assembly and submission phase: PyPTO hardware execution tasks are divided into AI CPU tasks and AI Core tasks. After completing the parameter assembly and task configuration required for these two types of tasks, they are submitted to the RTS for task dispatch.
-   Task execution phase: Through the close cooperation between AI CPU and AI Core in a Client-Server-like architecture, the entire computation task is completed. The AI CPU completes parsing and distribution of subtasks based on the Execute Graph, while the AI Core is responsible for receiving subtasks dispatched by the AI CPU and executing them.

Before execution, you can enable the collection and output of swimlane graph data. Using the PyPTO Toolkit visualization tool, you can intuitively view the inter-core parallel relationships and execution order of each subtask on AIC/AIV, helping developers understand the overall pipeline distribution and perform targeted operator performance optimization.

**Figure 2**  Computation Graph Execution Workflow
![](../figures/computation_graph_execution_process.png)

The following figure shows the relationship between AI CPU and AI Core during hardware runtime of PyPTO tasks, along with the detailed execution flow. The main process can be summarized as: HostMachine initializes resources \> DeviceMachine generates DeviceTasks via Stitch and schedules CallTasks \> CoreMachine executes CallTasks \> DeviceProgram coordinates the entire process.

-   HostMachine: Runs on the Host side and is responsible for actual hardware task execution, including resource preparation and task assembly.
-   DeviceMachine: Runs on the AI CPU side. Based on execution-state data such as the Execute Graph, it is responsible for distributing and scheduling AI Core execution subtasks. The specific process is: Control-AICPU uses Stitch (literally meaning "stitch together") to integrate CallTasks from multiple loop-independent iterations into a single DeviceTask, breaking loop boundaries and maximizing CallTask parallelism; Schedule-AICPU manages AI Core-CallTask dispatch and management based on DeviceTasks. Each DeviceTask is shared among 3 Schedule-AICPUs, each of which extracts ready CallTasks from the DeviceTask for dispatch based on the idle state of the AIC/AIV cores it manages.
-   CoreMachine: Runs on the AI Core side and is responsible for receiving and executing CallTasks dispatched by the AI CPU. A CallTask is the smallest execution unit on AIC/AIV. It consists of a series of CCE instructions and is used to execute specific transfer and computation tasks on the AI Core.
-   DeviceProgram: DeviceProgram is the core data for each PyPTO operator running on the Device side. It is generated from the information described in the Execute Graph combined with hardware resource management.

**Figure 3**  Execution State Runtime Diagram
![](../figures/execution_state_runtime_diagram.png)

## NPU On-Board Debug

If errors occur or results do not meet expectations during the graph compilation or graph execution process, you can enable debug mode to generate computation graph files at different stages. The computation graph describes the structure of the PyPTO program's computation flow. It consists of multiple computation nodes and data nodes, represented as a directed acyclic graph (DAG) that expresses data flow and computation logic, representing the complete compilation pipeline from abstract computation description to hardware execution for a PyPTO program. This section introduces how to collect and view the computation graph and highlights key information in the graph.

### Enabling Debug Mode

1.  Enable the graph compilation debug mode switch.

    ```python
    @pypto.frontend.jit(
        debug_options={"compile_debug_mode": 1}
    )
    ```

2.  Run the test case.

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3.  Upon successful execution, computation graph files at different stages (in .json format) are generated in the $\{work\_path\}/output/output\_\*/ directory (\* represents a timestamp).

    ```txt
    ├── Pass_xx_xx
    │   ├── After_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.json # computation graph file after pass optimization
    │   ├── After_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.tifwkgr # not required for user attention at this time
    │   ├── Before_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.json # computation graph file before pass optimization
    │   ├── Before_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.tifwkgr # not required for user attention at this time
    │   └── ExpandFunctionTENSOR_s0_Unroll1_PATH0_4.log
    ├── program.json # records static information such as function name and semantic label
    ├── ...
    ```

### Viewing the Computation Graph

The following selects the last computation graph from each compilation stage and uses the PyPTO Toolkit visualization tool to help users understand key information in each type of computation graph and assist developers in problem localization.

-   `Tensor Graph`: Before\_004\_ExpandFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json
-   `Tile Graph`: Before\_026\_SubgraphToFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json
-   `Block Graph`: After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_LEAF\_program\_id\_00\_15536366383870408930.json
-   `Execute Graph`: After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_ROOT.json

1.  View the Tensor Graph using PyPTO Toolkit.

    Right-click the Before\_004\_ExpandFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json file and select "Open with PyPTO Toolkit" from the pop-up menu.

    ![](../figures/zh-cn_image_0000002499728650.png)

    The graph type shown in the upper-right corner is Tensor Graph. The Tensor Graph consists of Tensors and Operations. The Tensor Shapes in the graph are consistent with the code definition and have not undergone tile expansion.

2.  View the Tile Graph using PyPTO Toolkit.

    Right-click the Before\_026\_SubgraphToFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json file and select "Open with PyPTO Toolkit" from the pop-up menu.

    ![](../figures/zh-cn_image_0000002499888764.png)

    The graph type shown in the upper-right corner is Tile Graph. Compared to before tile expansion, the Tile Graph contains many more nodes. This is because the Tensor with original Shape \(-1, 32, 1, 256\) has been expanded through tiling, partitioned into Tiles with Shape \(1, 4, 1, 64\). At the same time, memory levels are assigned to Tiles (asis = source address, tobe = destination address in the graph), and memory transfer nodes (TILE\_COPY\_IN and TILE\_COPY\_OUT in the graph) are automatically inserted.

3.  View the Block Graph using PyPTO Toolkit.

    Right-click the After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_LEAF\_program\_id\_00\_15536366383870408930.json file and select "Open with PyPTO Toolkit" from the pop-up menu.

    ![](../figures/zh-cn_image_0000002531608703.png)

    The graph type shown in the upper-right corner is Block Graph. In the Block Graph stage, the Tile Graph is partitioned into several subgraphs, each corresponding to one Block Graph. Therefore, the Block Graph is much smaller in scale compared to the Tile Graph.

    The current sample is partitioned into multiple structurally identical subgraphs (referred to as isomorphic subgraphs). Therefore, the Pass\_36\_CodegenPreproc directory contains only one JSON file whose name includes the After\_036\_CodegenPreproc\_\*\_**LEAF**\_\* keyword.

4.  View the Execute Graph using PyPTO Toolkit.

    Right-click the After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_ROOT.json file and select "Open with PyPTO Toolkit" from the pop-up menu.

    ![](../figures/zh-cn_image_0000002499728842.png)

    The graph type shown in the upper-right corner is Execute Graph. The Execute Graph contains Tensor nodes and call nodes (marked with fx, indicating one invocation of a Block Graph). Double-click a call node to view the corresponding Block Graph subgraph information and understand the specific execution process.

## CPU Simulation Debug

When an Ascend device is not available, testing is also supported in a CPU simulation environment:
- Performance simulation: Allows users to view the intra-core pipeline data of an operator.
- Precision simulation: Allows users to obtain operator computation results in a CPU environment (precision simulation depends on the CANN software package).

If you only need to run simulation and the current environment does not have an Ascend device, do not install torch\_npu, otherwise the execution may fail.

The run mode selection logic is:

-   Manually specify simulation mode:

    Explicitly call `@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})` in the operator code to forcibly enable CPU simulation mode for executing the operator program.
    PyPTO first runs performance simulation; after performance simulation completes, if the CANN software package is detected, it continues with precision simulation.

-   Auto-detection mode (supports performance simulation only, not precision simulation):
    -   CANN software package not detected: automatically enables simulation mode (no explicit configuration required).
    -   CANN software package detected: real hardware execution is prioritized; simulation mode does not take effect.

The specific operation steps are:

1.  Specify the run parameter run\_mode.

    ```python
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    ```

2.  Run the operator to automatically trigger simulation execution.

    ```bash
    python examples/00_hello_world/hello_world.py --run_mode=sim
    ```

3.  After performance simulation completes successfully, the following files are generated in the output directory.

    ![](../figures/zh-cn_image_0000002527468273.png)

4.  Right-click the merged\_swimlane.json file and select "Open with PyPTO Toolkit" from the pop-up menu.

    ![](../figures/zh-cn_image_0000002495188648.png)

    The swimlane graph shows the debug status of tasks within each core, including execution time and idle intervals. You can use this to optimize the operator based on specific conditions, such as adjusting the tile shapes of tensors.

5. Precision simulation is consistent with NPU execution. After completion, the runtime result is returned, which the user can obtain and process.
