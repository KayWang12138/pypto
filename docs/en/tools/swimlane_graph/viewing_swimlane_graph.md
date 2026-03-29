# Viewing the Swimlane Graph

## Feature Description

The swimlane graph intuitively displays the actual scheduling and execution process of the computation graph, clearly presenting the execution order and timing information of tasks, helping developers analyze operator performance bottlenecks. This section describes how to view the swimlane graph and the key information it displays.

## Prerequisites

Execute the PyPTO program to generate the swimlane graph file.

1.  Enable the performance data collection feature.

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2.  Re-execute the PyPTO program, for example:

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

    A swimlane graph data file named `merged_swimlane.json` is generated in the `${work_path}/output/output_*/` directory (where `*` represents a timestamp).

## Steps

1.  Open the swimlane graph file. Two methods are available:
    - In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    - In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  View the swimlane graph information in the displayed interface.

    ![](../../../tools/figures/zh-cn_image_0000002533984451.png)

    Region ①: Toolbar, supporting [searching swimlane graph nodes](searching_nodes.md), zooming the swimlane graph, [setting the coloring mode](setting_coloring_mode.md), [viewing the graph by time range](viewing_by_time_range.md), [viewing the performance report](viewing_performance_report.md), and other features.

    Region ②: Three-segment time axis. The topmost time axis shows the total execution time; the middle time axis shows the execution time occupied by the current task panel; the bottommost time axis shows nanosecond-level granularity.

    Region ③: Sidebar — the AIV and AIC thread areas, covering performance statistics for each AIV and AIC thread at the thread measurement dimension.

    Region ④: Task panel, displaying each task sequence in swimlane form. Each AIV/AIC thread corresponds to one horizontal swimlane. Each colored block in a swimlane represents a specific task executed on it; its length corresponds to the task's execution time, intuitively reflecting the computation load and density. Users can analyze potential performance bottlenecks by observing the idle gaps between adjacent tasks (shown as black areas in the figure, also called "bubbles") and identifying long-running tasks. The panel supports: scrolling the swimlane graph vertically (mouse scroll wheel), moving horizontally (mouse left-click + Ctrl, or A/D shortcut keys), zooming (mouse scroll wheel + Ctrl, or W/S shortcut keys). Click a task node to view its details — see [Step 3](#step3) for details.

    Region ⑤: Performance metrics step chart. See [Step 4](#step4) for details.

3.  Click a task node (Task node) in the swimlane graph to display its detailed information panel.<a id="step3"></a>

    ![](../../../tools/figures/zh-cn_image_0000002502270046.png)

    Region ①: Node information. For details, see [Table 1](#table1).

    Region ②: Node parameter information. For details, see [Table 2](#table2).

    Region ③: Node performance statistics. Display of relevant data is not currently supported.

    Region ④: Node dependency information. For details, see [Table 3](#table3).

    **Table 1**  Region ① parameter descriptions<a id="table1"></a>

    | Parameter | Description |
    |--|--|
    | Name | The node name contains the task name and semantic label — for example, `0-4-0-25-0(MatMul)`. The first three numbers `0-4-0` are the task identifier (unique ID), representing `seqNo`, `task's rootFunction part`, and `task's opIndex part`, respectively. The last two numbers `25-0` identify the content of the task execution, i.e., `rootIndex` and `psglD in this root`. `MatMul` is the semantic label name, displayed when the `pypto.set_semantic_label` interface is used to set a custom semantic label in the PyPTO program code. |
    | Subgraphid | The ID of the isomorphic subgraph to which the node belongs. Different task nodes may belong to the same isomorphic subgraph. |
    | Start | Start time. |
    | Duration | Duration. |
    | Hit rate | Cache hit rate. |

    **Table 2**  Region ② parameter descriptions<a id="table2"></a>

    | Parameter | Description |
    |--|--|
    | duration | Duration, in microseconds (us). |
    | end_time | End time, in microseconds (us). |
    | start_time | Start time, in microseconds (us). |
    | event-hint | `rootHash`: The hash value of the Execute Graph corresponding to this task. `callOpMagic`: The Magic ID of the call node in the Execute Graph. `leafHash`: The hash value of the Block Graph corresponding to this task. These parameters are primarily used for navigating to the computation graph. For details, see [Navigate from Swimlane Graph to Computation Graph](navigate_to_computation_graph.md). |
    | ioperand-hint | `rawmagic`: The Magic ID of the rawTensor of the input Tensor for this task, primarily used for navigating from the swimlane graph to the Dump Tensor. |
    | ooperand-hint | `rawmagic`: The Magic ID of the rawTensor of the output Tensor for this task, primarily used for navigating from the swimlane graph to the Dump Tensor. |
    | seqNo | Indicates which stitched function this corresponds to. |
    | taskId | Indicates which root and which call within the `seqNo`-th stitched function this corresponds to. |

    **Table 3**  Region ④ parameter descriptions<a id="table3"></a>

    | Parameter | Description |
    |--|--|
    | ![](../../../tools/figures/zh-cn_image_0000002521510305.png) | View predecessor dependencies. |
    | ![](../../../tools/figures/zh-cn_image_0000002502354012.png) | View successor dependencies. |
    | ![](../../../tools/figures/zh-cn_image_0000002533954015.png) | View both predecessor and successor dependencies. |
    | Task connection level | Setting `-1` shows all levels of the node; setting `1` shows the immediate predecessor and successor dependencies of the node, and so on. |

    (step4)=
4.  Scroll the swimlane graph vertically using the mouse scroll wheel to see the performance metrics step chart at the bottom of the page.<a id="step4"></a>

    **Table 4**  Swimlane graph performance metrics step chart parameter descriptions

    | Parameter | Description |
    |--|--|
    | Dependence Solving | The speed of resolving dependencies per unit time. |
    | Ideal_Mem_Usage | Ideal memory usage. |
    | ReadyCount_AIC | Number of tasks waiting to execute on the Cube core. |
    | ReadyCount_AIV | Number of tasks waiting to execute on the Vector core. |
    | ReadyCount_Total | Number of tasks waiting to execute on the Cube + Vector cores. |
