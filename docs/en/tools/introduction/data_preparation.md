# Data Preparation

## Feature Description

Before using PyPTO Toolkit, you need to prepare the relevant result files from the PyPTO program to support subsequent data visualization and analysis using PyPTO Toolkit.

## Prerequisites

PyPTO Toolkit analyzes the result files produced by a PyPTO program. It is therefore recommended that you run the PyPTO program within Visual Studio Code, or connect to the PyPTO program's runtime environment using the Visual Studio Code Remote - SSH extension (for container-based runtimes, connect to the interior of the container). This way, you can follow the steps below to view the relevant result files.

Alternatively, you can directly import the result folder into the Visual Studio Code workspace. However, this may cause some features to be unavailable — for example, the [control flow graph](../control_flow/viewing_control_flow_graph.md) display — though it will not affect the use of the [computation graph](../computation_graph/viewing_computation_graph.md), [swimlane graph](../swimlane_graph/viewing_swimlane_graph.md), or [three-column linked view](../three_column/three_column_linked_view.md) features.

## Steps

1.  Execute the PyPTO program.
2.  In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002489868856.png) icon to open PyPTO Toolkit.
3.  In the run results interface, the generated result files can be viewed.

    ![](../../../tools/figures/zh-cn_image_0000002523389705.png)

    Result folders are named using the format "execution method + execution date + execution time", for example: `npu_20251021_203500`. They mainly contain the following types of files:

    -   **Results overview**: Displays the control flow graph by default. Through the control flow graph, you can clearly understand the relationships between Loop–Loop, Loop–Path, Loop–If, If–If, and If–Path. For detailed information, refer to [Control Flow Graph](../control_flow/viewing_control_flow_graph.md).
    -   **Swimlane graph**: Intuitively displays the actual scheduling and execution process of the computation graph, clearly presenting the task execution order and timing information to help developers analyze operator performance bottlenecks. For detailed information, refer to [Swimlane Graph](../swimlane_graph/viewing_swimlane_graph.md).
    -   **Computation graph**: Describes the computation process structure of the PyPTO program. It consists of multiple computation nodes and data nodes, represented as a directed acyclic graph (DAG) showing data flow and computation logic. For detailed information, refer to [Computation Graph](../computation_graph/viewing_computation_graph.md).
