# Viewing the Health Report

## Feature Description

The health report displays statistical information for different stages of the computation graph.

-   The **Tensor Graph health report** displays the original statistical information of the user-defined computation graph.
-   The **Tile Graph health report** displays overall statistical information of the computation graph after TileShape expansion. This includes the total number of nodes, the bottleneck nodes with the highest in-degree and out-degree, the maximum concurrency, total data transfer volume, and redundant transfers.
-   The **Block Graph health report** displays overall statistical information of the subgraph scheduled to run on a single AI Core. This includes spill data volume, peak and average usage of each memory space, memory fragmentation rate, and other metrics.
-   The **Execute Graph health report** displays overall statistical metrics for task scheduling among the corresponding subgraphs. This includes maximum concurrency, maximum depth, total number of subgraphs, bottleneck subgraphs with the highest in-degree and out-degree, and subgraph isomorphism rate.

## Prerequisites

Execute the PyPTO program to generate the computation graph files and health report files. The steps to generate health report files are as follows.

1.  Configure the health report generation switch: `${install_path}/python/site-packages/pypto/lib/configs/tile_fwk_config.json`.

    ```python
    "global": {
        "pass":{
            "default_pass_configs": {
                "health_check": false  # Default is false; set to true to enable
            }
        }

        },
    ```

2.  Execute the PyPTO program. The relevant health report files are generated in the same directory as the computation graph files under `${work_path}/output/output_*/` (where `*` represents a timestamp). The filename format is `_xxxHealthReport.json`.
    -   Tensor Graph health report files are generated in the **`Pass_xx_ExpandFunction`** directory.
    -   Tile Graph health report files are generated in the **`Pass_xx_L1CopyInReuseMerge`** directory.
    -   Block Graph health report files are generated in the **`Pass_xx_OoOSchedule`** directory.
    -   Execute Graph health report files are generated in the **`Pass_xx_SubgraphToFunction`** directory.

## Viewing the Tensor Graph Health Report

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  In the computation graph interface, click the ![](../../../tools/figures/zh-cn_image_0000002480358584.png) button on the right side. ![](../../../tools/figures/zh-cn_image_0000002480201920.png)

    The health report data panel slides out automatically on the right side of the computation graph interface. The health report varies slightly depending on the type of computation graph; each type is described below.

3.  View the Tensor Graph health report.

    ![](../../../tools/figures/zh-cn_image_0000002486905898.png)

    Region ①: Displays summary information for Operation nodes. For details, see [Table 1](#table1).

    Region ②: Displays summary information for Tensor nodes. For details, see [Table 2](#table2).

    **Table 1**  Tensor Graph health report — Region ① parameter descriptions<a id="table1"></a>

    | Parameter | Description |
    |--|--|
    | MaxDepth | The total number of nodes on the longest path from an Incast node to an Outcast node in the computation graph. |
    | MaxWidth | The maximum concurrency in the computation graph — i.e., the maximum number of nodes that can execute concurrently at the same time. |
    | max Fanin | The maximum in-degree value among all nodes in the computation graph. In-degree represents the number of inputs received from upstream nodes. |
    | max FaninOps | The node(s) with the highest in-degree in the computation graph. Click the expand button on the right to list all Operation nodes. ![](../../../tools/figures/zh-cn_image_0000002534470071.png) Click the Magic ID value of a node (e.g., `10003`) to navigate to the corresponding position in the graph. |
    | max Fanout | The maximum out-degree value among all nodes in the computation graph. Out-degree represents the maximum number of downstream nodes to which each node must distribute its output. |
    | max FanoutOps | The node(s) with the highest out-degree in the computation graph. |

    **Table 2**  Tensor Graph health report — Region ② parameter descriptions<a id="table2"></a>

    | Parameter | Description |
    |--|--|
    | maxProducerCount | The maximum number of producers for any Tensor in the computation graph. |
    | maxConsumerCount | The maximum number of consumers for any Tensor in the computation graph. |
    | maxProducerTensors | The Tensor(s) with the most producers in the computation graph. Click the expand button on the right to list all Tensor nodes. ![](../../../tools/figures/zh-cn_image_0000002525509105.png) |
    | maxConsumerTensors | The Tensor(s) with the most consumers in the computation graph. |


## Viewing the Tile Graph Health Report

Follow the steps above to open the Tile Graph health report. The interface is displayed as follows:

![](../../../tools/figures/zh-cn_image_0000002534468137.png)

Region ①: Displays summary information for Operation nodes. For details, see [Table 3](#table3).

Region ②: Displays summary information for Tensor nodes. Data display is not currently supported.

**Table 3**  Tile Graph health report — Region ① parameter descriptions<a id="table3"></a>

| Parameter | Description |
|--|--|
| MaxDepth | The total number of nodes on the longest path from an Incast node to an Outcast node in the computation graph. |
| MaxWidth | The maximum concurrency in the computation graph — i.e., the maximum number of nodes that can execute concurrently at the same time. |
| max Fanin | The maximum in-degree value among all nodes in the computation graph. In-degree represents the number of inputs received from upstream nodes. |
| max FaninOps | The node(s) with the highest in-degree in the computation graph. Click the expand button on the right to list all Operation nodes. |
| max Fanout | The maximum out-degree value among all nodes in the computation graph. Out-degree represents the maximum number of downstream nodes to which each node must distribute its output. |
| max FanoutOps | The node(s) with the highest out-degree in the computation graph. |


## Viewing the Block Graph Health Report

Follow the steps above to open the Block Graph health report. The interface is displayed as follows:

![](../../../tools/figures/F3F0CD01-64BD-4DF6-993F-CBE9A7F9F5C9.png)

Region ①: Timing overview. For details, see [Table 4](#table4).

Region ②: Displays peak and average usage of each memory type throughout the entire scheduling process, as well as the utilization of each pipe. For details, see [Table 5](#table5).

Region ③: Displays Spill statistics. For details, see [Table 6](#table6).

Region ④: Summary statistics such as the maximum producers and consumers of subgraphs. For details, see [Table 7](#table7).

**Table 4**  Block Graph health report — Region ① parameter descriptions<a id="table4"></a>

| Parameter | Description |
|--|--|
| Total Cycles | The theoretical execution time of the current subgraph. |
| Theoretical Minimum Cycles | The theoretical optimal execution time of the current subgraph. |

**Table 5**  Block Graph health report — Region ② parameter descriptions<a id="table5"></a>

| Parameter | Description |
|--|--|
| L1 Avg/Peak | Average/peak usage of L1 memory. |
| L0A Avg/Peak | Average/peak usage of L0A memory. |
| L0B Avg/Peak | Average/peak usage of L0B memory. |
| L0C Avg/Peak | Average/peak usage of L0C memory. |
| UB Avg/Peak | Average/peak usage of UB memory. |
| PIPE_FIX | Utilization of PIPE_FIX. |
| PIPE_MTE1 | Utilization of PIPE_MTE1. |
| PIPE_MTE2 | Utilization of PIPE_MTE2. |
| PIPE_MTE3 | Utilization of PIPE_MTE3. |
| PIPE_M | Utilization of PIPE_M. |
| PIPE_S | Utilization of PIPE_S. |
| PIPE_V | Utilization of PIPE_V. |

**Table 6**  Block Graph health report — Region ③ parameter descriptions<a id="table6"></a>

| Parameter | Description |
|--|--|
| Spill Count | Total number of spill-out and spill-in operations. A higher value indicates tighter memory resources and worse graph performance. |
| magic | The Magic ID of the Tensor that was spilled out/in. |
| Type | The Buffer type where the spill-out/in occurred. |
| Buffer Usage / Rate / AllocSize | Currently occupied size of the Buffer / current Buffer occupancy rate / size of the Buffer currently occupied by Alloc. |
| Spill Tensor / Trigger Tensor / Spill CopyOut Tensor | Size of the Tensor that was spilled out/in / size of the Tensor that triggered the current spill-out/in / amount of data copied out to DDR during spill-out/in. |

**Table 7**  Block Graph health report — Region ④ parameter descriptions<a id="table7"></a>

| Parameter | Description |
|--|--|
| Workspace offset | The amount of Workspace occupied by the current subgraph. |
| OP | Total number of Operation nodes. |
| max Producers | The Tensor with the most producers within the subgraph. |
| max Consumers | The Tensor with the most consumers within the subgraph. |
| max Inputs | The Tensor with the most predecessor nodes within the subgraph. |
| max Outputs | The Tensor with the most successor nodes within the subgraph. |


## Viewing the Execute Graph Health Report

Follow the steps above to open the Execute Graph health report. The interface is displayed as follows:

![](../../../tools/figures/ScreenShot_20251225141115.png)

Region ①: Subgraph isomorphism information. For details, see [Table 8](#table8).

Region ②: Timing information. For details, see [Table 9](#table9).

Region ③: Maximum in-degree and out-degree information. For details, see [Table 10](#table10).

Region ④: Peak memory information. For details, see [Table 11](#table11).

Region ⑤: OP type distribution information. For details, see [Table 12](#table12).

**Table 8**  Execute Graph health report — Region ① parameter descriptions<a id="table8"></a>

| Parameter | Description |
|--|--|
| Total Subgraph | The total number of subgraphs — i.e., the total number of all subgraphs obtained after partitioning the entire computation graph. |
| Unique | The number of non-isomorphic subgraphs. |
| Isomorphism Rate | Total Subgraph / Total Subgraph value. A higher isomorphism rate indicates better concurrency for the graph. |
| MaxSubgraphDepth | The total number of nodes on the longest path from an Incast node to an Outcast node in the computation graph. |
| MaxSubgraphWidth | The maximum concurrency in the computation graph — i.e., the maximum number of nodes that can execute concurrently at the same time. |

**Table 9**  Execute Graph health report — Region ② parameter descriptions<a id="table9"></a>

| Parameter | Description |
|--|--|
| maxSubgraphCycle | `value`: The maximum theoretical execution time of a subgraph. `subgraphID`: The subgraph with the highest theoretical execution time. |
| minSubgraphCycle | `value`: The minimum theoretical execution time of a subgraph. `subgraphID`: The subgraph with the lowest theoretical execution time. |
| avgSubgraphCycle | The average theoretical execution time of subgraphs. |

**Table 10**  Execute Graph health report — Region ③ parameter descriptions<a id="table10"></a>

| Parameter | Description |
|--|--|
| maxSubgraphFanin | `value`: The maximum in-degree value of any subgraph. `subgraphID`: The subgraph with the highest in-degree. |
| maxSubgraphFanout | `value`: The maximum out-degree value of any subgraph. `subgraphID`: The subgraph with the highest out-degree. |

**Table 11**  Execute Graph health report — Region ④ parameter descriptions<a id="table11"></a>

| Parameter | Description |
|--|--|
| PeakUsage | `value` column: Peak GM memory usage. `subgraphID` column: The subgraph where the peak GM memory usage occurs. |

**Table 12**  Execute Graph health report — Region ⑤ parameter descriptions<a id="table12"></a>

| Parameter | Description |
|--|--|
| AIV | Number of subgraphs executing on AIV. |
| AIC | Number of subgraphs executing on AIC. |
| AICPU | Number of subgraphs executing on AI CPU. |
| mix | Number of subgraphs executing on both AIV and AIC. |
