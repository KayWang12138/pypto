# Viewing the Computation Graph

## Feature Description

The computation graph describes the structure of a PyPTO program's computation process. It is composed of multiple Tensor nodes (representing data) and Operation nodes (representing data operations), and expresses data flow and computation logic in the form of a directed acyclic graph (DAG). The computation graph in PyPTO differs from the concept of a GE (Graph Engine) computation graph. PyPTO's computation graph includes the Tensor Graph, Tile Graph, Block Graph, and Execute Graph, which characterize the complete compilation pipeline of a PyPTO program from abstract computation description to hardware execution. This section describes how to view the computation graph and the key information displayed within it.

## Prerequisites

Execute the PyPTO program to generate the computation graph files.

1.  Enable the graph compilation debug mode:

    ```python
    @pypto.frontend.jit(
        debug_options={"compile_debug_mode": 1}
    )
    ```

2.  Re-execute the PyPTO program, for example:

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3.  After execution, computation graph files in `.json` format for each compilation stage are generated in the `${work_path}/output/output_*/` directory (where `*` represents a timestamp).

    ```text
    ├── Pass_xx_xx
    │   ├── After_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json # computation graph file after pass optimization
    │   ├── After_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.tifwkgr # users do not need to pay attention to this file
    │   ├── Before_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json # computation graph file before pass optimization
    │   ├── Before_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.tifwkgr # users do not need to pay attention to this file
    │   └── ExpandFunctionTENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.log
    ├── program.json # records static information such as function name and semantic label
    ├── ...
    ```

The following selects the last computation graph file at each compilation stage to help users understand the key information of each type of computation graph.

-   Tensor Graph: `Before_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json`
-   Tile Graph: `Before_027_SubgraphToFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json`
-   Block Graph: `After_037_CodegenPreproc_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_LEAF_program_id_00_15536366383870408930.json`
-   Execute Graph: `After_037_CodegenPreproc_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_ROOT.json`

## Viewing the Tensor Graph

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Open the file `Before_004_ExpandFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json`. The following interface is displayed.

    ![](../../../tools/figures/zh-cn_image_0000002533278571.png)

    The computation graph type can be seen as Tensor Graph from the title area at the top right. The Tensor Graph consists of the following parts:

    -   **Incast/Outcast nodes**: Data source/result nodes, corresponding to label ① in the figure. For parameter descriptions, see [Table 1](#table1).
    -   **Tensor nodes**: Data nodes, corresponding to label ② in the figure. For parameter descriptions, see [Table 2](#table2).
    -   **Operation nodes**: Operation nodes, corresponding to label ③ in the figure. For parameter descriptions, see [Table 3](#table3).

    As shown in the figure, the Tensor shapes are consistent with the code definition and have not yet undergone Tile expansion.

    **Table 1**  Incast/Outcast node parameter descriptions<a id="table1"></a>

    | Parameter | Description |
    |--|--|
    | Node name | The name of the Incast/Outcast node in the computation graph, shown in the card title. For example, `TENSOR_1` in the figure above. |
    | Magic ID | The unique identifier of the node, shown as a number in the upper-right corner of the card. For example, `7` in the figure above. |
    | rawTensor_MagicID | The logical memory block number to which the node belongs. One `rawTensor` memory block can be allocated to multiple Tensors. Shown in parentheses after the card title. For example, `8` in the figure above. |
    | Slot ID | The Slot number to which the node belongs. If two nodes have the same `rawTensor_MagicID` and Slot ID, these two nodes share the same memory address. |
    | shape | The shape information of the node, as an integer array. |
    | rawshape | The shape information of the rawTensor. |
    | offset | The offset of the current Tensor within rawTensor memory, as an integer array. |
    | asis | The memory level at which the Tensor currently resides — i.e., the memory level after processing by the preceding Operation node. This is consistent with the `to offset` field of the preceding Operation node. Example values (consistent with PyPTO's MemoryType; not exhaustive):<br>**MEM_UB**: Unified Buffer, used for temporary data storage and computation.<br>**MEM_L1**: L1 cache, a high-speed cache for frequently accessed data.<br>**MEM_L0A**: L0A cache, used for caching matrix A data.<br>**MEM_L0B**: L0B cache, used for caching matrix B data.<br>**MEM_L0C**: L0C cache, used for caching matrix C data.<br>**MEM_DEVICE_DDR**: Device DDR memory, the main storage area with large capacity but slower access.<br>**MEM_UNKNOWN**: Memory data not output. |
    | tobe | The destination memory level — i.e., the memory level at which the Tensor will be operated on in the next Operation. Consistent with the `from` field of the next Operation node. |
    | datatype | The data type. |

    **Table 2**  Tensor node parameter descriptions<a id="table2"></a>

    | Parameter | Description |
    |--|--|
    | Node name | The name of the Tensor node in the computation graph, shown in the card title. For example, `INCAST_LOCAL_BUF0` in the figure above. |
    | Magic ID | The unique identifier of the node, shown as a number in the upper-right corner of the card. For example, `8` in the figure above. |
    | rawTensor_MagicID | The logical memory block number to which the node belongs. One `rawTensor` memory block can be allocated to multiple Tensors. Shown in parentheses after the card title. For example, `9` in the figure above. |
    | Subgraph ID | The ID of the subgraph to which the node belongs. The value is `-1` when the Tensor has not yet been assigned to a specific subgraph (before graph partitioning, i.e., before the GraphPartition Pass). |
    | shape | The shape information of the Tensor node, as an integer array. |
    | rawshape | The shape information of the rawTensor. |
    | offset | The offset of the current Tensor within rawTensor memory, as an integer array. |
    | asis | The memory level at which the Tensor currently resides — i.e., the memory level after processing by the preceding Operation node. This is consistent with the `to offset` field of the preceding Operation node. Example values (consistent with PyPTO's MemoryType; not exhaustive): MEM_UB: Unified Buffer, for temporary data storage and computation. MEM_L1: L1 cache, high-speed cache for frequently accessed data. MEM_L0A: L0A cache, for caching matrix A data. MEM_L0B: L0B cache, for caching matrix B data. MEM_L0C: L0C cache, for caching matrix C data. MEM_DEVICE_DDR: Device DDR memory, the main storage area with large capacity but slower access. |
    | tobe | The destination memory level — i.e., the memory level at which the Tensor will be operated on in the next Operation. Consistent with the `from` field of the next Operation node. |
    | datatype | The data type. |

    **Table 3**  Operation node parameter descriptions<a id="table3"></a>

    | Parameter | Description |
    |--|--|
    | Node name | The name of the Operation node in the computation graph. For example, `TILE_VIEW` in the figure above. |
    | Magic ID | The Magic ID of the current Operation. Operation Magic IDs are unique within a single graph. For example, `10007` in the figure above. |
    | Subgraph ID | The ID of the subgraph to which the current node belongs. The value is `-1` when the Operation has not yet been assigned to a specific subgraph (before graph partitioning, i.e., before the GraphPartition Pass). |
    | shape | The shape information of the Tensor being processed, as an integer array. |
    | from | The memory level of the source operand data of the Operation — i.e., the memory level before this operation is applied. Consistent with the `tobe` field of the preceding Tensor node. |
    | to offset | The memory offset of the destination operand data relative to the source operand data of the Operation. |


## Viewing the Tile Graph

Open the file `Before_027_SubgraphToFunction_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.json`. The following interface is displayed.

![](../../../tools/figures/zh-cn_image_0000002533531967.png)

The computation graph type can be seen as Tile Graph from the title area at the top right. The Tile Graph consists of the following parts:

-   **Incast/Outcast nodes**: Data source/result nodes, corresponding to label ① in the figure. For parameter descriptions, see [Table 1](#table1).
-   **Tensor nodes**: Data nodes, corresponding to label ② in the figure. For parameter descriptions, see [Table 2](#table2).
-   **Operation nodes**: Operation nodes, corresponding to labels ③ and ④ in the figure. The difference between the two is that Operations of type ④ are not related to the shape of the input/output Tensors, and therefore have no `shape` attribute. For detailed parameter descriptions, see [Table 3](#table3).

As shown in the figure, compared to before Tile expansion, the Tile Graph contains many more nodes. This is because the original Tensor with shape `(-1, 32, 1, 256)` has been Tile-expanded and split into Tiles with shape `(1, 4, 1, 64)`. At the same time, memory levels are assigned to each Tile (corresponding to `asis` for the source address and `tobe` for the destination address), and the Operation nodes connected before and after each Tensor are split and processed (for example, `TILE_COPY_IN` and `TILE_COPY_OUT` in the figure).

## Viewing the Block Graph

Open the file `After_037_CodegenPreproc_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_LEAF_program_id_00_15536366383870408930.json`. The following interface is displayed.

![](../../../tools/figures/zh-cn_image_0000002501792452.png)

The computation graph type can be seen as Block Graph from the title area at the top right. The Block Graph consists of the following parts:

-   **Incast/Outcast nodes**: Data source/result nodes, corresponding to label ① in the figure. For parameter descriptions, see [Table 1](#table1).
-   **Tensor nodes**: Corresponding to label ② in the figure. For parameter descriptions, see [Table 2](#table2).
-   **Operation nodes**: Corresponding to labels ③ and ④ in the figure. The difference between the two is that Operations of type ④ are not related to the shape of the input/output Tensors, and therefore have no `shape` attribute. For detailed parameter descriptions, see [Table 3](#table3).

As shown in the figure, at the Block Graph stage, the Tile Graph is split into several subgraphs, with each subgraph corresponding to one Block Graph. Therefore, the scale of the Block Graph is significantly smaller than that of the Tile Graph.

In this sample, the program is split into multiple structurally identical subgraphs (referred to as isomorphic subgraphs). As a result, there is only one JSON file in the `Pass_36_CodegenPreproc` directory whose filename contains the keyword `After_036_CodegenPreproc_*_**LEAF**_*`.

## Viewing the Execute Graph

Open the file `After_037_CodegenPreproc_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_ROOT.json`. The following interface is displayed.

![](../../../tools/figures/zh-cn_image_0000002501621864.png)

The computation graph type can be seen as Execute Graph from the title area at the top right. The Execute Graph consists of the following parts:

-   **Incast/Outcast nodes**: Data source/result nodes, corresponding to label ① in the figure. For parameter descriptions, see [Table 1](#table1).
-   **Call nodes**: Marked with the `fx` identifier, representing one invocation of a Block Graph, corresponding to label ② in the figure. For parameter descriptions, see [Table 4](#table4). Double-clicking a call node allows you to view the corresponding Block Graph subgraph information and understand the specific execution process.

    **Table 4**  Call node descriptions<a id="table4"></a>

    | Parameter | Description |
    |--|--|
    | Call node name | Call nodes are only displayed in the Execute Graph. They have the `fx` identifier in the lower-right corner of the node card. Node names carry the `CALL` prefix, serving as the entry node of a Block Graph subgraph that can be scheduled to run on an AI Core. Double-clicking a call node navigates to the Block Graph subgraph. |
    | Subgraph ID | The ID of the subgraph in which the node resides. |
    | InCast | The list of Magic IDs of the input Tensors. |
    | OutCast | The list of Magic IDs of the output Tensors. |


## Other Operations

The computation graph page supports drag-to-pan, minimap viewing, as well as zooming, and clicking nodes to view their details.
