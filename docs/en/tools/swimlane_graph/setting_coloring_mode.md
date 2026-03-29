# Setting the Coloring Mode

## Feature Description

Supports setting the coloring mode for the swimlane graph — for example, differentiating colors in the swimlane graph by subgraph or node type — to help users more easily identify information in the graph.

## Prerequisites

Execute the PyPTO program to generate the swimlane graph file, and ensure that a `dyn_topo.txt` file exists in the same directory.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    1.  In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    2.  In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  In the swimlane graph interface, click the ![](../../../tools/figures/zh-cn_image_0000002489256878.png) button on the right side.

    ![](../../../tools/figures/zh-cn_image_0000002502052124.png)

3.  Configure the layer options according to the interface.

    ![](../../../tools/figures/zh-cn_image_0000002533812261.png)

    **Table 1**  Configuration descriptions

    | Parameter | Description |
    |--|--|
    | Coloring mode | The coloring mode for nodes in the swimlane graph. Options include:<br>**By subgraph**: Different colors are displayed for different subgraphs. This is the default display mode.<br>**By stitched function**: Different colors are displayed for different stitched functions.<br>**By semantic label**: Different colors are displayed for different computation types. |
    | Task content display | Configures the name displayed on swimlane graph nodes. Options include:<br>**Task name**: Displays the task name, for example: `0-10-12-0-0`.<br>**Semantic label**: Displays the computation type, for example: `Matmul`.<br>If both are selected, both are displayed, for example: `0-10-12-0-0 (Matmul)`. |
    | Task connection level | Configures the connection level displayed for swimlane graph nodes. For example: configuring `-1` shows all levels of the node; configuring `1` shows the immediate predecessor and successor dependencies. |


4.  The configuration takes effect as displayed.

    For example, when **"Coloring mode"** is set to **"By semantic label"** and **"Task content display"** is set to **"Semantic label"**, the swimlane graph interface is displayed as follows:

    ![](../../../tools/figures/zh-cn_image_0000002502052876.png)
