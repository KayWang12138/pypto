# Searching Swimlane Graph Nodes

## Feature Description

Supports searching by swimlane graph node name, making it easy for users to find specific Task details and determine which swimlane the task belongs to.

## Prerequisites

Execute the PyPTO program to generate both the computation graph file and the swimlane graph file.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  In the search box above the swimlane graph, enter a node name (for example: `0-4-1`) and press Enter.

    ![](../../../tools/figures/zh-cn_image_0000002533808475.png)

    Search results support fuzzy matching and are sorted by time.

3.  Click the **left/right** arrows on the right side of the search bar to navigate to the "previous/next" node.

    ![](../../../tools/figures/zh-cn_image_0000002533808883.png)

4.  Click the **"A/a"** case-sensitivity button to enable case-sensitive search.

    ![](../../../tools/figures/zh-cn_image_0000002533809275.png)
