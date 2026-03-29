# Navigating from the Swimlane Graph to the Computation Graph

## Feature Description

Supports navigating from a Task node in the swimlane graph to the corresponding computation graph, making it easy for users to locate issues.

## Prerequisites

Execute the PyPTO program to generate both the computation graph file and the swimlane graph file.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Click a node in the swimlane graph to open the node details panel. ![](../../../tools/figures/zh-cn_image_0000002533806679.png)
3.  Click **"Bind Computation Graph"** and select the `program.json` file to bind it.

    If `merged_swimlane.json` and `program.json` are in the same directory, they will be bound automatically.

    ![](../../../tools/figures/zh-cn_image_0000002517774919.png)

4.  Click the relevant fields to navigate to the corresponding computation graph.

    ![](../../../tools/figures/zh-cn_image_0000002517780081.png)

    -   Click the `rootHash` value to navigate to the Execute Graph corresponding to the task, making it easy to view the task's predecessor and successor dependencies. You can then use `callOpMagic` to locate the Call Operation node, which corresponds to the task's Block Graph.
    -   Click the `leafHash` value to navigate directly to the Block Graph corresponding to the task.
