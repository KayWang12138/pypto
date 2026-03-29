# Measuring Time Intervals Between Nodes

## Feature Description

Supports measuring the time interval between two nodes in the swimlane graph. Both automatic measurement and manual measurement are supported.

-   **Automatic measurement**: Only supports measuring the time interval between adjacent nodes within the same swimlane.
-   **Manual measurement**: Supports measuring the time interval between any two nodes, including across different swimlanes and non-adjacent nodes.

## Prerequisites

Execute the PyPTO program to generate the swimlane graph file.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Automatic measurement.

    Hover the mouse between two nodes in the panel; the time interval between the two nodes is automatically displayed.

    ![](../../../tools/figures/zh-cn_image_0000002533650005.png)

3.  Manual measurement.

    Hold down the **Alt** key and click the start and end nodes in sequence; the time interval between the two nodes is automatically displayed.

    ![](../../../tools/figures/zh-cn_image_0000002533810067.png)
