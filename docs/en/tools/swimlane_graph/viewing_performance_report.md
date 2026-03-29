# Viewing the Performance Report

## Feature Description

Supports viewing aggregated performance data. Views are available at the global level, by time range, by region, and by task node.

## Prerequisites

Execute the PyPTO program to generate the swimlane graph file.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  View global-level performance data.

    Click the ![](../../../tools/figures/zh-cn_image_0000002521522013.png) button on the right side of the swimlane graph.

    ![](../../../tools/figures/zh-cn_image_0000002502051718.png)

    A performance statistics panel pops out on the right side of the page.

    ![](../../../tools/figures/zh-cn_image_0000002502428688.png)

    **Table 1**  Parameter descriptions

    | Parameter | Description |
    |--|--|
    | Task Time | Total execution time — i.e., the time from when the first Task starts executing to when the last Task finishes. |
    | iCache hit rate (not currently supported) | iCache hit rate. |
    | AICore Time | Total task execution time across all AI Core swimlanes. |
    | AICore utilization | AI Core Time / (total time axis duration × number of AICore swimlanes). |
    | Cube Cycles (not currently supported) | Clock cycles occupied by the Cube unit. |
    | Vector Cycles (not currently supported) | Clock cycles occupied by the Vector unit. |


3.  View performance data for a specific time range.

    For detailed steps, refer to [Viewing the Swimlane Graph by Time Range](viewing_by_time_range.md).

4.  View performance data for a specific region.

    In the swimlane graph interface, click and drag to define a measurement region. The performance data for that region is displayed below.

    ![](../../../tools/figures/zh-cn_image_0000002534229551.png)

    For detailed parameter descriptions, refer to [Viewing the Swimlane Graph by Time Range](viewing_by_time_range.md).

5.  View performance data for a specific node.

    For detailed steps, refer to [Viewing the Swimlane Graph](viewing_swimlane_graph.md).
