# Viewing the Swimlane Graph by Time Range

## Feature Description

Supports viewing the swimlane graph and performance report within a specific time range.

## Prerequisites

Execute the PyPTO program to generate the swimlane graph.

## Steps

1.  Open the swimlane graph file. Two methods are available:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491265870.png) icon to open PyPTO Toolkit, then open the swimlane graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the swimlane graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  In the swimlane graph interface, click the ![](../../../tools/figures/zh-cn_image_0000002521083697.png) button on the right side.

    ![](../../../tools/figures/zh-cn_image_0000002533811099.png)

3.  In the task panel, click and drag to define the start and end points of the measurement range, thereby determining the time range. The system has an auto-snap feature: after dragging is complete, the start and end points automatically snap to the nearest positions within a certain range.
4.  After dragging, the measurement range brightens. The time axis shows the time of the measured region, and a performance summary panel for the measured region pops up below. ![](../../../tools/figures/zh-cn_image_0000002533651509.png)

    **Table 1**  Performance data parameter descriptions

    | Parameter | Description |
    |--|--|
    | AICore utilization | Total task execution time in AICore (AIC + AIV) swimlanes within the measurement dimension range / (total duration of the selected range × number of AICore swimlanes). |
    | Cube utilization | Total task execution time in AIC swimlanes within the measurement dimension range / (total duration of the selected range × number of AIC swimlanes). |
    | Vector utilization | Total task execution time in AIV swimlanes within the measurement dimension range / (total duration of the selected range × number of AIV swimlanes). |
    | Cube Cycle (not currently supported) | Sum of the `cube_busy_cycles` attribute values of tasks within the measurement dimension range. |
    | Vector Cycle (not currently supported) | Sum of the `vec_busy_cycles` attribute values of tasks within the measurement dimension range. |


5.  In the search panel at the lower right, you can search for performance data of related events based on criteria.

    **Table 2**  Event performance data parameter descriptions

    | Parameter | Description |
    |--|--|
    | Root Func. Hash | List of root hashes from the `event-hint` field of all events within the selected range. |
    | Leaf Func. Hash | List of leaf hashes from the `event-hint` field of all events within the selected range. |
    | Occurrences | Number of tasks with that `leafHash` within the selected range. |
    | Wall duration (ms) | Total execution time of all tasks with that `leafHash` within the selected range. |
    | Avg. Wall duration (...) | Total execution time of all tasks with that `leafHash` within the selected range divided by the number of tasks with that `leafHash`. |


6.  Click the measurement button to exit this mode.
