# Comparing Computation Graph Differences

## Feature Description

Used to compare and view the data differences between two computation graphs. Differences include node additions, deletions, and modifications. When the difference between two graphs exceeds 50% and the number of differing nodes exceeds 400, the specific difference items are not displayed.

## Prerequisites

Execute the PyPTO program to generate the computation graph files.

## Steps

1.  PyPTO Toolkit provides two ways to enter the computation graph difference comparison interface.
    -   In the Visual Studio Code workspace, right-click a computation graph file (e.g., `xx.json`), select **"PyPTO Toolkit: Advanced Operations"** from the pop-up menu, and then select **"PyPTO Toolkit: Open Computation Graph Difference Comparison"** from the expanded submenu.

        ![](../../../tools/figures/Snipaste_2025-12-26_17-04-02.png)

    -   Click the ![](../../../tools/figures/zh-cn_image_0000002512375529.png) button in the computation graph interface. ![](../../../tools/figures/zh-cn_image_0000002480216826.png)

2.  In the pop-up interface, select the computation graph to compare against (by default, the tool automatically looks for a comparison file with the same filename; for example, if the currently open file is `After_xxx.json`, it automatically searches for `Before_xxx.json` in the same directory, and vice versa. The file selection button on the right only appears if no comparison file is found automatically).

    ![](../../../tools/figures/zh-cn_image_0000002480444986.png)

3.  View the comparison results.

    ![](../../../tools/figures/zh-cn_image_0000002512376711.png)

    The left panel displays the list of differences. Differences include node additions (![](../../../tools/figures/zh-cn_image_0000002483775352.png)), deletions (![](../../../tools/figures/zh-cn_image_0000002515775339.png)), and modifications (![](../../../tools/figures/zh-cn_image_0000002484831426.png)).

4.  Click a list item to highlight the corresponding node in the graph and move it to the center of the view, making it easy for users to see the difference point.
5.  Click the name of a differing node to display detailed difference information.

    ![](../../../tools/figures/zh-cn_image_0000002502411480.png)
