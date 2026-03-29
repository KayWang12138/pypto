# Locking a Computation Path

## Feature Description

Supports locking the current path and highlighting it, allowing users to focus on the information of the current path only.

## Prerequisites

Execute the PyPTO program to generate the computation graph files.

## Steps

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Right-click a node in the computation graph, or hover the mouse over a node and use the shortcut key (**Ctrl+H**) to select **"Lock current path and highlight"**.

    ![](../../../tools/figures/zh-cn_image_0000002502414338.png)

    The entire current path is highlighted and locked; all other nodes are grayed out. Press **ESC** to exit the lock.

    ![](../../../tools/figures/zh-cn_image_0000002480215508.png)

3.  Right-click a node in the computation graph, or hover the mouse over a node and use the shortcut key (**Ctrl+H**) to select **"View current path only"**.

    The entire current path is highlighted and locked; all other nodes disappear. Press **ESC** to exit the lock.

    ![](../../../tools/figures/zh-cn_image_0000002512495361.png)
