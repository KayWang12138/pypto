# Jumping to a Code Line

## Feature Description

In the computation graph, you can jump to the code line associated with an Operation node. However, not all Operation nodes support this feature. Only operations explicitly called in the PyPTO program — such as `matmul`, `add`, etc. — support jumping. Operations that are automatically inserted during the Pass compilation process cannot be jumped to.

## Steps

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Right-click an Operation node and select **"Jump to Code Line"**.

    ![](../../../tools/figures/zh-cn_image_0000002502264604.png)

    The following interface is displayed:

    ![](../../../tools/figures/zh-cn_image_0000002534144793.png)
