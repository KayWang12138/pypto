# Controlling Layer Layout Display

## Feature Description

The computation graph supports layer control capabilities. You can use the layer controls to change the layout direction of the computation graph and configure the node display type.

## Prerequisites

Execute the PyPTO program to generate the computation graph files.

## Steps

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Click the ![](../../../tools/figures/zh-cn_image_0000002512322027.png) button in the computation graph interface.

    ![](../../../tools/figures/zh-cn_image_0000002521609447.png)

3.  Configure the corresponding layer options.

    ![](../../../tools/figures/zh-cn_image_0000002489609666.png)

    **Table 1**  Parameter descriptions

    | Parameter | Description |
    |--|--|
    | Graph layout | **Left-to-right**: Displays the computation graph in a left-to-right layout.<br>**Top-to-bottom**: Displays the computation graph in a top-to-bottom layout. |
    | Node display mode | **Tensor+OP**: Shows both Tensor nodes and Operation nodes on the computation graph.<br>**OP**: Shows only Operation nodes on the computation graph. |
    | Node control | Specifies which parameter information is displayed on Tensor and Operation nodes. For example, if configured as `shape`, only shape information is shown. |
