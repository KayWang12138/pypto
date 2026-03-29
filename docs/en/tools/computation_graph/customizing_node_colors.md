# Customizing Node Colors and Node Information

## Feature Description

Supports customizing the colors and default display information of computation graph nodes.

## Steps

1.  In the Visual Studio Code workspace, right-click the PyPTO Toolkit plugin name, select **"Settings"**, and enter the configuration panel interface.
2.  Select **"Extensions > PyPTO Toolkit > Computational Graph"** to customize the configuration scheme.

    Configuration is supported by injecting JavaScript function scripts.

    -   The input parameter of the node attribute configuration function is the node's JSON content details. It returns a string array in `key:value` format; nodes will be displayed according to the returned result.
    -   The input parameter of the node color configuration function is the node's JSON content details. It returns a color string in HEX format; nodes will display the color according to the returned result.

    The following is an example of adjusting colors and display content through custom configuration:

    ![](../../../tools/figures/zh-cn_image_0000002521728497.png)

    This example respectively configures:

    -   Display `opcode` and `magic` on Operation nodes.
    -   Set the Operation node color to `#8d6f64`.
    -   Display `magic` and `shape` on Tensor nodes.
    -   Set the Tensor node color to `#cc9595`.

3.  Clear the cache to apply the configuration.

    ![](../../../tools/figures/zh-cn_image_0000002534151235.png)

4.  The displayed effect on the page is as follows.

    ![](../../../tools/figures/zh-cn_image_0000002489608896.png)
