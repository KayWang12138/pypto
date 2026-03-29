# Viewing the Control Flow Graph

## Feature Description

The control flow graph displays the structure of the control flow, helping developers clearly understand control flow logic such as loops and branch conditions, as well as the relationships between them.

## Prerequisites

Execute the PyPTO program to generate the run results interface. For details, refer to [Data Preparation](../introduction/data_preparation.md).

## Steps

1.  Open the control flow graph.

    In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002523427405.png) icon to open PyPTO Toolkit, then click **"Results Overview"** in the run results interface.

    **Figure 1**  Control flow graph
    ![](../../../tools/figures/control_flow.png)

    The control flow graph displays LOOP (loop nodes), IF (branch condition nodes), and PATH nodes, along with their corresponding execution relationships.

    You can use the mouse scroll wheel and the minimap to control the zoom level.

    -   When the zoom level is 50% or above, all control flow details are displayed.
    -   When the zoom level is between 30% and 50%, only the outermost LOOP labels are fully displayed.
    -   When the zoom level is below 30%, details inside PATH nodes are hidden.

2.  View the code snippet corresponding to a LOOP or IF node.

    Click the code preview icon to the right of the LOOP name to preview the corresponding code snippet. You can also click **"Split View"** to further view the source code.

    **Figure 2**  Viewing a code snippet

    ![](../../../tools/figures/zh-cn_image_0000002533948285.png)

3.  View the condition expression of an IF node.

    Hover the mouse over an IF node to view its condition expression.

    **Figure 3**  Viewing the IF node condition expression
    ![](../../../tools/figures/if_condition_expression.png)

4.  View PATH nodes.

    **Figure 4**  PATH node
    ![](../../../tools/figures/path.png)

    **Table 1**  PATH node parameter descriptions

    | Parameter | Description |
    |--|--|
    | PATH name | For example, `PATH0` as shown in the figure above. Click the PATH name to navigate to the corresponding computation graph; click the ![](../../../tools/figures/zh-cn_image_0000002490272452.png) icon on the right to navigate to all computation graphs. |
    | Number of leaf functions | Displays the number of sub-functions in the current PATH node. Click to navigate to the details page. |
    | Number of times leaf is called | Displays the number of times the current PATH node is called by other functions. Click to navigate to the details page. |
    | Number of times leaf is invoked | Displays the number of times the current PATH node calls other functions. Click to navigate to the details page. |


    The following provides a specific explanation of the parameters in the table above. As shown in [Figure 5](#fig5), suppose PATH0 has three functions: leaf1, leaf2, and leaf3, where leaf1 is called by PATH0 twice, leaf2 is called by PATH0 four times, and leaf3 is called by PATH0 once and by PATH1 three times. In total, the number of functions in PATH0 is 3, the number of times PATH0 is called is 7, and the total number of times leaf1, leaf2, and leaf3 inside PATH0 are invoked is 10. Special note: leaf3 belongs to PATH0, so PATH1 has only 1 leaf function, which is leaf4.

    (fig5)=
    **Figure 5**  Call relationship diagram<a id="fig5"></a>
    ![](../../../tools/figures/invoke_relation.png)
