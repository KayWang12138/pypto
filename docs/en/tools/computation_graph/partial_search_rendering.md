# Partial Search Rendering of the Computation Graph

## Feature Description

For extremely large data files with a high number of nodes, opening them may be slow or may fail entirely. By searching for specific nodes and partially rendering the computation graph, you can clearly display a specific path, effectively resolving loading delays and improving interaction efficiency. Two operation methods are currently provided; choose either one.

## Method 1

1.  Launch partial rendering mode using one of the following entry points:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002532734371.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface.
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  When opening a computation graph file with more than 10,000 nodes, the following interface is displayed. If the node count does not exceed 10,000, this interface is not shown and the entire graph is rendered directly.

    ![](../../../tools/figures/zh-cn_image_0000002484263090.png)

3.  Select **"Search Mode"**. The following interface is displayed.

    ![](../../../tools/figures/zh-cn_image_0000002500707780.png)

4.  Search for a node in **"Current Graph"** or **"Entire Graph"**. The computation graph will be partially rendered based on the node's predecessor and successor path links.

    **Figure 1**  Rendering based on the current graph
    ![](../../../tools/figures/rendering_based_on_current_graph.png)

    **Figure 2**  Rendering based on the entire graph
    ![](../../../tools/figures/rendering_based_on_entire_graph.png)

5.  Click a search result in the list to enter the rendering of the path related to that node. The plugin also records the search results for this file, so that the next time you open the search interface for this file, clicking the history entry will render the computation graph.

    ![](../../../tools/figures/zh-cn_image_0000002532707651.png)

## Method 2

1.  Launch partial rendering mode.

    In the Visual Studio Code workspace, right-click the computation graph file, select **"PyPTO Toolkit: Advanced Operations"** from the pop-up menu, and then select **"PyPTO Toolkit: Open Computation Graph by Node Search"** from the submenu.

2.  The following interface is displayed.

    ![](../../../tools/figures/zh-cn_image_0000002532704255.png)

3.  Search for a node in **"Current Graph"** or **"Entire Graph"**. The computation graph will be partially rendered based on the node's predecessor and successor path links.

    **Figure 3**  Rendering based on the current graph
    ![](../../../tools/figures/rendering_based_on_current_graph_0.png)

    **Figure 4**  Rendering based on the entire graph
    ![](../../../tools/figures/rendering_based_on_entire_graph_1.png)

4.  Click a search result in the list to enter the rendering of the path related to that node. The plugin also records the search results for this file, so that the next time you open the search interface for this file, clicking the history entry will render the computation graph.

    ![](../../../tools/figures/zh-cn_image_0000002500707666.png)
