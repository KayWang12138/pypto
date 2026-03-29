# Searching Computation Graph Nodes

## Feature Description

After opening a computation graph, you can search for node information within the graph. The following search modes are supported:

-   **Current graph search** (default): Searches only the nodes in the current computation graph.
-   **Entire graph search**: Searches all subgraphs associated with the computation graph.
-   **Path-only search**: Triggered when viewing or locking a specific path; searches only the node information along the currently locked path.

In addition, precise and fuzzy search by specific labels is also supported.

## Prerequisites

Execute the PyPTO program to generate the computation graph files.

## Steps

1.  Open the computation graph file. Two methods are available; choose either one:
    -   In the Visual Studio Code sidebar, click the ![](../../../tools/figures/zh-cn_image_0000002491082472.png) icon to open PyPTO Toolkit, then open the computation graph file in the run results interface (this method only supports opening files whose names start with `After`).
    -   In the Visual Studio Code workspace, right-click the computation graph file and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

2.  Search using different search modes.
    -   **Current graph search**: Select **"Current Graph"** in the top search bar, enter a search keyword, and click a search result to navigate to the corresponding node in the current graph.

        ![](../../../tools/figures/zh-cn_image_0000002512352587.png)

    -   **Entire graph search**: Select **"Entire Graph"** in the top search bar, enter a search keyword, and click a search result to navigate to the corresponding subgraph.

        ![](../../../tools/figures/zh-cn_image_0000002480352670.png)

    -   **Path-only search**: Lock a specific path (for detailed steps, see [Locking a Computation Path](locking_computation_path.md)), enter a search keyword, and click a search result to navigate to the corresponding node in the current graph.

        ![](../../../tools/figures/zh-cn_image_0000002480192706.png)

3.  Search by specified label. Configure the following settings in the top search bar.

    ![](../../../tools/figures/zh-cn_image_0000002512312555.png)

    You can select a parent label and a child label for a precise search, or omit the child label for a fuzzy search. When performing a fuzzy search, you can specify whether to be case-sensitive and whether to use regular expressions, as shown below:

    ![](../../../tools/figures/zh-cn_image_0000002512316779.png)
