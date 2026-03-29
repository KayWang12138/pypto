# Three-Column Linked View

## Feature Description

The three-column linked view simultaneously displays code, a computation graph, and a swimlane graph in a single page. These three views are interconnected through semantic labels. Users can click in any view to trigger synchronized highlighting of the corresponding content in the other two views, helping developers quickly understand the entire pipeline — from code implementation, to computation graph expression, to the final performance profile shown in the swimlane graph.

## Prerequisites

-   In the PyPTO program code, use the `pypto.set_semantic_label` interface to set a custom semantic label.

    ```python
    pypto.set_semantic_label("softmax")
    softmax_out = pypto.softmax(tile_logits_fp32, -1)
    ```

-   Execute the PyPTO program to generate both the computation graph and swimlane graph files.

## Steps

1.  In the Visual Studio Code workspace, right-click the PyPTO code file (e.g., `test_softmax_custom.py`) and select **"PyPTO Toolkit: Enable Linked Mode"** from the pop-up menu.

    The tool automatically searches for the `build/output/bin/out/output_<timestamp>` directory under the current workspace, identifies the latest timestamp directory, and reads the computation graph file `program.json` and swimlane graph file `merged_swimlane.json` from it. Combined with the code file, the three-column view is opened.

    If the target computation graph and swimlane graph files are not found during the above search, the user is prompted to select them.

2.  The three-column linked view is displayed as follows.

    ![](../../../tools/figures/zh-cn_image_0000002502528526.png)

    The interface simultaneously displays the code, computation graph, and swimlane graph. When the user moves the mouse, the system dynamically highlights view elements associated with the current semantic label.

    -   In the **code view**, colored blocks on the left mark the semantic label ranges, and the code segment associated with the current semantic label is highlighted.
    -   In the **computation graph view**, the system dynamically highlights the graph structure associated with the current semantic label, allowing users to observe the graph structure characteristics of the corresponding code line and determine whether graph structure optimizations are needed. For detailed usage of the computation graph, refer to [Computation Graph](../computation_graph/viewing_computation_graph.md). In the upper-left dropdown of the computation graph, you can select any subgraph under `program.json`.
    -   In the **swimlane graph view**, the system highlights the Task nodes associated with the current semantic label, allowing users to observe the task execution time and dependency relationships, identify performance bottlenecks, and devise optimization strategies. For detailed usage of the swimlane graph, refer to [Swimlane Graph](../swimlane_graph/viewing_swimlane_graph.md).

    If no color bars are shown in the code view, check whether the `filename` parameter in the `semantic_label` field of `program.json` is the absolute path of the source code file.

    ```text
    "semantic_label": {
    "filename": "D:\\code\\demos\\build\\output\\bin\\output\\output_20251129_095047_860533\\test_softmax_custom.py",
    "label": "topk",
    "lineno": 69
    },
    ```

3.  Enable lock mode.

    Click the lock button in any view of the three-column linked view to enable lock mode. Click the unlock button in any view to exit this mode.

    ![](../../../tools/figures/zh-cn_image_0000002534328503.png)

    After entering lock mode, in the upper-left dropdown of the computation graph, you can select other related subgraphs that share the same `semantic_label`.
