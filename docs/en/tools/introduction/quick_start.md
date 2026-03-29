# Quick Start

## Task and Goals

This section introduces how to use PyPTO Toolkit through a PyPTO program example to view the computation graph structure and observe related performance data through the swimlane graph. The specific goals are:

-   Generate and view the computation graph to understand the computation logic and data flow relationships through the graph structure.
-   Generate and view the swimlane graph to understand task execution order, timing information, and identify performance bottlenecks.

## Compiling and Executing a PyPTO Program

The following example uses the Softmax operator. The sample code is located in the [PyPTO open-source repository](https://gitcode.com/cann/pypto): `examples/02_intermediate/operators/softmax/softmax.py`.

1.  Switch to the directory containing the sample and run the following commands in an environment where PyPTO is installed.

    ```bash
    # Configure the CANN environment variables.
    # After installation, set the environment variables. Run the following command
    # using the actual path of set_env.sh.
    # These environment variable settings only take effect in the current shell window.
    # You can write the commands to an environment variable configuration file
    # (e.g., .bashrc) as needed.

    # Default installation path, using the root user as an example
    # (for non-root users, replace /usr/local with ${HOME}).
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Set the device ID.
    export TILE_FWK_DEVICE_ID=0

    # Run the script.
    python3 softmax.py
    ```

2.  If the program runs successfully, the following output is displayed.

    ```text
    Input shape: torch.Size([32, 32, 1, 256])
    Output shape: torch.Size([32, 32, 1, 256])
    Max difference：0.000000
    ✓ Softmax test passed
    ```

    At the same time, compilation and execution result files are generated in the `${work_path}/output/output_*/` directory (where `*` represents a timestamp).

## Viewing the Computation Graph

During the compilation of a PyPTO program, a graph structure composed of Tensors and Operations — the computation graph — is automatically generated. This computation graph undergoes the PyPTO compilation and optimization pipeline, completing the compilation process from the original computation graph to the executable graph, and ultimately generating executable code that can run in an Ascend hardware environment to perform actual computation tasks. Users can view key information in the computation graph using the PyPTO Toolkit visualization tool.

1.  Right-click the `${work_path}/output/output_*/program.json` file, and select **"PyPTO Toolkit: Open File"** from the pop-up menu.

    The `program.json` file contains summary information about the Execute Graph and Block Graph. Key information in the graph includes: cards on the left and right sides are Tensor nodes (representing input/output data), and cards in the middle are call nodes (marked with the `fx` identifier; double-click to drill down into details).

    ![](../../../tools/figures/zh-cn_image_0000002533799219.png)

2.  Double-click the middle card to drill down layer by layer to the Execute Graph shown below.

    ![](../../../tools/figures/zh-cn_image_0000002501879368.png)

    The different colored blocks (`CALL:TENSOR_xx`) each represent a call node, indicating that the computation graph has been partitioned into different Block Graph subgraphs.

3.  Double-click a call node in the above graph to see the Block Graph subgraph information, which identifies the specific execution process of the task.

    ![](../../../tools/figures/zh-cn_image_0000002533639257.png)

    After zooming in, you can see the specific Tensor and Operation node information and their connection relationships in the graph:

    ![](../../../tools/figures/zh-cn_image_0000002502039216.png)

## Viewing the Swimlane Graph

The swimlane graph intuitively displays the actual scheduling and execution process of the computation graph, clearly presenting the execution order and timing information of tasks, helping developers analyze operator performance bottlenecks. The following describes how to collect swimlane graph data and view the swimlane graph through PyPTO Toolkit.

1.  Enable the performance data collection feature.

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2.  Re-execute the operator program.

    ```bash
    python3 softmax.py
    ```

    A swimlane graph data file named `merged_swimlane.json` is generated in the `${work_path}/output/output_*/` directory (where `*` represents a timestamp).

3.  View the swimlane graph through the PyPTO Toolkit plugin.

    Right-click `merged_swimlane.json` and select **"PyPTO Toolkit: Open File"** from the pop-up menu, as shown in [Figure 1](#fig1).

    (fig1)=
    **Figure 1**  Swimlane graph interface  <a id="fig1"></a>
    ![](../../../tools/figures/swimlane_graph.png)

    The task panel area in the figure above displays each task sequence in swimlane form, with each AIV/AIC thread corresponding to one horizontal swimlane. Each colored block in a swimlane represents a specific task executed on it; its length corresponds to the task's execution time, intuitively reflecting the computation load and density. Users can analyze potential performance bottlenecks by observing the idle gaps between adjacent swimlanes (shown as black areas in the figure, also called "bubbles") and swimlane entries with long execution times.
