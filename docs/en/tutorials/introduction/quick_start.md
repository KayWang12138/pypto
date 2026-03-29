# Quick Start

## Task and Objectives

This section provides a detailed introduction to how to use the PyPTO framework to implement a simple Softmax operator and verify its correctness through test cases. By studying this section, you will learn how to use PyPTO's API to build custom operators. After completing the program, you can also use the PyPTO Toolkit visualization tool to view the computation graph structure and observe various performance metrics of the operator.

The example code for the Softmax operator implementation is located at: [softmax.py](../../../examples/02_intermediate/operators/softmax/softmax.py). Users can refer to the code example to understand this section.

## Operator Design Specification

**Table 1**  Softmax Operator Design Specification

| name                | shape            | data type | format |
| ------------------- | ---------------- | --------- | ------ |
| **inputs**          | (-1, 32, 1, 256) | float     | ND     |
| **outputs**         | (-1, 32, 1, 256) | float     | ND     |

* Mathematical expression

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i - M)}{\sum_j \exp(z_j - M)}$

* Primary interfaces used

  Basic computation interfaces: exp, sum, / (div), amax, - (sub)

## Importing PyPTO Modules

Before implementing the Softmax operator, you first need to import PyPTO, PyTorch, and NumPy modules. The PyPTO module provides tensor operations and compilation capabilities, while the PyTorch and NumPy modules are used for result validation.

```python
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
```

## Core Code Logic

1.  Implement the core computation function.

    PyPTO provides a rich set of Operation interfaces for implementing different computation logic. Developers can combine different Operation interfaces according to the operator's mathematical expression to implement complex computation logic. The following is the implementation of the core computation function for the Softmax operator:

    ```python
    def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
        row_max = pypto.amax(x, dim=-1, keepdim=True)  # Compute row maximum
        sub = x - row_max                              # Normalize values
        exp = pypto.exp(sub)                           # Exponential operation
        esum = pypto.sum(exp, dim=-1, keepdim=True)    # Sum
        return exp / esum                              # Normalize probabilities
    ```

2.  Implement the Softmax Kernel function.

    To enable the computation logic to run efficiently on hardware, you need to implement the Softmax Kernel function. Use the `@pypto.frontend.jit` decorator to convert the computation graph into hardware instructions, and define data partitioning and loop processing strategies within it. When calling, simply pass in PyTorch Tensors directly, and the PyPTO framework will automatically handle tensor type conversion.

    ```python
    @pypto.frontend.jit
    def softmax_kernel(
        input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
        output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    ):
        bs, seqlen, head, dim = input_tensor.shape
        tile_b = 1  # Process one batch at a time
        b_loop = bs // tile_b

        # Tiling shape setting for efficient execution
        pypto.set_vec_tile_shapes(1, 4, 1, 64)

        for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]
            softmax_out = softmax_core(input_view)
            output_tensor[b_offset:, ...] = softmax_out

    ```

    To improve the computation efficiency of the operator, you can specify the tiling method through the `set_vec_tile_shapes` or `set_cube_tile_shapes` interface. This tiling configuration decomposes computation into hardware-friendly tile granularity (e.g., 64), which can optimize memory access and parallel computation efficiency.

    ```python
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    ```

## Test Cases

To verify the correctness of the Softmax operator, write a test case. This test case uses PyTorch Tensors as input, computes using the PyPTO kernel, and compares the results against PyTorch's built-in Softmax function. Before running PyPTO and PyTorch related code, you need to specify the corresponding Device ID, or obtain the current Device ID through the torch.npu interface.

```python
def test_softmax(device_id: int = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    x = torch.rand(shape, dtype=torch.float, device=device)
    y = torch.zeros(shape, dtype=torch.float, device=device)

    softmax_kernel(x, y) # default dim: -1
    golden = torch.softmax(x, dim=-1).cpu()
    y = y.cpu()

    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()
```

## Compilation and Execution

Switch to the directory where the example code is located, and run in an environment with PyPTO installed:

```bash
# Configure CANN environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set device ID
export TILE_FWK_DEVICE_ID=0

# Execute script
python3 softmax.py
```

After successful program execution, the following information is displayed:

```text
Input shape: torch.Size([32, 32, 1, 256])
Output shape: torch.Size([32, 32, 1, 256])
✓ Softmax test passed
```

At the same time, compilation and execution result files will be generated in the $\{work\_path\}/output/output\_\*/ directory (\* represents a timestamp).

## Viewing the Computation Graph

During the compilation process, the PyPTO program automatically generates a graph structure composed of tensors and operations — the computation graph. This computation graph goes through PyPTO's compilation optimization process, completing the compilation from the original computation graph to an executable graph, ultimately generating executable code that can run in the Ascend hardware environment to perform actual computation tasks. Users can use the PyPTO Toolkit visualization tool to view key information in the computation graph.

1.  Right-click the $\{work\_path\}/output/output\_\*/program.json file, and select "Open with PyPTO Toolkit" from the pop-up menu.

    The program.json file contains summary information for the Execute Graph and Block Graph. Key information in the graph includes: cards on the left and right sides are Tensor nodes (representing input/output data), and the cards in the middle are call nodes (marked with fx identifiers; click to drill down for more information).

    ![](../figures/zh-cn_image_0000002499877218.png)

2.  Double-click the middle card to drill down layer by layer to the Execute Graph shown below.

    ![](../figures/zh-cn_image_0000002531853385.png)

    Different colored blocks (CALL:TENSOR\_xx) in the graph each represent a call node, indicating that the computation graph has been divided into different Block Graph subgraphs.

3.  Double-click a call node in the above graph to view the Block Graph subgraph information, which marks the specific execution process of the task.

    ![](../figures/zh-cn_image_0000002531638777.png)

    After zooming in, you can see the specific tensor and operation node information and connection relationships in the graph:

    ![](../figures/zh-cn_image_0000002499719036.png)

## Viewing the Swimlane Graph

The swimlane graph provides an intuitive display of the actual scheduling and execution process of the computation graph, clearly showing the execution order and timing information of tasks, helping developers analyze operator performance bottlenecks. The following describes how to collect swimlane graph data and view it through the PyPTO Toolkit.

1.  Enable the performance data collection feature by configuring the graph execution stage debug switch through the `debug_options` parameter of the `@pypto.frontend.jit` decorator.

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2.  Re-execute the operator program.

    ```bash
    python3 softmax.py
    ```

    A swimlane graph data file named `merged_swimlane.json` will be generated in the $\{work\_path\}/output/output\_\*/ directory (\* represents a timestamp).

3.  View the swimlane graph using the PyPTO Toolkit plugin.

    Right-click merged\_swimlane.json, and select "Open with PyPTO Toolkit" from the pop-up menu, as shown below.

    **Figure 1**  Swimlane graph interface
    ![](../figures/swimlane_graph.png "Swimlane graph interface")

    The colored sections in the figure above are the swimlanes, showing the task execution on each AIC/AIV. The length of swimlane entries corresponds to the task duration, intuitively reflecting the density of computation. Users can analyze potential performance bottlenecks by observing idle gaps between adjacent swimlanes (such as the black areas in the figure, also called bubbles) and swimlane entries with longer durations.

