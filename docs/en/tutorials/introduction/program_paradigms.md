# Programming Paradigms

PyPTO adopts the PTO programming paradigm. The core idea is to use tensors as the basic representation of data, and to describe and assemble a complete computation process (or computation graph) through a series of basic operations on tensors. In PyPTO, all operations take tensors as input or output, forming a traceable computation graph structure that facilitates subsequent debugging, optimization, and compilation and execution on specific hardware.

## Overview of the PTO Programming Paradigm

The core design principles of the PTO programming paradigm include:

-   Tensor-level abstraction: Describes computation using tensors rather than individual elements, staying close to the mathematical expressions of algorithm designers.
-   Declarative programming: Developers only need to describe "what to do," and the framework automatically handles "how to do it."
-   Tile-based computation: All computation is ultimately performed based on tiles (hardware-aware data blocks), fully exploiting hardware parallel computing capabilities.
-   Computation graph driven: By building a computation graph, the framework can automatically perform optimization, scheduling, and execution.

PyPTO provides three levels of programming interfaces:

-   Tensor-level programming: Directly using tensors and tensor operations to build computation graphs.
-   Tile-level programming: Expressing complete computation using tiles and tile operations, explicitly representing memory access and dependencies.
-   Block-level programming: Defining computation graphs executed by a single processor core and implementing overall computation through multiple instantiations.

The current version only exposes tensor-level programming, which is the most commonly used and recommended programming approach.

## Core Data Structures

-   Tensor: The most fundamental data structure in PyPTO, representing a multi-dimensional array. A tensor contains the following information:

    -   Data type (dtype): Such as FP32, FP16, INT32, BOOL, etc.
    -   Shape: An integer array describing the length of each dimension, e.g., \(32, 64\), \(1, 32, 128\), etc.
    -   Format: The arrangement format of data in memory.
    -   Name: Used to identify the tensor in the computation graph, facilitating debugging and visualization.

    Tensors can be combined through various operations (such as addition, multiplication, indexing, reduction, etc.), which typically generate new tensors or change their reference method in the computation graph (e.g., creating views, changing shapes, transposing, etc.).

-   Tile: A sub-interval (sub-tensor) of a tensor, created by tiling (partitioning) a large tensor into multiple sub-blocks. The design purpose of tiles is to:

    -   Allow them to be stored in the private cache (such as UB, L1) of a processor core to improve data locality.
    -   Fully utilize hardware parallel computing capabilities.
    -   Optimize memory access patterns.

    In tensor-level programming, tiling is done automatically by the framework. Developers only need to specify the TileShape through the configuration interface, and the framework will perform the partitioning automatically.

-   View and Assemble: Provide view and combination operations on sub-tensors, which are very useful for handling dynamic shapes and loop computation.
    -   View: Provides view operations on sub-tensors, allowing access to sub-intervals of a tensor without copying data.
    -   Assemble: Combines multiple sub-tensors into a larger tensor.

## Tensor-Level Programming

Tensor-level programming is the primary programming approach currently supported by PyPTO. Developers directly use tensors and tensor operations to build computation graphs without needing to worry about underlying tile partitioning and hardware details.

-   Basic programming pattern

    The typical tensor-level programming pattern is as follows. The kernel entry function is defined through the `@pypto.frontend.jit` decorator and will be JIT compiled on the first call.

    ```python
    import pypto

    # 1. Configure Tiling (optional, the framework will provide default values)
    pypto.set_vec_tile_shapes(64)

    # 2. Define computation function
    @pypto.frontend.jit
    def my_operator(a: pypto.Tensor(shape, dtype), b:  pypto.Tensor(shape, dtype), output:  pypto.Tensor(shape, dtype)):
        # Tensor operations
        result = a + b  # or use pypto.add(a, b)
        output[:] = result

    # 3. Execute
    my_operator(tensor_a, tensor_b, output_tensor)
    ```

-   Tensor operations

    PyPTO provides a rich set of tensor operations, including:

    -   Mathematical operations: add, sub, mul, div, matmul, etc.
    -   Logical operations: logical\_not, etc.
    -   Structural transformations: reshape, transpose, view, unsqueeze, etc.
    -   Reduction operations: sum, amax, amin, topk, etc.
    -   Activation functions: sigmoid, softmax, etc.
    -   Transcendental functions: exp, log, etc.
    -   Other operations: gather, scatter, concat, assemble, etc.

-   Control flow

    PyPTO supports control flow operations for handling dynamic shapes and conditional execution:

    -   Loop

        ```python
        # Process data with dynamic dimensions
        tile_size = pypto.symbolic_scalar(64)
        loop_count = dynamic_shape / tile_size

        for idx in pypto.loop(0, loop_count, 1, name="LOOP_BATCH"):
            offset = idx * tile_size
            end = (idx + 1) * tile_size
            input_view = input_tensor[offset:end, :]
            output_tensor[offset:end, :] = process_tile(input_view)
        ```

    -   Conditional branch

        ```python
        for idx in pypto.loop(b_loop):
            t3_sub = t0_sub + t1_sub
            if pypto.cond(idx < 2):  # Dynamic conditional check
                t2[b_offset:b_offset_end, ...] = t3_sub + 1
            else:
                t2[b_offset:b_offset_end, ...] = t3_sub
        ```

-   Symbolic programming

    PyPTO supports symbolic scalars (SymbolicScalar) to enable the expression and processing of dynamic shape tensors, allowing the framework to perform shape inference and optimization at compile time.

    ```python
    # Create a dynamic shape tensor
    tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "dynamic")

    # Get the symbolic scalar for the dynamic dimension, obtaining the concrete value at runtime
    b = pypto.symbolic_scalar(tensor_shape[0])
    ```

## Computation Graph

-   Composition of computation graphs

    PyPTO's computation graph consists of the following elements:

    -   Tensor: Data nodes.
    -   Operation (Op): Operations on data, divided into Tensor Ops and Tile Ops.
        -   Tensor Op: Operates on tensors, logically unconstrained by storage location or scale.
        -   Tile Op: A subset of Tensor Ops, restricted to inputs and outputs located in the L1 memory of the same core, ensuring data locality.

-   Computation graph transformation process

    The user-defined computation graph is ultimately converted into executable code:

    ![](../figures/transformation_process.png)

-   Viewing computation graphs

    PyPTO provides multiple ways to view computation graphs:

    -   JSON format: Export to JSON format for programmatic analysis.
    -   Visualization tool: Visualize the computation graph structure using the PyPTO Toolkit plugin.

## MPMD Execution Model

PyPTO is based on the MPMD (Multiple Program Multiple Data) execution model. Compared to the traditional SPMD (Single Program Multiple Data) model:

-   SPMD: Users need to write a single kernel logic and instantiate it to run on multiple processor cores, incurring synchronization overhead and performance bottlenecks.
-   MPMD: Computation is abstracted as a set of heterogeneous tasks organized through dependency relationships. The runtime scheduler assigns tasks to appropriate execution units based on dependencies, avoiding global synchronization constraints and improving overall utilization and efficiency.

The advantages of the MPMD execution model include:

-   Flexible scheduling: Different tasks can be assigned to different processor cores, avoiding global synchronization.
-   Better resource utilization: Selecting appropriate execution units based on task characteristics.
-   Fine-grained parallelism: Computation loads can be parallelized at fine granularity while also being flexibly scheduled at the task level.
-   Multi-core architecture compatibility: Better adaptation to the multi-core architecture of NPUs.

The execution flow is:

![](../figures/execution_process_flow.png)

## Programming Examples

Leveraging the PTO programming paradigm, developers can efficiently develop diverse operators and integrate them seamlessly with PyTorch.

-   Vector Add

    ```python
    import pypto

    # Configure Tiling
    pypto.set_vec_tile_shapes(64)

    # Define computation function
    @pypto.frontend.jit
    def vector_add(a:  pypto.Tensor(shape, dtype), b:  pypto.Tensor(shape, dtype), output:  pypto.Tensor(shape, dtype)):
        # Tensor operation: vector addition
        output[:] = a + b  # Output result

    # Execute
    vector_add(tensor_a, tensor_b, output_tensor)
    ```

-   Matrix Multiplication

    ```python
    import pypto

    # Configure Cube Tiling (for matrix multiplication)
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])

    @pypto.frontend.jit
    def matmul(a:  pypto.Tensor(shape_a, dtype), b:  pypto.Tensor(shape_b, dtype), output:  pypto.Tensor(shape_c, dtype)):
        outputs[:] = pypto.matmul(a, b)  # Matrix multiplication

    # Execute
    matmul(matrix_a, matrix_b, output_matrix)
    ```

-   Dynamic Shape Handling

    ```python
    import pypto

    def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
        row_max = pypto.amax(x, dim=-1, keepdim=True)  # Compute row maximum
        sub = x - row_max                              # Normalize values
        exp = pypto.exp(sub)                           # Exponential operation
        esum = pypto.sum(exp, dim=-1, keepdim=True)    # Sum
        return exp / esum                              # Normalize probabilities

    @pypto.frontend.jit
    def dynamic_softmax(input_tensor :  pypto.Tensor(in_shape, dtype), output_tensor:  pypto.Tensor(out_shape, dtype)):
        # Get dynamic dimension
        batch_size = input_tensor.shape[0]
        tile_size = pypto.symbolic_scalar(64)
        loop_count = batch_size // tile_size

        # Loop processing
        for idx in pypto.loop(0, loop_count, 1, name="LOOP_BATCH"):
            offset = idx * tile_size
            end = (idx + 1) * tile_size

            # Extract current tile
            x_view = input_tensor[offset:end, :]

            # Compute Softmax
            softmax_out = softmax_core(x_view)

            # Assemble results
            output_tensor[offset:end, :] = softmax_out

    # Execute
    dynamic_softmax(input_tensor, output_tensor)
    ```

-   Integration with PyTorch

    ```python
    import pypto
    import torch

    @pypto.frontend.jit
    def my_operator(x: pypto.Tensor(in_shape, dtype), output: pypto.Tensor(out_shape, dtype)):
        result = pypto.matmul(x, weight)
        output[:] = result

    # Use PyTorch Tensor
    input_torch = torch.randn(32, 128, device='npu')
    output_torch = torch.zeros(32, 64, device='npu')

    # Execute
    my_operator(input_torch, output_torch)
    ```

## Summary

The PTO programming paradigm, through tensor-level abstraction, enables developers to express computation logic in a more intuitive way, while the framework automatically handles the underlying optimization, scheduling, and execution. This design not only ensures simplicity of development but also fully utilizes the parallel computing capabilities of the hardware, providing an efficient and flexible solution for AI accelerator programming.

