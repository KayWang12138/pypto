# Introduction

PyPTO (pronounced: pai p-t-o) is a high-performance programming framework for AI accelerators, introduced by CANN, designed to simplify the operator development workflow while retaining high-performance computing capabilities. The framework adopts the innovative PTO (Parallel Tensor/Tile Operation) programming paradigm, with a tile-based programming model as its core design concept. Through multi-level computation graph representation, it progressively compiles AI models built by users through the API—from high-level tensor computation graphs down to hardware instructions—ultimately generating code that can be efficiently executed on the target platform, and scheduled on the device side in an MPMD (Multiple Program Multiple Data) manner.

## Core Architecture

The PyPTO framework adopts a layered architecture design. From the user API to the underlying hardware execution, it is organized into the following layers:

![](../figures/pypto_architecture.png)

-   User Interface Layer: This is the interface layer between the PyPTO framework and developers. It provides Python-friendly programming interfaces that allow developers to express computation logic intuitively, without needing to understand low-level hardware implementation details.
-   Compute Graph Compile Layer: PyPTO adopts multi-level computation graph representation, supporting optimization and transformation of computation graphs across multiple abstraction levels from high to low.

    -   Tensor Graph: High-level tensor operations, close to the mathematical expressions of algorithm designers.
    -   Tile Graph: Hardware-aware tile operations that fully exploit hardware parallelism and memory hierarchy.
    -   Block Graph: Subgraph partitioning, supporting parallel execution and resource management.
    -   Execute Graph: Execution graph, containing dependency relationships and scheduling information.

    The compilation process is implemented through modular passes, where each stage consists of multiple passes responsible for specific-stage optimization or transformation tasks.

    -   Tensor Graph stage: Implements hardware-independent graph optimizations, including redundant operation elimination, type conversion, memory conflict inference, etc.
    -   Tile Graph stage: Performs tile expansion based on TileShape, implementing tile-level optimizations including memory type allocation, move operation generation, subgraph partitioning, etc.
    -   Block Graph stage: Generates computation subgraphs through partitioning, performing block-level optimizations including out-of-order scheduling, memory reuse, synchronization point insertion, etc.
    -   Execute Graph stage: Integrates computation subgraph information and orchestrates the generation of the final execution graph.

-   Code Generation Layer: Converts the optimized computation graph into executable code for the target platform.
    -   Virtual instruction generation: Generates PTO virtual instruction code (PTO Virtual Instructions) from the Execute Graph.
    -   Target platform compilation: Compiles virtual instructions into target platform code.

-   Scheduling & Execution Layer: Responsible for scheduling and executing the generated code on the device.
    -   MPMD scheduling: Executable code is scheduled to device processor cores on the device side via MPMD.
    -   Control flow execution: Manages task dependency relationships and executes control flow logic.

## Core Features

-   Technical innovations:
    -   Tile-based programming model: Computation is performed based on tiles (hardware-aware data blocks), fully utilizing the hardware's parallel computing capabilities and memory hierarchy.
    -   Multi-level computation graph representation and optimization: Through the Compute Graph Compile Layer, Tensor Graph is converted to Tile Graph, Block Graph, and Execute Graph, with each step containing a series of pass optimization processes.
    -   Automated code generation: Compilation results are processed through the Code Generation Layer to generate PTO virtual instruction code, which is then compiled by the compiler into executable code for the target platform.
    -   MPMD execution scheduling: Executable code is loaded onto the device side and scheduled to device processor cores via MPMD for efficient parallel execution.
    -   Complete toolchain support: Compilation intermediates and runtime performance data across the entire workflow can be visualized through IDE-integrated toolchains to identify performance bottlenecks. Developers can also control compilation and scheduling behavior through the toolchain.
    -   Python-friendly API: Provides intuitive tensor-level abstraction close to the thinking mode of algorithm developers, supporting dynamic shapes and symbolic programming.
    -   Layered abstraction design: Exposes different abstraction levels to different developers — algorithm developers use the Tensor level, performance experts use the Tile level, and system developers use the Block level.

## Use Cases

PyPTO is suitable for the following scenarios:

-   Deep learning operator development: Rapidly implementing various neural network operators.
-   Large model development: Supporting large model components such as Attention, MoE, FFN, etc.
-   Dynamic shape handling: Supporting dynamic shape scenarios such as dynamic batch sizes.

## Design Philosophy

Traditional model development typically separates algorithm developers from operator developers. The root of this division lies in the complexity of high-performance operator development: operator developers must not only understand the mathematical computation properties of operators but also consider how to transform them into hardware-friendly execution forms. This is similar to the early CPU era, when programmers needed to manually arrange pipeline instructions before out-of-order execution and compiler technologies had matured.

To reduce this complexity, PyPTO proposes a new programming framework design philosophy aimed at simplifying the operator development workflow while retaining the potential for high-performance computing.

-   Computation Layer Design

    The design philosophy of the computation layer is to stay as close as possible to the mathematical expressions of algorithm designers, using tensors rather than individual elements to describe the computation process. AI models built by users through the API are expressed through a Tensor Graph. This design retains maximum optimization potential, including:

    -   Memory layout optimization: Automatically optimizing the arrangement of data in memory
    -   Data transfer optimization: Minimizing data transfers between different memory hierarchy levels
    -   Multi-operator joint optimization: Identifying and fusing optimizable operator combinations

    By using tensors as the basic data unit, the computation layer can express complex mathematical operations more naturally, while providing rich information for subsequent compilation optimization.

-   Compile Layer Design

    The compile layer is the key link connecting the computation layer and the execution layer, responsible for transforming the Tensor Graph into hardware-friendly execution form. The compilation process is implemented through a multi-stage lowering pipeline:

    -   Tensor Graph to Tile Graph: Converting tensor operations to tile operations through compilation passes, selecting tiling strategies, and performing layout transformations, tile fusion, tile reordering, etc.
    -   Tile Graph to Block Graph: Partitioning the tile graph into subgraphs, detecting isomorphic subgraphs, normalizing Block Graph, tracking dependency relationships.
    -   Block Graph to Execute Graph: Building the execution graph, analyzing dependency relationships between Block Graphs, planning global resources, generating scheduling hints.

    Each stage contains multiple optimization passes that, through modular graph transformations and optimization processes, translate the optimization space retained by the computation layer into actual performance improvements.

    The compile layer provides the following core capabilities:

    -   Quick availability: Guarantees timely generation of runnable results, meeting rapid development needs.
    -   Flexible tuning: Supports performance-sensitive configuration adjustments, allowing developers to optimize based on actual requirements.
    -   Deep optimization: Allows advanced users to deeply customize the compilation process to pursue ultimate performance.

-   Execution Layer Design

    The execution layer is responsible for converting compiled code into hardware-friendly instructions and executing them. The execution process includes:

    -   Code generation: Compilation results are processed through CodeGen to generate low-level PTO virtual instruction code.
    -   Target platform compilation: Virtual instruction code is compiled by the compiler into executable code for the target NPU platform.
    -   MPMD scheduling: Executable code is loaded onto the device side and scheduled to device processor cores via MPMD.

    Through automated code generation technology, the execution layer can automatically generate optimal execution instructions based on hardware characteristics, fully unleashing hardware computing power. This design avoids the complexity of manually adjusting hardware instructions in traditional operator development while ensuring high-performance computing.

-   Toolchain Design

    PyPTO provides complete toolchain support, including:

    -   Compilation intermediate visualization: Supports saving intermediates (computation graphs) at different stages of compilation (such as Tensor Graph, Tile Graph, Block Graph, Execute Graph, etc.) for debugging and analysis.
    -   Runtime performance analysis: Collects and visualizes runtime performance data (swimlane graphs) to help identify performance bottlenecks.
    -   Compilation and scheduling control: Developers can control compilation pass execution and scheduling behavior through the toolchain for deep customization.

    Through the above design philosophy, PyPTO achieves efficient collaboration between algorithm development and operator development, significantly reducing the complexity of operator development while retaining high-performance computing capabilities.

## Supported Products

PyPTO is supported on the following products:

-   Atlas A3 Training Series / Atlas A3 Inference Series
-   Atlas A2 Training Series / Atlas A2 Inference Series

