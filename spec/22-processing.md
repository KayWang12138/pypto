# 22. Processing

This section describes key processing flows that the framework applies to IR. The content corresponds to `pictures/ir_transform.svg`.

---

## 22.1 ControlFlowFunction Stage

The frontend constructs initial ControlFlowFunction instances.

At this stage, the computation graph corresponding to ControlFlowFunction is referred to as a **tensor graph**.

### 22.1.1 Loop Unroll

For loops with dynamic iteration counts, the framework performs loop unrolling to improve runtime performance. Based on user-specified unroll factors, the framework expands the loop into multiple sibling loops, ordered from largest to smallest unroll factor.

### 22.1.2 Control Flow Optimization

Control flow is optimized to generate as large `statement.block` regions as possible, enabling subsequent passes to perform broader static optimizations. Control flow optimization techniques include loop fusion, branch hoisting, and similar transformations.

---

## 22.2 DataFlowFunction Stage

After control flow optimization, the framework generates a DataFlowFunction for each `statement.block` in the ControlFlowFunction for subsequent static optimizations. All operations within the `statement.block` are removed and replaced with a call operation that invokes the newly generated DataFlowFunction. Static optimizations on DataFlowFunctions can be executed in parallel.

### 22.2.1 Tiling

Initial DataFlowFunctions operate on tensor-grained data. Therefore, tensors are first partitioned according to user-specified tile size parameters, splitting them into multiple Tile objects. Operations that produce or consume these tensors are correspondingly replicated.

At this stage, the computation graph corresponding to DataFlowFunction is referred to as a **tile graph**.

### 22.2.2 Graph Partition

A large tile graph is partitioned into multiple smaller tile graphs, which we call **block graphs**. These block graphs can be viewed as nodes that form a DAG layer.

The partitioning algorithm can be configured with different parameters to balance intra-core buffer utilization, subgraph isomorphism, and inter-core parallelism.

### 22.2.3 Block Deduplicating

Many block graphs produced in the previous stage are isomorphic. Therefore, deduplication is performed. Deduplicated block graphs are converted into KernelFunctions. An ExecuteFunction is also generated, containing call operations to invoke the corresponding KernelFunctions.

---

## 22.3 KernelFunction Stage

### 22.3.1 Instruction Scheduling

Operations within KernelFunctions are scheduled to maximize buffer reuse, improve intra-core pipe parallelism, and insert synchronization instructions between operations to ensure execution correctness.

### 22.3.2 Codegen

KernelFunctions are generated into hardware-specific instruction files for compiling executable units that run on AICore.

ControlFlowFunctions are generated into corresponding `ctrl_flow.cpp` files, and the compiled executables run on AICPU.

---
