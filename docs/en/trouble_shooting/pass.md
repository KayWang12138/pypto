# PASS Component Error Codes

(To be supplemented)

- **Range**: F4-F5XXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the PASS component.
- When adding error codes, you may annotate the **Associated Skill** (link to the corresponding skill under [.opencode/skills](../../.opencode/skills) for loading during troubleshooting).


## Error Code Definitions and Usage

The unified definitions for related error codes can be found in the `framework\src\passes\pass_utils\pass_error.h` file.
---

### Error Content Passed from Frontend
Frontend user self-check

**Tensor-related errors**:
   1. TENSOR_NULL_POINTER
   ## Description: Tensor or its associated operation has a null pointer reference
   Behavior:
   - Tensor's producer is null
   - Tensor's consumer is null
   - Operation's input tensor is null
   - Operation's output tensor is null
   - There is a null consumer among the tensor's consumers
   - There is a null producer among the tensor's producers

   2. TENSOR_INVALID_MEMORY_TYPE
   ## Description: Tensor's memory type configuration is invalid or mismatched

   3. TENSOR_SUBGRAPH_BOUNDARY
   ## Description: Tensors used across subgraphs are not correctly marked as boundaries
   Behavior:
   - DDR tensor is not marked as subgraph boundary
   - Cross-subgraph tensor is not marked as subgraph boundary
   - Tensor's subgraph id is NOT_IN_SUBGRAPH

   4. TENSOR_SHAPE_MISMATCH
   ## Description: Tensor's shape configuration does not match the operation semantics
   Behavior:
   - Input/output tensor shape or memType of a specific operator does not comply

   5. TENSOR_UNSUPPORTED_DATATYPE
   ## Description: Tensor's data type is not supported by the operation
   Behavior:
   - Operator and input/output tensor supported data types do not match

   6. TENSOR_MEMORY_ALLOCATION
   ## Description: Tensor's memory allocation configuration is invalid


   7. TENSOR_DYNAMIC_ATTR
   ## Description: Dynamic shape-related attributes are missing or incorrectly configured
   Behavior:
   - Operator's dynamic-related attributes are missing
   - Tensor's dynValidShape is empty

   8. TENSOR_MEMORY_CORRUPTION
   ## Description: Tensor memory is corrupted, memory state is invalid, or memory data integrity is abnormal, preventing normal participation in computation
   Behavior:
   - Detected invalid tensor memory pointer, out-of-bounds memory, released memory, or tampered data

**Operation-related errors**:
   1. OP_INVALID_OPERAND_COUNT
   ## Description: The number of operands for the operation does not match the expectation

   2. OP_NULL_POINTER
   ## Description: The operation or its attributes have a null pointer reference
   Behavior:
   - Operation is null
   - Operation's op attribute is null
   - Operation's IOperands or OOperands are null

   3. OP_INVALID_OPCODE
   ## Description: The operation's opcode is invalid in the current context
   Behavior:
   - Operator is non-compliant


   4. OP_PRODUCER_CONSUMER
   ## Description: The input/output dependency relationship of the operation is incomplete
   Behavior:
   - Operator has no producer or consumer

   5. OP_SPECIAL_CONSTRAINT
   ## Description: A special operation violates specific constraints
   Behavior:
   - Producer/consumer operator types of a specific operator are non-compliant
   - The `to memType` type of a specific operator is non-compliant

   6. OP_NESTING_DEPTH
   ## Description: The nesting depth of a specific operation exceeds the limit
   Behavior:
   - The nesting depth of a specific operator exceeds the limit

   7. OP_SEQUENCE_ERROR
   ## Description: Disallowed operation combinations exist in the operation sequence
   Behavior:
   - Disallowed operators or operator combinations exist

 **Function-related errors**:
   1. FUNCTION_GRAPH_STRUCTURE
   ## Description: The graph structure of the Function is incomplete or invalid
   Behavior:
   - There is a null operation in the Function
   - The Function's incast is empty
   - The Function's outcast is empty
   - There is a circular dependency in the Function
   - Subgraph topology structure is incorrect
   - Subgraph ID is out of range
   - Empty subgraph exists

   2. FUNCTION_BOUNDARY_COMPLETENESS
   ## Description: The input/output boundaries of the Function are incomplete
   Behavior:
   - Incast has no consumer
   - Outcast has no producer
   - Operation's subgraphID is negative and is not a NOP operation

   3. FUNCTION_GRAPH_CONNECTION
   ## Description: The graph connection relationships of the Function are incorrect
   Behavior:
   - Input/output graph mismatch
   - Subgraph boundary tensor is not correctly marked
   - Edge index exceeds operations_ size
   - Operation's magic number cannot be found

   4. FUNCTION_EXPAND_FEATURE
   ## Description: The state of the Function's expand feature is incorrect
   Behavior:
   - ExpandFunctionAccelerate flag has not been reset to false
   - Locally defined temporary tensor is used as operation input (has no producer)

   5. FUNCTION_MEMORY_REACHABILITY
   ## Description: Memory type conversion in the Function is unreachable
   Behavior:
   - Input/output memory types of a specific operator are unreachable
   - Input/output memory type conversion path does not exist

   6. FUNCTION_UNIQUENESS
   Description: Duplicate identifiers exist in the Function
   Behavior:
   - Duplicate magic numbers for Operations
   - Duplicate magic numbers for Tensors

   7. FUNCTION_SPECIAL_STRUCTURE
   ## Description: Special structural issues exist in the Function

**Graph-related errors**:
   1. GRAPH_LOOP_DETECTION
   ## Description: Circular dependencies exist in the graph
   Behavior:
   - OperationLoopCheck failed, circular dependency exists
   - LoopCheck failed, cycle exists

   2. GRAPH_TOPOLOGY_STRUCTURE
   ## Description: The topological structure of the graph is incorrect
   Behavior:
   - Subgraph topology structure is incorrect
   - Parent-child subgraph ID relationship is incorrect (parent subGraphId should be less than or equal to subGraphId)
   - Edge index exceeds operations_ size

   3. GRAPH_SUBGRAPH_EMPTY
   ## Description: An empty subgraph exists
   Behavior:
   - Subgraph is empty
   - Empty subgraph exists

   4. GRAPH_SUBGRAPH_ID_INVALID
   ## Description: Subgraph ID configuration is invalid
   Behavior:
   - Subgraph ID is negative and is not a NOP operation
   - Subgraph ID exceeds the totalSubGraphNum range

   5. GRAPH_EDGE_CONSISTENCY
   ## Description: Graph edge connections are inconsistent
   Behavior:
   - inEdgeGraph and outEdgeGraph sizes do not match
   - Node position in inGraph_ exceeds outGraph_ range
   - Node is in inGraph_ but cannot be found in outGraph_
   - There are untraversed edges in outEdgeGraph

   6. GRAPH_COLOR_CONSISTENCY
   ## Description: Graph coloring information is inconsistent
   Behavior:
   - colorInGraph_ and colorOutGraph_ consistency check failed
   - colorOutGraph_ and input matching failed
   - Edges between original operations and subgraph operations are missing in colorOutGraph_
   - Edges in colorOutGraph_ have no corresponding edges in outGraph_

   7. GRAPH_READY_STATE
   ## Description: Graph ready state is inconsistent
   Behavior:
   - Ready state is inconsistent in the topology structure
   - readyState does not match the negative predecessor count

   8. GRAPH_AIV_AIC_MIX
   ## Description: Incompatible compute units are mixed in the subgraph
   Behavior:
   - Both AIV and AIC operations coexist in the subgraph
   - Both UB and L0/L1 memory type tensors coexist in the subgraph

**Config-related errors**:
   1. CONFIG_MEMORY_TYPE_REACHABLE
   ## Description: No reachable conversion path exists between memory types
   Behavior:
   - Input/output memory types are unreachable
   - Memory type conversion path does not exist

   2. CONFIG_SUBGRAPH_BOUNDARY
   ## Description: Tensor boundary markers for cross-subgraph tensors are missing
   Behavior:
   - DDR tensor is not marked as subgraph boundary
   - Cross-subgraph tensor is not marked as subgraph boundary

   3. CONFIG_TENSOR_MEMORY_TYPE
   ## Description: Tensor memory type configuration is invalid
   Behavior:
   - Memory types do not match

**Manager-related errors**:
