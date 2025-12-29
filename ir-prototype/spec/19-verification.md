# 19. Verification(TODO)

This section defines the verification rules and well-formedness constraints that all valid PTO-IR programs must satisfy.

---

## 19.1 Verification Overview

Verification ensures that PTO-IR programs are:

* **Well-formed**: Structurally correct
* **Type-safe**: Types are consistent
* **SSA-correct**: SSA rules are followed
* **Memory-safe**: No memory safety violations
* **Semantically valid**: Operations are valid

---

## 19.2 Structural Verification

### 19.2.1 Program Structure

* **Single root**: Exactly one `program.module`
* **Entry point**: At least one `program.entry`
* **Function references**: All referenced functions must exist
* **No cycles**: No recursive module nesting

---

### 19.2.2 Function Structure

* **Entry block**: Exactly one entry block
* **Reachability**: All blocks must be reachable
* **Termination**: All blocks must have terminators
* **Block arguments**: Block arguments must match incoming branches

---

### 19.2.3 Operation Structure

* **Operand count**: Operands match operation requirements
* **Result count**: Results match operation requirements
* **Region structure**: Regions are well-formed
* **Attribute validity**: Attributes are valid for operation

---

## 19.3 SSA Verification

### 19.3.1 Dominance

* **Definition before use**: All values must be defined before use
* **Dominance property**: Definitions must dominate uses
* **Phi nodes**: Block arguments (phi nodes) must have definitions from all predecessors

---

### 19.3.2 Value Usage

* **Single definition**: Each SSA value defined exactly once
* **Valid uses**: All uses reference valid definitions
* **Type consistency**: Uses match definition types

---

## 19.4 Type Verification

### 19.4.1 Type Consistency

* **Operand types**: Operand types must match operation requirements
* **Result types**: Result types must be valid
* **Type compatibility**: Types must be compatible for operations

---

### 19.4.2 Shape Verification

* **Shape compatibility**: Shapes must be compatible for operations
* **Symbolic dimensions**: Symbolic dimensions must be bound
* **Dynamic dimensions**: Dynamic dimensions must be resolvable

---

### 19.4.3 Layout Verification

* **Layout compatibility**: Layouts must be compatible
* **Format validity**: Tile formats must be valid
* **Memory space validity**: Memory spaces must be supported

---

## 19.5 Memory Safety Verification

### 19.5.1 Lifetime

* **No use-after-free**: Values not used after deallocation
* **No double-free**: Memory not freed twice
* **Proper pairing**: Allocations paired with deallocations

---

### 19.5.2 Bounds Checking

* **Index bounds**: Memory accesses within bounds
* **Array bounds**: Array indices valid
* **Buffer size**: Accesses respect buffer sizes

---

## 19.6 Control Flow Verification

### 19.6.1 Structured Control Flow

* **Valid conditions**: Condition types must be boolean
* **Region structure**: Regions properly structured
* **Yield values**: Yield values match region requirements

---

### 19.6.2 CFG Verification

* **Terminator presence**: All blocks have terminators
* **Branch targets**: Branch targets must exist
* **Argument matching**: Branch arguments match block arguments

---

## 19.7 Dialect-Specific Verification

### 19.7.1 Tensor Dialect

* **Shape compatibility**: Shapes compatible for operations
* **Broadcasting validity**: Broadcasting rules satisfied
* **Layout consistency**: Layouts consistent

---

### 19.7.2 Tile Dialect

* **Tile size constraints**: Tile sizes fit hardware
* **Format compatibility**: Formats compatible
* **Memory space validity**: Memory spaces valid

---

### 19.7.3 Statement Dialect

* **Loop bounds**: Lower bound < upper bound
* **Parallel safety**: Parallel loops have no dependencies
* **Memory safety**: No memory safety violations

---

### 19.7.4 Block Graph Dialect

* **Unique identity**: `(color, key)` pair must be unique per block.
* **Dependency completeness**: Every RAW/WAR/WAW relationship is represented via `blockgraph.dep`.
* **Binding coverage**: All inputs/outputs/tiles have bindings when instantiated.
* **Side-effect isolation**: No global side effects beyond declared results.

---

### 19.7.5 Pipe Execution Dialect

* **Pipe coverage**: Every block graph op assigned to exactly one pipe.
* **Event correctness**: Each event has one producer and at least one consumer.
* **Buffer lifetime**: `pipe.exec.buffer` scopes enclose all uses; spills target valid memory spaces.
* **Synchronization**: `pipe.exec.wait`/`pipe.exec.barrier` satisfy dependency constraints (RAW/WAR/WAW) inherited from block graph metadata.

---

### 19.7.6 Execution Dialect

* **Acyclic graph**: `exec.graph` is acyclic unless edges are explicitly marked streaming.
* **Dependency scope**: `exec.dependency` references nodes within the same graph.
* **Resource lifecycle**: `exec.allocate` buffers are deallocated or marked persistent.
* **Hint validity**: Scheduling hints (core groups, priorities) match platform capabilities.

---

### 19.7.7 Distributed Dialect

* **Worker count**: Collective ops have matching participants
* **Sharding compatibility**: Sharding strategies compatible
* **Remote memory**: Remote memory references valid

---

### 19.7.8 Configuration Dialect

* **Type correctness**: Configuration values match expected types
* **Value validity**: Values within valid ranges (e.g., tile sizes > 0)
* **Platform compatibility**: Configuration compatible with target platform
* **Resource constraints**: Resource limits satisfiable
* **Conflict resolution**: Conflicting configurations resolved per inheritance rules

---

### 19.7.9 Platform Abstractions Dialect

* **Type correctness**: Platform properties match expected types
* **Value validity**: Values within valid ranges (sizes > 0, latencies >= 0)
* **Consistency**: Platform model internally consistent
* **Connectivity**: Interconnects connect valid components
* **Hierarchy validity**: Memory hierarchy well-formed

---

## 19.8 Verification Passes

### 19.8.1 Compile-Time Verification

Verification performed:

* **After parsing**: Verify structure after deserialization
* **After transformations**: Verify after each pass
* **Before codegen**: Final verification before code generation

---

### 19.8.2 Verification Levels

* **Strict**: All checks enabled (development)
* **Standard**: Essential checks (default)
* **Relaxed**: Minimal checks (performance-critical paths)

---

## 19.9 Error Reporting

### 19.9.1 Error Messages

Verification errors include:

* **Location**: Source location (if available)
* **Operation**: Operation that failed verification
* **Reason**: Explanation of failure
* **Suggestion**: How to fix (if applicable)

---

### 19.9.2 Error Categories

* **Structural errors**: Malformed IR structure
* **Type errors**: Type inconsistencies
* **SSA errors**: SSA violations
* **Memory errors**: Memory safety violations
* **Semantic errors**: Invalid operations

---

## 19.10 Summary

Verification ensures:

* **Correctness**: Programs are well-formed
* **Safety**: No memory or type safety violations
* **Consistency**: IR is internally consistent
* **Reliability**: Programs can be safely executed

It is a critical component of the PTO-IR system.

---

