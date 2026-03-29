# pypto.set\_pass\_options

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Modifies Pass optimization parameter information. Its primary function is to dynamically modify runtime parameter configurations for specific optimization strategies and individual passes during the compilation process, enabling fine-grained control and debugging.

## Function Prototype

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     sg_set_scope: Optional[int] = None,
                     )
```

## Parameters

| Parameter               | Input/Output | Description                                                                 |
|-------------------------|--------------|-----------------------------------------------------------------------------|
| vec_nbuffer_setting     | Input        | Meaning: Graph fusion parameter used to configure the merge count of AIV subgraphs with identical structure. <br> Note: This parameter applies to merging AIV subgraphs with the same structure. <br> Type: dict[int, int] <br> Values:<br> {-1: 1}: Skip AIV subgraph merging <br> {} (empty dict): Automatic merging; merge granularity is computed automatically based on the number of AIV cores<br> {-1: N, 0: N2, ...}: Manual merging; default granularity is N <br> Default: {} empty dict <br> Affected Pass: NBufferMerge |
| cube_l1_reuse_setting   | Input        | Meaning: Graph fusion parameter used to configure the merge count for subgraphs that repeatedly transfer the same GM data. <br> Note: This parameter applies to merging subgraphs that contain CUBE computations. <br> Type: dict[int, int] <br> Values:<br> {-1: 1}: Skip L1Reuse merging <br> {} (empty dict): Automatic merging; merge granularity is computed automatically based on the number of AIC cores<br> {-1: N, 0: N1, ...}: Manual merging; default merge granularity is N. <br> Default: {} empty dict <br> Affected Pass: L1ReuseMerge |
| cube_nbuffer_setting    | Input        | Meaning: Graph fusion parameter used to configure the merge count of AIC subgraphs with identical structure. <br> Note: This parameter applies to merging AIC subgraphs with the same structure. <br> Type: dict[int, int] <br> Values:<br> {-1: 1}: Skip AIC subgraph merging <br> {} (empty dict): Automatic merging; merge granularity is computed automatically based on the number of AIC cores<br> {-1: N, 0: N1, ...}: Manual merging; default merge granularity is N <br> Default: {-1: 1} <br> Affected Pass: L1ReuseMerge |
| sg_set_scope            | Input        | Meaning: Manual graph fusion parameter. <br> Note: Assigns a specific scopeId to an operation. If adjacent operations share the same non-(-1) scopeId, they will be forcibly merged into a single subgraph, and that subgraph will not be merged with other subgraphs. This parameter only takes effect for operations that have a direct upstream-downstream connection path — for example, when the output of operation A is used as the input of operation B. <br> Type: int <br> Value range: -1~2147483647 <br> Default: -1 <br> Affected Pass: GraphPartition <br> Configuration recommendations: 1) View-type operations and their corresponding compute-type operations should be assigned the same scopeId. 2) Reshape operations are special — in some scenarios they form their own subgraph, and manual graph fusion control may not take effect. |

## Return Value

None.

## Constraints

- Timing: There is no requirement to call this before graph compilation begins; it may be called at any time.
- Type safety: The type of the value passed in must exactly match the type defined for the parameter; otherwise undefined behavior or runtime errors may occur.
- Scope: Parameter settings are local and only affect the compilation process within the current jit or loop. If not set, the value is inherited from the enclosing scope.

## Example

```python
   pypto.set_pass_options(
                       vec_nbuffer_setting={},
                       cube_l1_reuse_setting={},
                       cube_nbuffer_setting={})
```

### dict Type Configuration Reference
#### Key-Value Semantics
Key (hashorder): Isomorphic subgraph group ID.<br>
- Value M: Matches the specific subgraph group whose hashorder is M.<br>
- Value -1: Matches all subgraph groups not explicitly specified.<br>

Value (N): Represents the merge granularity. That is, every N subgraphs within an isomorphic subgraph group are merged into one new subgraph for execution.<br>
#### Configuration Behavior
When processing subgraph merges, the pass follows the logic "exact match > default config > automatic handling":<br>
- Exact match: If hashorder hits a specific key in the dictionary, merge according to its corresponding value N.<br>
- Default config: If no exact match is found but -1 exists in the dictionary, merge according to the value associated with -1.<br>
- Automatic handling: If neither an exact match nor a -1 entry is found, the merge granularity is computed automatically.<br>
#### Configuration Examples
| Configuration         | Description                                                                 |
|---------------------- |-----------------------------------------------------------------------------|
|{-1: 1}|Skip subgraph merging. Merge granularity is 1, meaning subgraphs within all isomorphic subgraph groups are not merged.|
|{0: 5}|For the isomorphic subgraph group with hashorder 0, every 5 subgraphs are merged into one;<br>for all other isomorphic subgraph groups, the merge granularity is computed automatically based on the hardware core count.|
|{0: 5, 2: 8, -1: 2}    |For hashorder 0: every 5 subgraphs merged into one;<br>for hashorder 2: every 8 subgraphs merged into one;<br>all other isomorphic subgraph groups use the default merge granularity from -1, i.e., every 2 subgraphs merged into one.<br> |
|{0: 5, -1: 1}    |For hashorder 0: every 5 subgraphs merged into one;<br>all other isomorphic subgraph groups are left unchanged. |
