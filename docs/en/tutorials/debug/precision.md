# Precision Debug

## Introduction

When a PyPTO operator executes without functional warnings or errors but the output data does not meet expectations, you can use the following methods to narrow down and locate precision issues. Precision issues mainly arise from two sources:

-   Functional errors: Obvious data errors or deviations caused by hardware silent faults, software silent functional issues, or formula implementation errors.
-   Computation errors: Obvious data deviations caused by differences in data types, algorithms (tiling, accumulation, formula approximation), and so on.

## Overall Workflow

![](../figures/zh-cn_image_0000002532402113.png)

## Confirming Problem Validity

1.  Confirm whether the problem determination method is reasonable (based on experience and experimental validation).
    1.  Check whether the error threshold is reasonable. For example:
        -   An operator containing low-precision data types such as bfloat16 uses the error threshold for float32.
        -   An operator with a deep computation path uses the error threshold for a small operator.

    2.  Correct the unreasonable parts based on experience. If the problem disappears, precision debugging is complete.

2.  Confirm that the problem is stably reproducible.
    1.  Execute multiple rounds and the output data is consistent and shows abnormal values.
    2.  Switch to other environments and execute multiple rounds; the output is still consistent and shows abnormal values.
    3.  If stable reproduction is not possible, identify the issue as a functional problem and suspend precision debugging.

## Basic Pre-Check

The basic pre-check is a guidance description pointing out common but easily overlooked high-probability error items. If users or debug personnel have confirmed that the corresponding check items are correct, these steps can be skipped.

1.  Use the asys tool to check for hardware issues.
    1.  Use a hardware self-check tool to rule out hardware installation problems.
    2.  Use a hardware stress test tool to rule out hardware failures.

2.  Check for software issues.
    1.  Confirm that the software version is correct based on the installation guide.
    2.  Run the example test cases in the project and confirm the results are correct.

3.  Check whether problems were introduced on the user side.
    1.  Review the operator code with multiple parties and confirm:
        -   The computation process is consistent with the algorithm prototype.
        -   The data types and computation types are consistent with the reference implementation (if no reference implementation is available, the designer who provided the operator implementation plan must confirm the types).

    2.  If confirmation is not possible, when analyzing problem points found subsequently, additionally analyze whether the issue was introduced on the user side.

## Avoiding Known Issues

Before precision debugging, ensure that known issues in the current software have been avoided. For details, refer to [Known Issues](../appendix/issue.md).

## Reducing Problem Scope

Reducing the problem scope is usually an optional step aimed at simplifying the problem and improving the efficiency of reproduction and localization.

-   After reducing the problem scope, the same problem must be reproducible, then proceed with subsequent tool self-checks or manual debugging.
-   For cases where new problems appear after reduction, it is recommended to try other reduction methods to reproduce the original problem. New problems should not be included in the key localization process.

However, in some cases, reducing the problem scope is a required step, such as for larger models:

-   Insufficient host memory prevents the self-check tool from executing.
-   Insufficient file storage space prevents the self-check tool from saving intermediate computation data.
-   Other analysis processes or tools exceed the subjectively tolerable time limit, or blocking situations such as inability to execute.

Typically, the problem scope is reduced by the following methods:

-   Reduce the number and size of subgraphs. For example, reduce the number of loop iterations or reduce the number of cube/vector tiling blocks (i.e., increase the TileShape size).
-   Trim the model. For example, reduce the model shape specifications, such as batch\_size and seq\_len.
-   Use binary search to remove tail computations.
    1.  Following the model's computation order, use binary search to remove computations closer to the tail, and add the disconnected outputs to the operator's output list.
    2.  Execute the operator and observe and analyze the new output list:
        -   If the data is normal (no inf/nan, no subjectively random values, or small deviation from reference baseline data), return to the previous step to continue binary search.
        -   If the data is abnormal, restore the code to the state before the current removal, and treat it as the latest candidate problem scenario.
        -   If the trimmed model is already quite small, stop the binary search and select the latest candidate problem scenario for subsequent localization.

## Tool Self-Check and Analysis

### Tool Overview

PyPTO has a complete intermediate representation at each pass stage of computation graph compilation, which can be translated into third-party computation code and used to simulate computation on other computation units (such as the Host CPU). By comparing simulated computation results with baseline data, the tool can detect whether there is an anomaly in the operator or in a pass's processing result, and locate the first computation node where the anomaly appears.

Key features and use cases:

-   Tensor Graph verification: Used to verify the correctness of operator code and frontend processing. Based on baseline (golden) input/output data provided by the user, it compares the final result from Tensor Graph simulation to detect the correctness of the overall computation. Common use cases:
    -   When the user has available operator baseline (golden) input and output data, the coarse-check feature can first be enabled to roughly rule out whether operator code or frontend processing introduced differences.

-   Pass stage verification: Used to self-check the correctness of passes. Based on the simulated computation results of each pass, it compares results to detect correctness and abnormal computation nodes in passes. Common use cases:
    -   When the user's operator precision has just started showing issues and there is no clear direction, enable the self-check feature first to rule out whether the pass processing stage introduced potential errors.
    -   When the user has roughly identified that a certain pass has a problem, enable the self-check feature to obtain simulated computation intermediate data for that pass and prior passes, and compare the data to find potential problematic computation operations.

-   Intermediate result analysis: Specify a single computation result and save it to a file or print it in a readable form to output/log.
    -   When Tensor Graph verification fails, use the pass_verify_print/pass_verify_save feature to print and save simulated computation intermediate data, compare the data to find potential problematic computation operations.


### Usage Constraints

The current precision debug tool has the following limitations (complete computation flow representation is only saved in the pass runtime context), which prevent use of the detection functionality:

-   Does not support on-board execution intermediate data inspection; only frontend and pass inspection is supported.
-   Does not support collective communication scenarios.
-   Does not support specific passes. Certain passes (such as SubgraphToFunction) are intermediate optimization processes that lack complete computation information; the tool handles these with automatic skipping.
-   Does not support automatic comparison verification between passes (manual data comparison is required).
-   Does not support constructing and simulating computation in an arbitrary runtime environment after the program exits. Construction and simulation must be performed on the corresponding host CPU and process during operator compilation.
-   Does not support constructing and simulating computation using Ascend C calls based on the Ascend AI processor.
-   Does not support constructing and simulating computation based on GPU.
-   Does not support verification of computations containing GATHER_IN_UB and GATHER_IN_L1 operations.
-   If a B200BU error appears in ExpandFunction verification results, the verification result for that scenario is only valid after InferDynShape.

### Environment Preparation

The latest master branch code and versions after 0.1.1 (excluding version 0.1.1) support online compilation of the C++ binaries required by the precision tool at runtime, without needing to recompile and reinstall PyPTO. However, you must confirm that the build tools required for online compilation meet the following requirements:

    - cmake >= 3.16.3
    - make
    - g++ >= 9.4.0

Earlier PyPTO source code requires recompiling and installing PyPTO before the tool can be used.

1.  Confirm that GCC is installed and upgrade to version 9.4.0 or higher.
2.  Recompile and install PyPTO from source. The main difference is to add the `--no-build-isolation` option to the build and install command; for other operations, refer to [Build and Install](../../install/prepare_environment.md).

    ```bash
    python3 -m pip install . --verbose --no-build-isolation
    ```

### Tool Usage Steps

1. Enable the precision debug switch. A reference example is: [hello_world.py](../../../examples/00_hello_world/hello_world.py).

    ```python
    ...
    verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        ...
    }

    @pypto.frontend.jit(verify_options=verify_options)
    def add_kernel(
        input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = input0 + input1

    ...
    ```

    **verify_options Parameters**

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `enable_pass_verify` | bool | False | Master enable switch that determines whether all `pass_verify_*` options and interfaces take effect. Must be set to `True` for other parameters to take effect |
    | `pass_verify_save_tensor` | bool | False | Whether to save simulated computation data to disk. When set to `True`, a `verify_*` directory is generated in the `{work_path}/output/output_*/` directory |
    | `pass_verify_save_tensor_dir` | str | "{RUNNING_DIR}/output/output_{TS}" | Save path for detection results and data. An absolute path can be specified |
    | `pass_verify_pass_filter` | List[str] | empty | Configures the list of pass names to self-check. If not specified, specific passes are verified by default; specify `"all"` to verify all passes; specify `[]` to skip pass verification and only verify tensor_graph |
    | `pass_verify_error_tol` | List[float] | [1e-3, 1e-3] | Tolerance configuration for precision comparison. The first value is the relative error tolerance (rtol), the second is the absolute error tolerance (atol) |

2. Set golden data (optional)

    If tensor_graph verification is required, set golden data:

    ```python
    ...
    def test_add():
        shape = (1, 16, 1, 64)
        input_data0 = torch.rand(shape, dtype=torch.float)
        input_data1 = torch.rand(shape, dtype=torch.float)
        torch_add = torch.add(input_data0, input_data1)
        # Set golden data
        pypto.set_verify_golden_data(goldens=[None, None, torch_add])

        input_data0 = input_data0.to('npu')
        input_data1 = input_data1.to('npu')
        out = torch.empty(shape, dtype=torch.float, device='npu')

        add(input_data0, input_data1, out)
    ...
    ```

    **set_verify_golden_data Interface**

    **Function Prototype**:

    ```python
    set_verify_golden_data(in_out_tensors=None, goldens=None)
    ```

    **Parameters**:

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `in_out_tensors` | List[Union(pypto.Tensor, torch.Tensor)] | Sets the actual input and output list used when executing the operator to the detection tool, with positions corresponding to each other. Under jit call mode, this option does not need to be set |
    | `goldens` | List[Union(pypto.Tensor, torch.Tensor)] | Sets the user's available computation baseline (golden) output data into the tool for comparison detection. This list has the same length as and positionally corresponds to the operator's input and output parameter list. Setting a position to None skips data comparison at that position. **Note: The `device` attribute of `torch.Tensor` must be CPU; NPU is not supported** |

    **Constraints**:
    - This function takes effect only after `pypto.set_verify_options(enable_pass_verify=True)` is set

3.  Run the modified test case.

    ```bash
    python3 examples/00_hello_world/hello_world.py
    ```

4.  Output similar to the following is printed, indicating whether the corresponding self-check result is PASS, FAIL\(ED\), or skipped (NO\_COMPARE):

    ```text
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 0 result NO_COMPARE
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 1 result NO_COMPARE
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 2 result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_00_RemoveRedundantReshape Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_01_AutoCast Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_02_InferMemoryConflict Verify result PASS
    ...
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_34_InsertSync Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_35_MixSubgraphSplit Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_36_CodegenPreproc Verify result PASS
    ```

5.  After execution, a verify\_\* directory is generated in the $\{work\_path\}/output/output\_\*/ directory (\* represents a timestamp) to store detection result files.

    ```text
    ├── tensor_graph # stores intermediate data from simulated computation of the frontend initial computation graph, used as baseline data
    │   ├── *.data
    │   └── ...
    ├── verify_graph_data_metainfo.csv # result report, stores metadata of intermediate data and corresponding data file names
    ├── Pass_{PASS_SEQ}_{PASS_NAME} # stores intermediate data from simulated computation of intermediate pass computation graphs, used as test data
    │   ├── *.data
    │   └── ...
    ```

6.  Recommended follow-up actions.

    For cases where FAIL is marked in tensor_graph verification results, it is recommended to:

    1.  Review the correctness of the PyPTO frontend code from multiple angles.
    2.  When the frontend code has no obvious anomalies, use `pass_verify_print` and `pass_verify_save` to save/print intermediate results for further analysis (see step 7 for details).

    For cases where tensor_graph verification passes but FAIL is marked in pass stage verification results, it is recommended to:
    1.  Collect the relevant result information and submit an ISSUE for handling.

7. Use `pass_verify_print` and `pass_verify_save` to analyze intermediate results (optional).

    **Use case**: When Tensor Graph verification fails, you can use these two interfaces to print and save simulated computation intermediate data, and compare the data to find potential problematic computation operations.

    **Important notes**:
    - `pass_verify_print` and `pass_verify_save` save the **results from simulated computation during the tensor graph verification phase**
    - These results are obtained by simulating execution of the computation graph on the host CPU
    - **Results may differ from actual NPU on-board execution results; they are mainly used for algorithm logic verification**

    **Usage example**:

    ```python
    @pypto.frontend.jit(verify_options=verify_options)
    def add_kernel(
        input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        # Save intermediate result to file
        pypto.pass_verify_save(input1, "input1_by_pass_verify")
        # Print intermediate result to console
        pypto.pass_verify_print(input0)
        out[:] = input0 + input1

    def add(input_data0, input_data1, out):
        add_kernel(input_data0, input_data1, out)

    def test_add():
        shape = (1, 4, 1, 64)
        input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
        input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
        out = torch.empty(shape, dtype=torch.float, device='npu')

        add(input_data0, input_data1, out)
    ...
    ```

    **Run the modified test case**

    ```bash
    python3 examples/00_hello_world/hello_world.py
    ```

    **Console output example**:

    ```text
    input0:<64x64xFP16/64x64xFP16>
    [[0.03955 0.6094 0.1519 ... 0.7339 0.8789 0.8662]
     [0.6284 0.01465 0.6333 ... 0.2422 0.03516 0.8423]
     [0.231 0.02686 0.6055 ... 0.7466 0.2529 0.2231]
     ...
     [0.3477 0.4243 0.05273 ... 0.9287 0.1138 0.5083]
     [0.05273 0.9941 0.4985 ... 0.8345 0.8613 0.188]
     [0.3184 0.8047 0.833 ... 0.7734 0.2578 0.1392]]
    ```

    **Generated file structure**:

    After execution, a `tensor/` directory is generated in the `{work_path}/output/output_*/` directory (* represents a timestamp):

    ```text
    ├── tensor/
    │   ├── input1_by_pass_verify.data     # saved specified simulated computation data, in direct memory dump format of tensor data
    │   ├── input1_by_pass_verify.csv      # metadata for simulated computation data, including data type and shape information
    ```

    **Recommended follow-up data processing**:

    Use the metadata information with common interfaces such as `torch.from_file()` and `numpy.load()` to open the data files and convert them into parseable values. Then apply common data analysis methods used by developers, such as checking the offset patterns of abnormal data and characteristics of abnormal data values (inf/nan/zero, etc.).

## On-Board Execution Tensor Dump

### 1. Description

Supports dumping the input and output data of leaf functions during on-board execution, for precision issue localization. The dump data can be compared and analyzed against frontend simulation computation results.

### 2. How to Enable

```python
import os

# Set environment variable to enable on-board dump, or set the environment variable separately before execution: export PTO_DATADUMP_ENABLE=true
os.environ["PTO_DATADUMP_ENABLE"] = "true"

# Configure verification options
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True
    }
)
def kernel(...):
    ...
```

### 3. Dump Data Output Path

```
output/output_*/dump_tensor_*/device_{deviceId}/
└── {taskId}_{seqNo}_{callopMagic}_{rootHash}_{funcHash}_{rawMagic}_{timeStamp}_{dataType}_{input/output}{index}.tdump
```

### 4. Data Processing Tool

**Tool location:** `tools/verifier/parse_dump_tensors.py`

**Main features:**
- Parse dumped binary data (.tdump files)
- Extract tensor data and save as .data files
- Compare and verify against tensors saved by pass verify (only supports comparison with tensors saved by the codegen pass)
- Merge multiple tensors into a complete raw tensor
- Generate a tensor_info.csv report file

**Usage:**

```bash
# Basic usage (no verification)
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/dump_tensor/device_0

# Usage with verification, with enable_pass_verify enabled and verify_path specified
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/output_*/dump_tensor_*/device_0 \
    --verify_path output/output_*/verify_*/
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dump_tensor_path` | Path to the dump data directory | `output/output_*/dump_tensor_*/device_0` |
| `--verify_path` | Directory containing verify_graph_data_metainfo.csv | `""` (no comparison with verify results) |

**Output files:**

```
output/output_*/dump_tensor_*/device_0/
├── tensor_info.csv              # parsed result report
├── *.data                       # extracted tensor data files
└── raw_{rawMagic}_{dataType}_{ioflag}.data  # merged raw tensor
```

**tensor_info.csv field descriptions:**

| Field | Description |
|-------|-------------|
| headSize | Header size |
| funcId |  |
| taskId | Task ID |
| callopMagic | Operation magic |
| coreId | Core ID |
| dataType | Data type (numeric) |
| dataTypeStr | Data type (string) |
| rawMagic | Raw tensor magic |
| dims | Number of dimensions |
| exeStart | Execution start time; related data recorded only when profiling is enabled |
| exeEnd | Execution end time; related data recorded only when profiling is enabled |
| exeDuration | Execution duration; related data generated only when profiling is enabled |
| rootHash | Root function hash |
| funcHash | Function hash |
| timeStamp | Timestamp |
| shape | Tensor shape |
| offset | Offset within the raw tensor |
| rawShape | Raw tensor shape |
| tensorAddr | Tensor address |
| ioflag | Input/output flag |
| seqNo | Sequence number |
| bin_file | Data file path |
| verify_tensor_file | Verification data file path (if verification is enabled) |
| cmp_res | Comparison result (True/False/"NO_CMP") |
