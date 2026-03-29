# CODEGEN Component Error Codes

- **Range**: F6XXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the CODEGEN component.
---

## Error Code Definitions

The unified definitions for related error codes can be found in the `framework/src/codegen/utils/codegen_error.h` file.

---

## Troubleshooting Recommendations

### General Troubleshooting Steps

When encountering CodeGen component validation errors, or when the generated kernel code does not meet expectations, follow these steps to collect and analyze logs:

1. **Set log level to INFO**
   - Set the log output path:
   export ASCEND_PROCESS_LOG_PATH=*{user-specified log path}*
   - Set the log level to global INFO level:
   export ASCEND_GLOBAL_LOG_LEVEL=1 // 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
   Or specify the CodeGen module log level to INFO, for example:
   export ASCEND_MODULE_LOG_LEVEL=CODEGEN=1
2. **Set parallel compilation count to 1**
   Since the CodeGen module saves compilation time by compiling multiple subgraphs in parallel, to prevent log output disorder during problem diagnosis, parallel compilation must be changed to serial. The method is as follows:
   - Modify `parallel_compile` to 1 in tile_fwk_config.json
   - Recompile and install the pypto package

   ```bash
   cd pypto_project_path && python3 build_ci.py -f python3 --disable_auto_execute
   pip install build_out/pypto*.whl --force --no-deps
   cd -
   ```

3. **Re-run the test case and obtain log and kernel code files**
   The log path is generally: *{user-specified log path}*/debug/plog/pypto-log***.log
   The kernel code file path is generally: search for the kernel_aicore folder under the pypto project path or test framework execution path; the TENSOR***.cpp files inside are the kernel code files.

4. **Analyze logs**
   - For FRAMEWORK (F60XXX) and OPERATION_ADAPTER (F61XXX) type errors, they are generally caused by upstream data anomalies and need to be analyzed in conjunction with PASS logs.
   - Other types of errors need to be analyzed in conjunction with context.
<br>


### Scenario Examples

Note: All scenario log analyses are based on logs obtained through the above troubleshooting steps.

#### A TileOP Call Parameter in Generated Kernel Code Does Not Meet Expectations

1. Find the TileOp call that does not meet expectations in the kernel code, for example:

   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```

2. Use the above TileOp call code as a keyword to search in the logs.
3. After finding the log entry, search upward for the first occurrence of the "Op CodeGenNPU Start" keyword — this is the starting position of the TileOp generation. From there, check the log information sequentially to see if it meets expectations.
4. If you suspect it is related to data passed in from PASS, search for the "Gen OP IS" keyword after the "Op CodeGenNPU Start" keyword. This contains the dump information for that Operation. An example is as follows:

   ```c++
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32 / sym_3_dim_0 x sym_3_dim_1 x sym_3_dim_2 x sym_3_dim_3 x DT_FP32> %152@5#(0)MEM_UB::MEM_UB = !10010 TILE_ADD(g:0, s:-1) %3@3#(0)MEM_UB::MEM_UB, %4@4#(0)MEM_UB::MEM_UB #IS_CUBE{0} #last_use{[0, 1, 1]}
   ```

   Here !10010 is the unique identifier for this operator. Use it as a keyword to search in PASS graphs or logs for related information. For PASS troubleshooting guidance, see [pass trouble shooting](./pass.md)
<br>


#### Error Code F62014: SYMBOL_NOT_FOUND

This error code indicates that the kernel code calls an undefined variable. Common error scenarios are as follows:

##### Operation Missing need_alloc Attribute

The log context for this type of error scenario will contain the "UNDEFINED_VAR" keyword.
CodeGen needs to rely on the need_alloc attribute in the Operation to generate variable definition statements. If this attribute is missing, it will cause the variable definition statement to be missing, resulting in an error.
You can follow the steps in [A TileOP Call Parameter in Generated Kernel Code Does Not Meet Expectations](#a-tileop-call-parameter-in-generated-kernel-code-does-not-meet-expectations) above to find the Operation with the missing attribute and continue troubleshooting with PASS.
<br>

#### Error Code F63001: COMPILE_CODE_FAILED

Kernel code compilation errors can have various causes. Troubleshooting guidance will be improved for different scenarios going forward.

##### Scenario 1: Stack Overflow

Error keyword example:

```log
error: stack frame size (*****) exceeds limit (32768) in function
```

Refer to: [Operator compilation stack overflow error](../tutorials/appendix/faq.md#算子编译报堆栈溢出错误)

##### Scenario 2: PTO Instruction Data Type Mismatch

Error keyword "maybe need a type", example:

```log
/usr/local/Ascend/cann-9.0.0/include/pto/npu/a5/TStore.hpp:233:41: error: the 2nd parameter maybe need a type 'cc float *'
copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
```

Possible causes of data type mismatch:
- Incorrect parameter passing in frontend calls to the Operation interface, refer to: [PTO-related errors in executed code](https://gitcode.com/cann/pypto/issues/705)
- Using a data type not supported by PTO-ISA; re-analyze the use case scenario and use hardware-supported data types.

##### Scenario 3: Generated PTO Instructions Do Not Match the Hardware Platform Specified in Binary Compilation Parameters

Error keyword "does not support the given target feature", example:

```log
error: function type 'void (__cbuf__ void *, __gm__ void *, unsigned char, unsigned short, unsigned short, unsigned short, unsigned short, unsigned int) noexcept' of 'copy_gm_to_cbuf' does not support the given target feature
    copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
    ^
```

Possible causes for this type of error:
- The kernel code compilation parameters are for Vector, but the generated kernel code contains Cube-related instructions, or vice versa — compilation parameters are for Cube but the generated kernel code contains Vector-related instructions, causing the bisheng compiler to report an error.
  This type of problem is generally caused by PASS mixing Vector and Cube Operations into the same subgraph before the CodeGen stage. PASS needs to further analyze the subgraph partitioning logic. The CodeGen stage must see independent, pure Vector or pure Cube subgraphs.
- CodeGen's basis for using Cube or Vector compilation parameters is the `Function::IsCube()` interface. PASS must confirm whether this interface has been set to the correct value for different subgraphs.


##### Scenario 4: Undefined Variables

1. Error keyword contains sym_***:

```log
output/output_20260317_102613_935544_121641/kernel_aicore/TENSOR_loop_0_Unroll1_PATH0_hiddenfunc1_9_416851834981923603_3_aiv.cpp:16:70: error: use of undeclared identifier 'sym_209_dim_0'; did you mean 'sym_65_dim_0'?
UBTileTensorBF16Dim2_1 ubTensor_1((uint64_t)UB_S0_E512_T, (Shape2Dim(sym_209_dim_0, sym_209_dim_1)));
```

Such variables are used to dynamically obtain Shape and Offset sizes at runtime; the data source is the `Function::GetDynParamTable` interface. In the error log, search upward for the first occurrence of the "subprogram id" keyword, find the subgraph ID, and inform PASS to continue analyzing the cause of the missing variable.
<br>


#### Binary Compilation Duration Statistics

The time spent in the CodeGen module can be observed from the statistics provided by the Compiler Monitor in the screen output after executing the operator. An example is as follows:

```log
[Compiler Monitor] Stage: CodeGen(completed) | Stage elapsed: 1.2s | Total elapsed: 1.2s
[Compiler Monitor] Compilation finished 6/6 | Total functions: 6
[Compiler Monitor] Stage timing (aggregated by stage):
  CodeGen  1.2s   (sum over 6 functions)
  Pass     0.0s   (sum over 6 functions)
  Prepare  0.0s
[Compiler Monitor] Monitoring stopped | Total elapsed: 1.2s
```

Since the current CodeGen time is mainly binary compilation, the binary compilation duration at the Top Function granularity is specifically recorded in the INFO-level logs of the CodeGen module. You can grep for the keyword "Top Function magic:". This log records the total binary compilation duration of all subgraphs within that Top Function (all subgraphs are compiled in parallel via make). An example is as follows:

```log
[INFO ] PYPTO(726656):2026-03-19 10:59:35.426 [codegen_cloudnpu.cpp:760][CODEGEN]:Top Function magic: 8, hash: 16874966534923480783: Starting parallel compilation: 128 jobs, 1 tasks
[INFO ] PYPTO(726656):2026-03-19 10:59:36.135 [codegen_cloudnpu.cpp:768][CODEGEN]:Top Function magic: 8, hash: 16874966534923480783: Parallel compilation finished in 709.613831 ms

```
The log records the number of concurrent processes executing the bisheng command to compile binaries for all subgraphs within that Top Function, as well as the total elapsed time.

- Method for confirming individual kernel file compilation duration:
  1. Find the kernel_aicore folder and the kernel code file Tensor**.cpp to be verified under the pypto project path or test framework path, for example:
     {prefix path}/output/output_20260319_145742_163710_1702013_6466B4B5/**kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp**
  2. Open the kernel code file, go to the bottom to find the bisheng command for compiling this file and copy it, for example:

  ```bash
  bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
  ```
  3. cd {prefix path}
  Confirm the current directory is one level above the output folder.
  4. Execute the copied bisheng command and confirm it runs successfully. If the bisheng command is not found, refer to:
  [prepare_environment](../../.agents/skills/pypto-environment-setup/references/prepare_environment.md) "CANN Environment Loading (General Template)" section.
  5. Use system tools such as `time`, `perf`, or other shell commands combined with the bisheng command to measure duration, for example:

  ```bash
  time bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
  ```
