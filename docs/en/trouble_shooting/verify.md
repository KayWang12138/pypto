# VERIFY Component Error Codes


- **Range**: FBXXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the VERIFY component.
---

## Error Code Definitions

The unified definitions for related error codes can be found in the `framework/src/interface/interpreter/verify_error.h` and `framework/src/interface/interpreter/calculator/calc_error.h` files.

## Troubleshooting Recommendations

### Typical Error Code Scenarios
The error codes involved in this scenario are part of the precision verification tool component's functionality. When these error codes appear, the precision verification tool is still functioning normally. Users can use the following troubleshooting recommendations to independently locate precision issues:

#### Error Code 0xB4001U: VERIFY_RESULT_MISMATCH
##### Log Example
This scenario is a precision verification failure scenario, accompanied by output such as the following:
```log
[ERROR] PYPTO(1535746):2026-03-20 09:43:12.887 [flow_verifier.cpp:105][VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED
[INFO ] PYPTO(1535746):2026-03-20 09:43:12.887 [flow_verifier.h:151][VERIFY]:
    Error rtol=0.001000 atol=0.001000 index:0 golden:10 output:1 absDiff:9 relDiff:1.63636
    Error rtol=0.001000 atol=0.001000 index:1 golden:10 output:1 absDiff:9 relDiff:1.63636
    Error rtol=0.001000 atol=0.001000 index:2 golden:10 output:1 absDiff:9 relDiff:1.63636
    Error rtol=0.001000 atol=0.001000 index:3 golden:10 output:1 absDiff:9 relDiff:1.63636
    Error rtol=0.001000 atol=0.001000 index:4 golden:10 output:1 absDiff:9 relDiff:1.63636
    Error rtol=0.001000 atol=0.001000 index:5 golden:10 output:1 absDiff:9 relDiff:1.63636
  All size:256 failNum:256 maxAbsDiff:9 maxRelDiff:1.63636 averageAbsDiff:9 averageRelDiff:1.63636 errorCount:256 errorRatio:1 zeroCount:0 zeroRatio:0
  maxAbs-> index:0 golden:10 output:1 absDiff:9 relDiff:1.63636
  maxRel-> index:0 golden:10 output:1 absDiff:9 relDiff:1.63636
```
First, use entries like `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` to determine the stage where the precision error occurred. In the example, the error stage is `tensor_graph`, meaning the precision error occurred before entering the pass.
##### Location Guidance

###### pass_verify_print and pass_verify_save
For precision issues at the `tensor graph` stage, perform manual data analysis.
The precision verification tool provides `pypto.pass_verify_print` and `pypto.pass_verify_save` to support users in printing or saving the computation results of tensors in their written pypto kernel functions. (Note: The tensor can be either the final output or an intermediate tensor, but the printed results are not the computation results on the NPU. They are simulation results based on the precision tool's simulated execution and the user's frontend expression.)
Before enabling precision location, first confirm that the final output dumped by the precision verification tool is consistent with the NPU computation results.
For details, see [pass_verify_print interface example](docs/api/others/pypto-pass_verify_print.md) and
[pass_verify_save interface example](docs/api/others/pypto-pass_verify_save.md).

###### Precision Verification Tool Skill
For precision issues at the `tensor graph` stage, use an AI agent for precision location.
When an operator's precision is incorrect but the specific problem location is unknown, you can invoke the pypto-binary-search-verify skill. Simply send a clear instruction to the assistant such as "The operator test_my_op.py failed precision verification, please help me use the operator precision problem search skill to locate where the problem is," or directly specify "Use the pypto-binary-search-verify skill to locate the precision issue in test_my_op.py." This will automatically insert checkpoints, generate data files based on test results, analyze results using a comparison script, and identify the problematic operator.
###### Precision Verification Tool Automated Comparison Script
For precision issues occurring during the pass execution stage, use automated scripts for location.
Script path: `tools/verifier/pass_compare.py`
When a precision comparison fails for a certain pass, you can use the `pass_compare.py` script to compare the precision of that failing pass with the preceding passes. The comparison will generate a result file such as `verify_pass@SplitK@ExpandFunction@1773821696834386.csv` in the directory where the precision tool dumps data. This file records the result of each operator node comparison between the failing pass and the preceding passes. Nodes that could not be matched are also recorded and marked as skip. This allows locating the first matched node that resulted in an error.
Script usage: `python3 pass_compare.py --p ExpandFunction RemoveUndrivenView --verify_path=.....`
The `--p` parameter is followed by the two passes to compare, separated by a space. The first is the pass where precision comparison failed, and the second is the pass serving as the golden reference. The `--verify_path` parameter is the absolute path to the directory where the precision tool dumps data files.
#### Error Code 0xB200FU: RUNTIME_EXCEPTION
##### Log Example
This scenario is an operation simulation execution failure, often caused by incorrect attributes on the operation. A log example is as follows:
```log
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[0] tensorMagic=94, shape=[82816, 512], offset=[0, 0], dynValidShape=[82816, 512], dynOffset=[]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[1] tensorMagic=3717, shape=[1, 16], offset=[0, 1792], dynValidShape=[RUNTIME_GetViewValidShapeDim(RUNTIME_GetInputShapeDim(ARG_topk_indices,0),((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx),1), RUNTIME_GetViewValidShapeDim(2048,((s2_idx*2048)+1792),16)], dynOffset=[((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx), ((s2_idx*2048)+1792)]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[2] tensorMagic=12209, shape=[1, 512], offset=[0, 0], dynValidShape=[RUNTIME_GetViewValidShapeDim(RUNTIME_GetInputShapeDim(ARG_block_table,0),bIdx,1), 512], dynOffset=[bIdx, 0]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) output[0] tensorMagic=3716, shape=[16, 512], offset=[0, 0], dynValidShape=[sym_3716_dim_0, sym_3716_dim_1], dynOffset=[]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.116 [flow_verifier.cpp:299][VERIFY]:ErrCode: FB200F! VerifyPass failed for function TENSOR_LOOP_L4_s2_SA_LoopUnroll1_Unroll1_PATH0_hiddenfunc0_20, pass InferParamIndex (passIndex: 28, captureIndex: 4): 10447510811372117949, 3415564916459561148, 10564, GATHER_IN_UBOpError
func: TENSOR_LOOP_L4_s2_SA_LoopUnroll1_Unroll1_PATH0_hiddenfunc0_leaf22
filename: /data/l00504208/g00955608/pypto/models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py
lineno: 136
/* /data/l00504208/g00955608/pypto/models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py:136 */
<16 x 512 x DT_INT8 / sym_3716_dim_0 x sym_3716_dim_1 x DT_INT8> %3716@3264#(22)MEM_UB::MEM_UB = !10564 TILE_GATHER_IN_UB(g:22, s:-1) %94@106#(1)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %3717@100(0, 1792)(((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx), ((s2_idx*2048)+1792))#(22)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %12209@102(bIdx, 0)#(22)MEM_DEVICE_DDR::MEM_DEVICE_DDR #IS_CUBE{0} #block_size{128}
<16x512xINT8/0x512xINT8> = GATHER_IN_UB <82816x512xINT8/82816x512xINT8>, <1x16xINT32/1x16xINT32>, <1x512xINT32/1x512xINT32>
out must have shape [topk_count, hidden_dim]
```
#### Location Guidance
The core error is:
`<16x512xINT8/0x512xINT8> = GATHER_IN_UB <82816x512xINT8/82816x512xINT8>, <1x16xINT32/1x16xINT32>, <1x512xINT32/1x512xINT32>`
  `out must have shape [topk_count, hidden_dim]`
  The first line is a brief summary of the failing `operation`. In the example, `<16x512xINT8/0x512xINT8>` is the output's `<shape/validshape>`, and to the right of the equals sign is this operation's `opcode` and all inputs' `<shape/validshape>`.
  In the example, `torch cpp` throws the error message `out must have shape [topk_count, hidden_dim]`. Combined with the line above, we can initially determine that the error originates from this `operation`'s output having an empty `validshape`.
  For further information to narrow down the location, refer to the more detailed error several lines above, which includes information about the `operation`'s input/output `tensors` and the `operation`'s `IR`.

### Non-Typical Error Code Scenarios
#### Error Code 0xB0001U: VERIFY_NOT_ENABLE
Check that the local `torch >= 2.1.0`.
#### Other Error Codes
For other types of error codes, they are often caused by internal defects in pypto. If encountered, please contact the developers in the community for resolution.
