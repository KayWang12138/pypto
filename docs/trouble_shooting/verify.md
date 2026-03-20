# VERIFY 组件错误码


- **范围**：FBXXXX
- 本文档说明 VERIFY 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义

相关错误码的统一定义，参见 ` framework/src/interface/interpreter/verify_error.h`与`framework/src/interface/interpreter/calculator/calc_error.h`文件。

## 排查建议

### 典型错误码场景
该场景涉及的错误码属于精度工具组件功能的一部分，出现这些错误码，精度工具仍然在正常发挥作用，用户可结合一下排查建议自行开展精度问题定位：

#### 错误码：0xB4001U：VERIFY_RESULT_MISMATCH
##### 日志示例
该场景为精度校验失败场景，同时会伴随如下输出：
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
首先根据类似示例中`[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED`确定精度出错的阶段。如示例中，出错阶段为`tensor_graph `，即进入pass前精度出错。
##### 定位手段指导

###### pass_verify_print与pass_verify_save  
精度工具提供`pypto.pass_veirfy_print`与`pypto.pass_verify_save`支持用户将自己编写的pypto kernal函数中的tensor的计算结果打印或者保存下来。（注意：该tensor既可以是最终的输出，也可以是中间产生的tensor，但是打印出来的结果并不是在npu中的计算结果，而是基于精度工具模拟执行和用户前端表达的模拟结果）。  
再开启精度定位前，可先确认精度工具Dump的最终输出与npu计算结果保持一致。
详参见[pass_verify_print接口示例](docs/api/others/pypto-pass_verify_print.md)与
[pass_verify_save接口示例](docs/api/others/pypto-pass_verify_save.md)  。

###### 精度工具skill
###### 精度工具自动比对脚本
脚本路径：`tools/verifier/pass_compare.py`  
当某个pass精度对比失败的时候，可以利用 `pass_compare.py` 这个脚本将该对比失败的pass和前面的pass进行精度对比。对比会在精度工具dump数据的目录生成一个类似 `verify_pass@SplitK@ExpandFunction@1773821696834386.csv` 这样的对比结果文件，里面记录了精度对比失败的pass的每个op节点和前面pass对比的结果，未能匹配上的也会记录在表中标注skip。这样就能定位到匹配上的第一个出错的节点。  
脚本使用方法：`python3 pass_compare.py --p ExpandFunction RemoveUndrivenView --verify_path=.....`
`--p`参数后面的是对比的两个pass，空格隔开，前面的是精度对比失败的pass，后面的是作为golden的pass，`--verify_path`参数是精度工具dump数据文件的那个目录的绝对路径。
#### 错误码：


### 非典型错误码场景
