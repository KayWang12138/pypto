# CODEGEN 组件错误码

- **范围**：F6XXXX
- 本文档说明 CODEGEN 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义

相关错误码的统一定义，参见 `framework/src/codegen/utils/codegen_error.h` 文件。

---

## 排查建议

### 通用排查步骤

遇到CodeGen组件校验报错，或生成的Kernel代码不符合预期，可通过如下步骤进行日志收集并分析：

1. **设置日志级别为DEBUG**
   - 设置日志输出路径
   export ASCEND_PROCESS_LOG_PATH=*{用户指定日志路径}*  
   - 设置日志级别为全局Debug级别
   export ASCEND_GLOBAL_LOG_LEVEL=0 // 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
   或指定CodeGen模块日志级别为Debug，如：
   export ASCEND_MODULE_LOG_LEVEL=CODEGEN=0
2. **设置并行编译数量为1**
   由于CodeGen模块通过并行编译多个子图方式节省编译时长，故为了防止输出日志乱序，定位问题时需要将并行编译改为串行，设置方法如下：
   - 修改tile_fwk_config.json中的parallel_compile为1
   - 重新编译并安装pypto包
   ```bash
   cd pypto_project_path && python3 build_ci.py -f python3 --disable_auto_execute
   pip install build_out/pypto*.whl --force --no-deps
   cd -
   ```
3. **再次执行用例，获取日志及kernel代码文件**
   日志路径一般为：   *{用户指定日志路径}*/debug/plog/pypto-log***.log
   kernel代码文件路径一般为：   pypto工程路径或测试框架执行路径下，搜索kernel_aicore文件夹，文件夹内的TENSOR***.cpp即kernel代码文件。

4. **分析日志**
   - 对于FRAMEWORK（F60XXX）、OPERATION_ADAPTER（F61XXX）类错误，一般为上游数据异常导致，需要结合PASS日志分析
   - 其他类型错误需要结合上下文进行分析



### 场景举例

#### 生成kernel代码中某个TileOP调用参数不符合预期

1. 在kernel代码中找到不符合预期的TileOp调用，例如：
   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```
2. 以上面TileOp调用代码为关键字在日志中进行搜索。
3. 找到日志后往上搜索出现的第一个”Op CodeGenNPU Start”关键字，即该TileOp生成的开始位置，由此往后以此检查日志信息是否符合预期。
4. 若怀疑和PASS传入的数据有关，则可以在”Op CodeGenNPU Start”关键字后搜索“Gen OP IS”关键字，后面包含了该Operation的Dump信息，样例如下：
   ```c++
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32 / sym_3_dim_0 x sym_3_dim_1 x sym_3_dim_2 x sym_3_dim_3 x DT_FP32> %152@5#(0)MEM_UB::MEM_UB = !10010 TILE_ADD(g:0, s:-1) %3@3#(0)MEM_UB::MEM_UB, %4@4#(0)MEM_UB::MEM_UB #IS_CUBE{0} #last_use{[0, 1, 1]}
   ```
   其中!10010即该OP的唯一标识码，可以此为关键字在PASS的图或日志中搜索获取相关信息，PASS定位指导详见[pass trouble shooting](./pass.md)
   
  
#### 错误码：COMPILE_CODE_FAILED(F63001)

kernel代码编译错误可能有多种原因导致，后续将根据不同场景完善排查指导。

##### 堆栈溢出
报错关键字样例：
```log
error: stack frame size (*****) exceeds limit (32768) in function
```
可参考：  [算子编译报堆栈溢出错误](../tutorials/appendix/faq.md#算子编译报堆栈溢出错误)

##### PTO指令数据类型不匹配
报错关键字样例：
```log
/usr/local/Ascend/cann-9.0.0/include/pto/npu/a5/TStore.hpp:233:41: error: the 2nd parameter maybe need a type 'cc float *'
copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
```
数据类型不匹配可能原因有：
- 前端调用Operation接口参数传递错误，参考：[执行代码有pto相关报错](https://gitcode.com/cann/pypto/issues/705)
- 使用了PTO-ISA不支持的数据类型，需要重新分析用例场景，使用硬件支持的数据类型

##### 变量未定义
报错关键字样例：
```log
output/output_20260317_102613_935544_121641/kernel_aicore/TENSOR_loop_0_Unroll1_PATH0_hiddenfunc1_9_416851834981923603_3_aiv.cpp:16:70: error: use of undeclared identifier 'sym_209_dim_0'; did you mean 'sym_65_dim_0'?
UBTileTensorBF16Dim2_1 ubTensor_1((uint64_t)UB_S0_E512_T, (Shape2Dim(sym_209_dim_0, sym_209_dim_1)));
```
