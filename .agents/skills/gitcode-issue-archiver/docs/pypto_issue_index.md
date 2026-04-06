# PyPTO 问题现象索引

> **用途**: 遇到问题时检索相似案例，了解规避方案
> **更新时间**: 2026-04-03
> **数据来源**: /data/s00454010/issues/archive/pypto_issues/

---

## 统计信息

- **Bug-Report总数**: 431 (含 Issue #600~#650 新增 25 个)
- **已修复**: 195 (问题已解决，无需关注)
  - 其中预期行为: 12 (不是bug，是正常行为)
- **仍需关注**: 236 (closed=195, open=41)

---

## 第一部分：已修复问题（无需关注）

以下 178 个问题已在评论中确认修复或不再复现，无需重点关注。

### 预期行为（不是bug）

- Issue #426 - [distributed] MoE 多轮重入下 signal 等待/清理语义疑似不匹配
- Issue #245 - [GreenLight]当前样例运行是否不支持Ascend 950
- Issue #509 - gdr非整除算子开启unroll_list，运行时间很长
- Issue #624 -  FFN算子不开启unrool_list上，上板执行不到1分钟；开启unrool_list之后，上板执行需要4分钟，性能差异大
- Issue #682 - cube和vector的结果交替assemble，再view报错AssignMemoryType failed since view/assemble has unreachable input-to-output memoryType
- Issue #567 - PASS对Spill场景的支持存在问题
- Issue #681 - PyPTO QAT算子接入整网时重复编译多轮
- Issue #804 - PyPTO 非对称QAT算子接入整网时重复编译2轮
- Issue #569 - 添加搬运语句，影响到后续计算结果
- Issue #574 - Precision Error in Gated Delta Rule backward with Chunksize = 128
- Issue #661 - compare 的结果不符合预期

### A. 编译错误类

- Issue #25 - 存在CV运算融合的场景，pass未能正常完成子图的切分，造成codegen生成错误代码
- Issue #27 - view + assemble的场景, inplace pass推导有误导致精度问题
- Issue #47 - PyPTO实现了FFN算子，编译阶段OoOSchdule pass卡死（几个小时无响应），该相同算子torch运行耗时不到1s
- Issue #50 - PyPTO实现GDR算子，尾块处理添加view语句后出现OoOSchedule Pass 报错
- Issue #51 - PyPTO实现GDR算子，卡在DynAttrToStatic编译过程
- Issue #71 - [Bug-Report|缺陷反馈]:编译时间久，且无法开启unroll_list
- Issue #139 - 手动设置的valid_shape 在pass28时被优化掉
- Issue #146 - gdr算子输入输出bfloat16，OoOSchedule报错
- Issue #177 - 在Schede OoO Pass中CalcBufferSize不合理
- Issue #269 - 编译安装报错，Can't get python3-dev, Python Frontend can't build
- Issue #351 - [codegen_distributed] OP_SHMEM_PUT 的 shapeIndex 配置错误导致 dstStride 计算错误（shmem 含额外 rank 维时）
- Issue #371 - 【green light】Conv Pypto实现过程，在 Pass_20_CommonOperationEliminate 阶段将不来自于同一块L1地址的Tile进行了合并，不符合预期
- Issue #499 - 开启精度工具后，pass PreGraphProcess 出现shape不匹配报错
- Issue #510 -  无cann包的仿真环境上，编译报machine相关的报错，编译失败
- Issue #516 - 【green light】最新主线代码在 g++ 13 的环境下运行 python build_ci.py 有编译报错
- Issue #535 - [green light] reshape接口如果输入shape和new shape大小一致，那么pass会存在问题
- Issue #582 - gdr反向算子实现，增加一行搬运语句，报错OoOSchedule Failed
- Issue #690 - GCC 13+ 下 AnyCastRef 测试用例 dangling-reference 有编译告警
- Issue #709 - pypto.get_pass_options() API 返回11个参数，文档显示只能配置4个参数
- Issue #713 - Conv Pto fp32场景编译拦截报错
- Issue #765 - amin切分后tile块过多会编译报错
- Issue #855 - Conv Pto 输出通道大小为1，有bias场景，编译报错
- Issue #857 - 修复cast的codegen代码
- Issue #959 - 用例test_pg_lightning_indexer_prolog_quant_mxfp8.py开启mix后，算子编译报错
- Issue #963 - TREM/TREMS接口变更，remainder编译报错

### B. 运行时错误类

- Issue #29 - 用PyPTO实现了GroupQueryAttn算子，上板执行的端到端耗时非常长，大约100s，相比torch小算子的端到端耗时不到1s
- Issue #38 - 静态for报错
- Issue #48 - 无cann环境，costmodel_cpu_swimlane.py运行报错
- Issue #57 - pypto.loop，step > stop时，循环执行次数异常
- Issue #72 - 【GDN算子开发】求逆代码优化，引入concat后出现流同步报错
- Issue #82 - models/glm_v4_5/glm_select_experts.py执行报错
- Issue #120 - 整网运行会产生大量文件落盘
- Issue #140 - 非整除场景reshape扩维报错
- Issue #154 - GDR算子非整除场景，上板执行时报错
- Issue #209 - validshape推导靠后导致精度工具校验validshape时报错
- Issue #242 - pypto.tensor接口创建的tensor直接assemble给outcast时，在machine层报错
- Issue #272 - 样例代码执行失败
- Issue #273 - 新前端，动态轴异常，第二次调用仍按第一次 shape 执行
- Issue #338 - 【GreenLight】pypto实现算子时，如果内嵌python函数，报错无法指明具体问题行
- Issue #354 - Round Operation部分case报错尾轴非对齐，且部分bf16 case精度失败
- Issue #374 - 【green light】开启精度调试，调用scatter接口报错
- Issue #395 - [green light] cumsum接口存在报错，排查非python接口问题
- Issue #415 - 【green light】assemble 的 src 和 dst 的 dtype 不同时，没有拦截，导致报错信息不好定位
- Issue #463 - 【green light】elementwise_ops.py的add::test_add_basic样例跑动态模式报错
- Issue #493 -  用例执行成功，但是pto-instr.hpp多处报告警
- Issue #494 - 体验问题：新前端多个返回值写法，用元组返回类型上板执行正确，但是这种写法语法检测飘红，体验不好
- Issue #500 - [GreenLight] 有场景显示括号嵌套层数过高 根据已有报错信息无法解决
- Issue #560 - mla_prolog_quant算子，修改适配为非量化，pto_isa报错
- Issue #612 - PReLU二维输入，报错内存不对齐
- Issue #613 - get_conv_tile_shapes接口报错
- Issue #651 - QAT算子接入训练整网时出现out of memory报错
- Issue #671 -  多个reshape和matmal/vector级联之后，IntraSubgraphAdapter postcheck报错
- Issue #672 - A3环境中出现了mix子图相关的报错，预期不应该出现
- Issue #673 - pypto.where开启精度工具之后出现报错
- Issue #678 - [Bug-Report|缺陷反馈]:用例正常执行，出现了非预期的ERROR日志
- Issue #715 - matmul泛化性测试时出现CopyMode报错
- Issue #720 - 算子配置unroll_list，不同档位计算图hashorder个数不同，nbuffer校验报错
- Issue #726 - scatter算子输入index以及src的参数类型检测报错提示不明确
- Issue #732 - 写算子遇到 DeviceStitchContext::FullCoverDefaultUpdateStitch 报错 Exception Signum[11] Act
- Issue #738 - LReLU在昇腾上bf16会报错
- Issue #759 - 批跑glm算子用例时偶现执行终断
- Issue #760 - 开启mix功能后glm_moe_fusion算子运行报错
- Issue #779 - pypto算子执行精度失败
- Issue #786 - 8节点64卡整网环境运行2000步保存checkpoint之后，aicpu执行异常超时报错
- Issue #787 - Log1p与PReLU用例运行报错
- Issue #795 - GM非确定性累加精度报错。
- Issue #813 - reshape级联assemble场景，执行时报错，Shape size mismatch
- Issue #815 - gatherelement 在执行库上用例报错
- Issue #819 -  开启校验，GraphPartition postcheck failed；关闭校验，用例执行通过无报错，精度正确。
- Issue #827 - 使用set_verify_options开启精度工具，业务执行core dump
- Issue #837 - GM非确定性累加精度报错。
- Issue #844 - glm_attn_fusion多次运行同一个用例结果不一致
- Issue #846 - Remainder的int16数据类型执行错误
- Issue #854 -  用例执行精度正确，但是有多余的ERROR日志打印误报：[MACHINE]:AllocDev failed: size=5239537664
- Issue #865 - 执行 Python ST 有预期外输出
- Issue #884 - 配置控核后，业务执行成功，日志中有PYPTO打印的三级流水拦截报错的ERROR日志
- Issue #916 - ddr+vector算子测试，pregraph报错，执行流程终止
- Issue #925 - 配置stitch_function_max_num=1后，业务执行卡住

### C. 精度问题类

- Issue #108 - 合轴配置精度问题
- Issue #109 - GDR算子配置合轴优化出现精度问题
- Issue #114 - GDR算子打开合轴优化之后，精度异常，当前精度工具尚未定位出来问题点，请分析优化工具
- Issue #172 - 精度工具设置golden后，使用错误数据与golden进行对比（原因为golden数据类型不匹配，希望增加类型校验）
- Issue #174 - 非整除场景精度问题，上板和golden数据第8行误差大
- Issue #223 - gdr算子unroll之后出现精度问题
- Issue #260 - pr引入GDR算子nan精度问题
- Issue #340 - full创建位置导致的精度问题
- Issue #341 - gdr算子非整除场景开发精度问题
- Issue #343 - Reshape with inplace=True generates precision error with a golden
- Issue #355 - Round Operation部分bf16 case精度失败
- Issue #488 - Tensor先reshape再到loop内被使用，有精度问题
- Issue #498 - pypto.reshape 不支持valid_shape推导，可能会导致精度问题
- Issue #645 - 开启合轴后出现精度问题
- Issue #666 -  SplitK precheck校验失败，去掉校验之后精度正确。
- Issue #680 - 使能精度工具，使用set_verify_golden_data对输出进行粗检，goldens值参数使用不同语法方式传入，会影响粗检结果。
- Issue #691 -  多卡通信测试，有一张卡精度失败时，报告结果不对
- Issue #724 - Log1p测试用例精度失败
- Issue #778 - expandexpdif在ascend950某些场景有精度问题
- Issue #799 - UB2L1场景精度错误
- Issue #835 - UB2L1场景精度错误
- Issue #867 - UB2L1场景精度错误
- Issue #897 - IsFinite 算子在类型为 int 且尾轴为奇数情况下存在精度问题
- Issue #902 - ExpandExpDif在使用新前端用法测试时，广播场景下，所有轴设置为动态轴会产生精度误差

### D. 性能问题类

- Issue #175 - 基础的ffn 或 大维度矩阵乘法 对比torch npu、cce实现 性能差距较大
- Issue #430 - 开启合轴优化自动识别，只能部分生效
- Issue #431 - 开启合轴优化，性能劣化
- Issue #784 - deepseekv3.2 mla与mla+ip算子性能劣化
- Issue #927 -  PerfData 内存释放时机调整

### E. 功能缺失类

- Issue #184 - pypto.forntend.jit 针对 loop_unroll 功能暂不支持
- Issue #304 - tensor.py文件中的代码出现错误，缺少一个左括号
- Issue #617 - [Bug-Report]: Tensor类缺少反向运算符实现（__rmul__, __rsub__, __rtruediv__）
- Issue #652 - config.json 中的配置不支持map格式
- Issue #711 - conv算子缺少校验项，当tileHout>1时，需满足tileWout=tileW=wout

### G. 其他问题类

- Issue #1 -  Examples中kennel参数类型不正确
- Issue #2 - 泳道图部分情况无法导出
- Issue #17 - 存在合图场景，controlFlow_host*.cpp文件生成过于复杂的表达式
- Issue #26 - L1Reuse不生效
- Issue #31 - 关于Python前端中pypto.zeros和pypto.full在部分用例中无法正常工作的BUG
- Issue #37 - Assemble相关的两个bug
- Issue #67 - Cannot run tests and examples in debug mode
- Issue #75 - Incorrect CopyToHost call in legacy codes which causes cost model not runnable
- Issue #79 - Eager Mode suffers from high operator scheduling latency
- Issue #94 - “!!! Kernel Launch”打印信息需要删除
- Issue #107 - 用1月13号最新的pypto仓跑的泳道图，点进去计算图的OP中backtrace内容为空，无法定位到对应的是代码前端哪行，影响调试
- Issue #117 - pypto.concat接口没有正确设置输出tensor的valid_shape
- Issue #141 - 对于包含动态轴的tensor进行transpose，会导致validshape信息丢失
- Issue #142 - unsqueeze操作无valid_shape参数
- Issue #185 - Cannot run test_block_call example in ir_builder
- Issue #243 - 【green light】op 之间的依赖关系不符合预期
- Issue #249 - [Bug-Report|缺陷反馈]:新前端loop内变量生命周期问题
- Issue #276 -  license扫描工具的模板文件中license的描述信息不规范，需要修改。
- Issue #342 - [Bug-Report|缺陷反馈]:【green light】 for循环内部的变量算的对着的，外部拿到就还是个初始值
- Issue #384 - [Green light] scatter_update文档写的input支持二维，但是实际只能跑四维
- Issue #457 - pypto.sum会改变vec_tile_shapes
- Issue #522 - Wrong source code location for operations in python test
- Issue #533 - kernel内部定义的tensor如果使用动态轴需要手动取出，否则功能错误
- Issue #589 - 【green light】L0C buffer写死为128KB，未考虑Ascend 950场景
- Issue #597 - shape变大后出现507015错误，VEC instruction error: the ub address out of bounds
- Issue #602 - 比较操作（ge、le）返回的布尔值为极小浮点数而非标准的0.0或1.0
- Issue #605 - BatchMatmul 4D场景，搬出时Reshape走入了UB
- Issue #610 - 7个文件的版权声明不标准，建议修改成附件E列的内容
- Issue #658 -  Conv算子tileK校验和L1 size的计算逻辑不正确，需要修改 
- Issue #663 - DuplicateOp Postcheck校验失败
- Issue #664 - OOO发生了copyIn的spill，经过重连边后，原copyIn无消费者空悬
- Issue #674 - 泳道图的LeafHash出现为0或者与计算图对应不上的情况
- Issue #676 - Compile Stage配置出现segment fault
- Issue #677 - 日志使能ASCEND_WORK_PATH环境变量的时候，日志落盘目录和CANN不一致
- Issue #708 -  pypto-operator-develop-workflow skill 默认使用0卡导致多用户NPU卡冲突
- Issue #712 - 多个kernel之间配置项相互干扰
- Issue #719 - 调用floor算子出现错误
- Issue #737 - 网络中where算子出错
- Issue #740 - A5平台上ExpandExpDif的入参格式错误
- Issue #741 - 多个outcast返回时，reshape后接assemble到其中outcast1，并和其他的view并联，可能会导致断图
- Issue #742 - signbit测例出错
- Issue #768 - Build error on master branch
- Issue #769 - 【green light】基于最新主线跑python用例的泳道图，只采集到了一些FakeTensor节点
- Issue #785 - 打开precheck校验之后，有mergeviewassemble的误报
- Issue #801 - 【GreenLight】采集到的泳道图打开为空
- Issue #812 - matmul_allreduce_add_rmsnorm里面的batch_size非动态轴
- Issue #833 - Conv算子枚举重定义出错
- Issue #834 - cfgcache_size开启后在64卡整网中拉起失败，出现segment fault
- Issue #892 - BatchMatmul st 库上全0问题修复
- Issue #893 - AOT size大于设定的AOT_CODE_POOL_CODE_SIZE而导致的内存越界
- Issue #894 - _dtype_from 函数里的 dtype 的类型注解建议改为 torch.dtype
- Issue #903 - UB2L1数据链路ND2NZ操作BUG
- Issue #908 - 覆盖率处理逻辑在 lcov 高版本如 2.3.2 版本存在问题
- Issue #920 - signbit 在bf16场景下，nan没有对齐竞品
- Issue #953 - Mix切分时报成环错误

---

## 第二部分：仍需关注的问题

以下问题尚未明确修复，可能需要规避或仍在处理中。

**说明**: 
- `[closed]` - 问题已关闭但未明确修复，可能需要规避方案
- `[open]` - 问题仍在处理中

### A. 编译错误类

**Issue 数量**: 26

#### Issue #7 - Ubuntu系统编译失败 [closed]

- **现象**: Distributor ID:	Ubuntu Description:	Ubuntu 22.04 LTS Release:	22.04
- **状态**: 已关闭
- **来源**: Issue #7

#### Issue #62 - 0108最新的代码，在无cann的环境上Debug模式ir_builder_stmt.cpp编译报错( 在有cann的x86和arm上编译正常) [closed]

- **现象**: 0108最新的代码，在无cann的arm环境上Debug模式编译报错( 在有cann的x86和arm环境上编译正常)： **python3 build_ci.py -f=python3 --build_type=Debug** [ 38%] Building CXX object framework...
- **状态**: 已关闭
- **来源**: Issue #62

#### Issue #88 - pypto.tensor操作，替换为pypto.full操作，报构图错误 [open]

- **现象**: pypto.tensor操作，替换为pypto.full操作，报构图错误 定位进展： 定位结果为
- **规避方案**: ![image.png](assets/issue-88/image.png 'image.png') 收到，我来排期。
- **状态**: 仍在处理
- **来源**: Issue #88

#### Issue #104 - pypto.set_pass_options(sg_set_scope=-1)配置项的功能不完善，未能正确按预期合图 [closed]

- **现象**: 进行自定义求逆算子模块开发时，得到的最初泳道图如下： 其中除去中间的求逆串行计算模块，前后的view、concat、assemble操作导致的碎块较快，十分影响性能，因此希望与对应的求逆计算模块合图，消除碎块及其调度开销，但是发现使用pypto.set_pass_options(sg_set_sco...
- **规避方案**: ![image.png](assets/issue-104/image.png 'image.png') 但这样ADDS 0.0操作，一是影响前端代码直观性、二是影响性能、三是发现加了这些0.0，且开启unroll_list后编译构图出现卡住10分钟，随后报segment fault的错误 ![image.png](assets/issue-104/image.png 'image.png') 希...
- **状态**: 已关闭
- **来源**: Issue #104

#### Issue #111 - BUILD_WITH_NEW_CANN编译选项找不到 [closed]

- **现象**: 在pypto的代码仓中找不到BUILD_WITH_NEW_CANN编译选项，但是在device_runner.cpp和load_aicpu_op.cpp中有多出有关该分支的判断。 代码问题，与环境无关 代码问题，不需要复现
- **状态**: 已关闭
- **来源**: Issue #111

#### Issue #127 - PASS 28 InferParamIndex对view操作输出tensor的validshape推导错误 [closed]

- **现象**: Pass28 推导view输出的valid_shape时，计算结果出错。 ```txt  %43@55(RUNTIME_COA_GET_PARAM_OFFSET(2,82,0), RUNTIME_COA_GET_PARAM_OFFSET(2,82,1))#(6)MEM_UB::MEM_UB = !1...
- **状态**: 已关闭
- **来源**: Issue #127

#### Issue #128 - 基于CANN社区版本8.5编译失败 [closed]

- **现象**: cc1plus: error: /home/tsungl4/opt/ascend-toolkit/latest/pkg_inc/runtime: No such file or directory [-Werror=missing-include-dirs] cc1plus: all warning...
- **状态**: 已关闭
- **来源**: Issue #128

#### Issue #133 -  无cann仿真环境，编译失败 /usr/bin/ld: cannot find -lhccl_stubs [closed]

- **现象**: 无cann仿真环境，编译失败 /usr/bin/ld: cannot find -lhccl_stubs 编译命令：python3 build_ci.py --frontend=cpp -u=TestExpandFunctionPass.ExpandFunctionUTest1 报错日志：
- **状态**: 已关闭
- **来源**: Issue #133

#### Issue #134 - Pass_24_ReplaceTensor后，tensor的rawshape异常 [closed]

- **现象**: 方案穿刺时发现精度有异常 后续看泳道图时，发现有tensor的rawshape有问题 该部分前端代码为
- **状态**: 已关闭
- **来源**: Issue #134

#### Issue #152 - 用1/15最新代码仓跑GDR算子，编译构图卡死不动 [closed]

- **现象**: 拉1月15的pypto仓编包，跑GDR算子编译构图卡死不动 运行环境 系统版本：Linux DevServer-BMS-36154e3e 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:29:11 UTC 2023 aarch64 aarch64 aa...
- **状态**: 已关闭
- **来源**: Issue #152

#### Issue #234 - 【green light】参考文档编译安装部分优化建议 [closed]

- **现象**: 目前看参考文档存在3种安装方式，分别为：1. 实际带NPU卡的真实环境 2.仿真环境 3.docker镜像；但是文档中将各类安装方式都放在一起进行说明，不是很清晰，是否针对这三种方式各写一个文档比较好 Ascend 950 无
- **状态**: 已关闭
- **来源**: Issue #234

#### Issue #377 - AIC/编译 pto::TLOAD + 栈上 buffer 在 O1/O2/O3 下触发 error pointer address space cast [closed]

- **现象**: 在 AIC 编译路径中，使用 `pto::TLOAD` 且 L1 tile buffer 为函数内栈上数组（alloca）时，`bisheng` 在 `-O1/-O2/-O3` 下稳定报错： `fatal error: error in backend: ERROR: error pointer a...
- **状态**: 已关闭
- **来源**: Issue #377

#### Issue #379 -  [green light]pass图执行到TuneTileOpSeqForVF会报错，scatter_nd_sub算子结果不符合预期 [closed]

- **现象**: 通过pypto实现scatter_nd_sub算子功能，代码中执行完pypto.scatter_接口后结果是正确的，但代码最终结果不符合预期。 无论是否开启精度工具开关，结果都是 正确结果：
- **状态**: 已关闭
- **来源**: Issue #379

#### Issue #383 - [green light] 开启 enable_pass_verify后，output路径下无dump json数据生成 [closed]

- **现象**: 开启 enable_pass_verify后，执行kernel时，output路径下无dump json数据生成 950 verify_options = {
- **状态**: 已关闭
- **来源**: Issue #383

#### Issue #464 - 【green light】pypto.frontend.jit 设置enable_pass_verify=True报错 [closed]

- **现象**: pypto.frontend.jit 设置enable_pass_verify=True后报错 A2 1. 基础环境准备ok
- **状态**: 已关闭
- **来源**: Issue #464

#### Issue #477 - 在outcast上进行累加操作导致构图成环，报错不清晰 [closed]

- **现象**: 在如下代码中，d_key_gamma 为 outcast ```python for b_idx in pypto.loop(0, B, 1, name="b_loop"):
- **状态**: 已关闭
- **来源**: Issue #477

#### Issue #512 - [green light] 主线代码在910B服务器上编译报错 [closed]

- **现象**: 主线代码分别在两台910B上服务器上编译，一台能成功，另一台编译报错： In member function ‘void std::__new_allocator::deallocate(_Tp*, size_type) [with _Tp = npu::tile_fwk::SymbolicScal...
- **状态**: 已关闭
- **来源**: Issue #512

#### Issue #548 - [green light] 打开verify，并且设置golden后，各个pass verify结果都是通过，但是实际输出是错误的 [closed]

- **现象**: 打开verify，并且设置golden后，各个pass verify结果都是通过，但是实际输出是错误的 950 执行脚本，必现
- **状态**: 已关闭
- **来源**: Issue #548

#### Issue #572 - mla_prolog改进融合算子，报ooschedule错误 [closed]

- **现象**: mla_prolog改进融合算子，报ooschedule错误，报错日志如下 c113edf6a8e44989bbbcca7f0251dc50.log 910B2
- **根因**: [@KaiqiJin](https://gitcode.com/KaiqiJin) 请关注处理~
- **规避方案**: [@KaiqiJin](https://gitcode.com/KaiqiJin) 请关注处理~
- **状态**: 已关闭
- **来源**: Issue #572

#### Issue #573 - mla_prolog改进融合算子，报编译错误 [open]

- **现象**: mla_prolog改进融合算子，报编译错误 ``` 2026-03-02 23:57:32.590 E | [OoOSchedule][Operation][ERROR]: SpillOnBlock failed at GenBufferSpill.
- **状态**: 仍在处理
- **来源**: Issue #573

#### Issue #595 - 算子在OoOSchedule当中卡死 [closed]

- **现象**: ```python def create_embedding_head_quant_kernel(weight_shape, eps=1e-4, min_v=-128.0, max_v=127.0): @pypto.frontend.jit()
- **状态**: 已关闭
- **来源**: Issue #595

#### Issue #600 - 开启精度工具后，发现expandFunction pass出现精度问题 [closed]

- **现象**: 算子代码如下： ```shell import torch
- **规避方案**: https://gitcode.com/cann/pypto/pull/1898 精度工具定位过程：
- **状态**: 已关闭
- **来源**: Issue #600

#### Issue #623 - mla_prolog算子改进，输入query_start_loc是INT64且涉及动态轴，需要cast为INT32，pass报错 [closed]

- **现象**: mla_prolog算子改进，输入query_start_loc是INT64且涉及动态轴，需要cast为INT32，pass报错 ``` (EngineCore_DP0 pid=1462894) (Worker_TP4_EP4 pid=1462913) ERROR 03-10 17:38:46 [m...
- **状态**: 已关闭
- **来源**: Issue #623

#### Issue #754 - 泛化跑出function GetSortedOperations报错，最终编译在pregraphprocess 这个pass终止，请分析 [closed]

- **现象**: **自定义ddr+vector算子泛化测试，部分用例执行报错，最终编译在pregraphprocess 这个pass终止，请分析：** cycle detected:  %6077@2462#(1)MEM_UB::MEM_UB = !99818 TILE_COPY_IN(g:1, s:-1) %31...
- **状态**: 已关闭
- **来源**: Issue #754

#### Issue #772 - [Bug-Report|缺陷反馈]:Conv算子子图数量太多时会静态编译时间太长，应向用户发出告警提示 [closed]

- **现象**: 当Conv算子的子图数量太多时，会导致静态编译时间过长，应向用户发出告警提示，建议优化模型结构或拆分计算图以降低编译负担。 Device: Ascend 950 CANN: 9.0.0
- **状态**: 已关闭
- **来源**: Issue #772

#### Issue #797 - 【green light】avgpool算子编译时间长，执行时性能很差，ub内反复copyin [closed]

- **现象**: 自己实现的avg_pool池化代码，运行泳道图包含多次copyIn copyout，整体耗时很慢。 我理解input_reshaped是搬到ub的，后面就不需要copyIn了。结果还是按for循环copyin 编译时间有10多分钟：
- **状态**: 已关闭
- **来源**: Issue #797


### B. 运行时错误类

**Issue 数量**: 91

#### Issue #3 - deepseekv32_sparse_flash_attention_quant.py 报错：got an unexpected keyword argument 'combine_axis' [closed]

- **现象**: 执行： python deepseekv32_sparse_flash_attention_quant.py ERROR:root:Record function sparse_flash_attention_quant_d failed: set_operation_config() got an...
- **状态**: 已关闭
- **来源**: Issue #3

#### Issue #6 - 运行obs案例成功后AICore利用率不归零 [closed]

- **现象**: 安装：python3 -m pip install -e . --verbose 执行命令： export TILE_FWK_DEVICE_ID=1
- **状态**: 已关闭
- **来源**: Issue #6

#### Issue #8 - 运行glm_attention.py出现pytorch报错 [closed]

- **现象**: [root@devserver-com glm_v4_5]# python glm_attention.py Traceback (most recent call last): File "/opt/mnt2/pyf/pto/pypto/examples/models/glm_v4_5/glm_a...
- **状态**: 已关闭
- **来源**: Issue #8

#### Issue #24 - Golden 生成脚本存在部分报错，请帮忙看下是否是可以优化； [closed]

- **现象**: 我在尝试自己实现一个 C++的网络，并添加了 `GPT.DecodeLayerTest001` 用例，然后我自己在 `framework/tests/st/` 路径下新增了 `gpt` 路径下存放了我的 C++ 用例代码和 golden 脚本。但是当我运行时，我发现原有 Golden 生成脚本存在一...
- **状态**: 已关闭
- **来源**: Issue #24

#### Issue #39 - c=b[:16,:16]报错 [closed]

- **现象**: 代码如下，使用c=b[:16,:16]报错，使用c=b[0:16,0:16]没问题。 ``` import pypto
- **状态**: 已关闭
- **来源**: Issue #39

#### Issue #49 - PyPTO实现GDR算子，输入输出复用一个tensor，多轮执行出现流同步错误 [closed]

- **现象**: PyPTO实现GDR算子，输入输出复用一个tensor，多轮执行出现流同步错误。 npu: 910B cann: 8.5.0
- **规避方案**: 错误现象如下：通过循环多轮执行该用例，第一轮执行完成并pass，第二次执行报错 ![image.png](assets/issue-49/image.png 'image.png') [@ren-wenqian1](https://gitcode.com/ren-wenqian1) 使用pass临时规避代码 framework/src/interface/operation/op_infer_sh...
- **状态**: 已关闭
- **来源**: Issue #49

#### Issue #52 - 直接使用pip源上发布的pypto whl包，安装之后example用例执行报错module 'pypto' has no attribute 'frontend' [closed]

- **现象**: **直接使用pip源上发布的pypto whl包，安装之后example用例执行报错AttributeError: module 'pypto' has no attribute 'frontend'。** [root@localhost pypto]# pip install pypto Look...
- **状态**: 已关闭
- **来源**: Issue #52

#### Issue #53 - 官方提供的hello_world用例执行报错 [closed]

- **现象**: 1. 已经按照官网的要求进行依赖安装，编译OK,但是执行example里的hello_world用例，报如下错误： 很简单，按照官方的步骤，在910B3环境上能复现 预期官方用例能正常运行
- **状态**: 已关闭
- **来源**: Issue #53

#### Issue #59 - 存在多batch场景，受限于不同stich的顺序执行barrier，算子kernel时间随batch场景线性增长 [open]

- **现象**: 存在多batch场景，受限于不同stich的顺序执行barrier，算子kernel时间随batch场景线性增长 `def chunk_gated_delta_rule(query, key, value, beta, gate, states, mask, tril_mask, eye, act_...
- **状态**: 仍在处理
- **来源**: Issue #59

#### Issue #73 - 【GDN算子开发】求逆代码优化，将向量乘加替换为矩阵乘后出现路径缺失报错 [closed]

- **现象**: 在开发GDN算子过程中，优化下三角矩阵求逆部分代码时，将向量乘加替换为矩阵乘后出现路径缺失报错，提示无法找到从UB到L0C的内存传输路径‌。 npu: 910B cann: 8.3.0.1.200:8.3.RC1
- **状态**: 已关闭
- **来源**: Issue #73

#### Issue #81 - ./01_beginner/basic/basic_ops.py执行报错 [closed]

- **现象**: 执行python3 ./01_beginner/basic/basic_ops.py 报错： python3 ./01_beginner/basic/basic_ops.py ============================================================
- **规避方案**: 执行python3 ./01_beginner/basic/basic_ops.py 报错。 执行python3 ./01_beginner/basic/basic_ops.py不报错。 > >请问把jit改回去的目的是？两种装饰器对应的input/output标记方式略有不同，若确实需要看老的样例可以参考 https://gitcode.com/cann/pypto/pull/33 之前的 [@...
- **状态**: 已关闭
- **来源**: Issue #81

#### Issue #97 - A2环境执行kernelLaunch卡死 [closed]

- **现象**: 根据环境安装后，发现执行算子报错 +------------------------------------------------------------------------------------------------+ | npu-smi 25.3.rc1                ...
- **状态**: 已关闭
- **来源**: Issue #97

#### Issue #99 -  runtime 头文件缺失 [closed]

- **现象**: 无法找到 runtime 下的头文件 910B python3 build_ci.py -u=IRTEST.* -f=cpp --build_type=Debug -j=256
- **根因**: ![image.png](assets/issue-99/image.png 'image.png') [@luohuan40](https://gitcode.com/luohuan40) 你好，麻烦抽空回复下issue中的问题是否已经解决 以及是否还有其他问题~ 当前解决方案有两个 1、临时修改本地的framework/tests/ut/machine/CMakeLists.txt文件，把se...
- **规避方案**: 1、临时修改本地的framework/tests/ut/machine/CMakeLists.txt文件，把set(_Include "include")修改为如下代码 ``` set(_Include "include" ``` set(_Include "include" ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
- **状态**: 已关闭
- **来源**: Issue #99

#### Issue #103 - 运行npu example报错“The MPU address access is invalid” [closed]

- **现象**: 运行命令：python3 hello_world.py --run_mode=npu，报错： 驱动：24.1.rc1 CANN：8.5.0
- **规避方案**: [@DCGDDD](https://gitcode.com/DCGDDD) 有同样问题的issue：https://gitcode.com/cann/pypto/issues/97 你好，已知晓这个问题，正在修复中
- **状态**: 已关闭
- **来源**: Issue #103

#### Issue #105 - 样例运行readme内容错误 [closed]

- **现象**: https://gitcode.com/cann/pypto/blob/master/docs/invocation/examples_invocation.md 圈出来的地方应该是“00_hello_world” /
- **状态**: 已关闭
- **来源**: Issue #105

#### Issue #112 - 合轴优化的force_combine_axis选项开启后报错，开发反馈普通用户不应使用，建议直接从资料里删除，或描述清楚应该使用的场景 [closed]

- **现象**: 合轴优化的force_combine_axis选项开启后报错，开发反馈普通用户不应使用，建议直接从资料里删除，或描述清楚应该使用的场景 npu-smi 25.3.rc1                 Version: 25.3.rc1 910B2
- **规避方案**: 已收到，请@gguuaanngg 处理下，谢谢！
- **状态**: 已关闭
- **来源**: Issue #112

#### Issue #119 - 【PyPTO环境配置】最新PyPTO开源仓环境下，运行代码报错Connection refused [closed]

- **现象**: 下载安装最新PyPTO开源仓并编包后，运行代码会出现报错。尝试了好几个代码均出现此问题。 npu: 910B cann: 8.5.0
- **状态**: 已关闭
- **来源**: Issue #119

#### Issue #125 - 编码流同步报错RuntimeError: ACL stream synchronize failed, error code:507015 [closed]

- **现象**: 写法一流同步报错： 写法二能运行，但有不符合预期的精度问题，且精度异常现象集中 系统版本：Linux DevServer-BMS-36154e3e 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:29:11 UTC 2023 aarch64 aarch...
- **规避方案**: ![image.png](assets/issue-125/image.png 'image.png') ![image.png](assets/issue-125/image.png 'image.png') 写法二能运行，但有不符合预期的精度问题，且精度异常现象集中 ![image.png](assets/issue-125/image.png 'image.png') ![image.png...
- **状态**: 已关闭
- **来源**: Issue #125

#### Issue #193 - GDR算子中runtime_options中stitch设置和memBudget设置对精度的影响 [closed]

- **现象**: GDR算子接入sglang整网时，输入文本序列长度为64k时，"stitch_function_num_initial = 128， stitch_function_inner_memory，stitch_function_outcast_memory设置为32 * 128精度就可以通过。 但当整网...
- **状态**: 已关闭
- **来源**: Issue #193

#### Issue #198 - 执行examples/01_beginner样例采集泳道图失败 [closed]

- **现象**: 执行python3 examples/01_beginner/basic/basic_ops.py combined_operations::test_combined_operations可以成功，此时想采集泳道图数据，并通过PyPTO Toolkit查看泳道图。  但是生成失败。 硬件：910B...
- **状态**: 已关闭
- **来源**: Issue #198

#### Issue #199 - 按教程执行无法运行成功example [closed]

- **现象**: hello world 执行错误 npu-smi info +------------------------------------------------------------------------------------------------+
- **状态**: 已关闭
- **来源**: Issue #199

#### Issue #214 - 【green light】hello_world Example 执行失败 [closed]

- **现象**: 准备好PyPTO运行环境之后执行hello_world Example失败 硬件型号：A2 CANN包等依赖环境：使用教程中 bash tools/prepare_env.sh --type=all --device-type=a2 进行安装
- **状态**: 已关闭
- **来源**: Issue #214

#### Issue #221 - hello_world样例执行失败 [closed]

- **现象**: 根据[环境部署](https://gitcode.com/cann/pypto/blob/master/docs/install/prepare_environment.md)搭建开发环境后，参考[样例执行](https://gitcode.com/cann/pypto/blob/master/do...
- **状态**: 已关闭
- **来源**: Issue #221

#### Issue #222 - 【green light】初级样例test_matrix_multiplication执行报错 [closed]

- **现象**: # 运行所有初级基础操作样例（默认为NPU模式运行） python3 examples/01_beginner/basic/basic_ops.py Ascend950
- **状态**: 已关闭
- **来源**: Issue #222

#### Issue #224 - 【green light】初级样例test_combined_operations执行报错 [closed]

- **现象**: 初级样例test_combined_operations执行报错 Ascend950 python3 examples/01_beginner/basic/basic_ops.py combined_operations::test_combined_operations
- **状态**: 已关闭
- **来源**: Issue #224

#### Issue #229 - [GreenLight]运行时报torch_npu相关错误 [closed]

- **现象**: 运行 examples/01_beginner/basic/basic_ops.py 时，报错缺torch_npu  pip install torch_npu Successfully installed torch_npu-2.2.0后 仍然报错 cannot import name 'FILE...
- **状态**: 已关闭
- **来源**: Issue #229

#### Issue #230 - 在simulator模式下运行 models/deepseek_v32_exp/deepseekv32_lightning_indexer_prolog_quant.py文件，报错卡死 [open]

- **现象**: 打开simulator后： 执行python deepseekv32_lightning_indexer_prolog_quant.py 运行到这里报错卡死：
- **状态**: 仍在处理
- **来源**: Issue #230

#### Issue #235 - 【green light】根据参考文档执行 python3 -m pip install . --verbose 安装报错 [closed]

- **现象**: 根据参考文档执行 python3 -m pip install . --verbose 安装报错 Ascend 950DT python3 -m pip install pypto
- **状态**: 已关闭
- **来源**: Issue #235

#### Issue #236 - Notebook跑入门样例报错 [closed]

- **现象**: 我用Notebook跑examples/hello_world/hello_world.py报错 bash git clone https://gitcode.com/cann/pypto.git
- **状态**: 已关闭
- **来源**: Issue #236

#### Issue #238 - 样例modles/glm_v4_5/glm_attention.py执行报错 [closed]

- **现象**: 执行命令`python3 models/glm_v4_5/glm_attention.py` 报错 910B4 cann8.5
- **状态**: 已关闭
- **来源**: Issue #238

#### Issue #244 - 【green light】执行examples/01_beginner/basic/basic_ops.py报错 [closed]

- **现象**: 执行examples/01_beginner/basic/basic_ops.py报错 Ascend950 python3 examples/01_beginner/basic/basic_ops.py
- **状态**: 已关闭
- **来源**: Issue #244

#### Issue #247 - GDR算子接入整网打开ENABLE_COMPILE_VERBOSE_LOG出现报错 [closed]

- **现象**: 在定位isuue：https://gitcode.com/cann/pypto/issues/193时，当打开debug日志：export ASCEND_GLOBAL_LOG_LEVEL=0，并设置ENABLE_COMPILE_VERBOSE_LOG=1，整网运行会报错： device日志： npu...
- **状态**: 已关闭
- **来源**: Issue #247

#### Issue #248 - 执行示例代码失败，error: no member named 'TROWEXPANDADD' in namespace 'pto' [closed]

- **现象**: 执行任意示例代码，会执行卡死。 报错日志： ```
- **状态**: 已关闭
- **来源**: Issue #248

#### Issue #270 - 运行样例时报错，g++: not found [closed]

- **现象**: 环境准备中没提g++ 8.5.0 python3 hello_world.py
- **状态**: 已关闭
- **来源**: Issue #270

#### Issue #288 - 更新pypto版本至最新，gdr算子运行报错 [closed]

- **现象**: 更新pypto版本至最新，gdr算子运行报错 error.txt npu-smi 25.3.rc1                 Version: 25.3.rc1
- **状态**: 已关闭
- **来源**: Issue #288

#### Issue #293 - 通过ST跑上版实测，只有host侧的数据，无法得到device测的测试数据 [closed]

- **现象**: 分别运行ST用例 python3 build_ci.py -s=DynamicBasicTest.TestDD -f=cpp -d=1 python3 tools/scripts/run_operation_test_with_config.py Add -s=12 -e=12 -d=1
- **状态**: 已关闭
- **来源**: Issue #293

#### Issue #296 - 精度工具在batchmatmul时报错，报错维度信息与实际传入维度不匹配 [closed]

- **现象**: 两个三维tensor进行batchmatmul： tensor.shape = [1, 32, 16] # mat A weight_view = weights[idx] # weights.shape = [4, 16, 16], weight_view.shape = [16, 16]
- **状态**: 已关闭
- **来源**: Issue #296

#### Issue #301 - 【green light】tile切分已设置，但有op [MULS]tile shape not set报错 [closed]

- **现象**: tile切分已设置，但有op [MULS]tile shape not set报错 Ascend950 python3 pypto_scatter_nd_sub.py
- **状态**: 已关闭
- **来源**: Issue #301

#### Issue #336 - aicore使用数量较少，部分task串行执行 [closed]

- **现象**: models/arctic/sum_lstm算子经优化后最佳性能可达19us，但是该性能不能稳定复现，观察泳道图发现，由于aiv core使用数量较少导致部分task串行执行 如下图所示 19us泳道图
- **状态**: 已关闭
- **来源**: Issue #336

#### Issue #339 - 【green light】exception aicore error日志难以定位出错原因 [closed]

- **现象**: 执行pypto自定义算子代码时报错，没有显示具体原因。 [ERROR] Exception Type: exception aicore error 仿真环境
- **根因**: **State**: closed **Author**: rsj007 **Created**: 2026-02-05T09:38:07+08:00 [ERROR] Exception Type: exception aicore error [ERROR] Exception Type: exception aicore error taskid: 5, streamid: 5, tid: 2...
- **状态**: 已关闭
- **来源**: Issue #339

#### Issue #347 - 【green light】pypto.reshape执行自动推导失败，导致aicore error [closed]

- **现象**: target.shape:  [2, 3, 4] shape1 [-1, 4] target_reshaped.shape:  [6, 4]
- **状态**: 已关闭
- **来源**: Issue #347

#### Issue #349 - [Bug-Report|缺陷反馈]:无法运行example [closed]

- **现象**: 无法执行python3 /home/j00911874/pypto/examples/00_hello_world/hello_world.py --run_mode=npu运行example，环境安装和运行方式和readme完全一致 cann8.5.0 A3
- **根因**: 本地执行了下是可以执行的 ![image.png](assets/issue-349/image.png 'image.png') 如果有更详细相关日志可以提供下，这个目录下 (t_pyp) root@qingpu-910c-x86-ip182:~/ascend/log# pwd /root/ascend/log
- **状态**: 已关闭
- **来源**: Issue #349

#### Issue #378 - [green light] 日志报错不清晰 [closed]

- **现象**: ``` 执行脚本后，报错打印不知道是哪一个cast，而且开始compile option，并且output里也没有日志 ============================================================
- **状态**: 已关闭
- **来源**: Issue #378

#### Issue #393 - [green light] pypto执行时 coredump，但是没有具体日志 [closed]

- **现象**: 未执行完毕，直接core dumped。还未device侧执行，并且没有具体日志。 950 未执行完毕，直接core dumped。还未device侧执行
- **状态**: 已关闭
- **来源**: Issue #393

#### Issue #397 - 【green light】models/glm_v4_5/glm_attention_fusion.py 运行失败 [closed]

- **现象**: 运行glm_attention_fusion.py 时报错 [OoOSchedule][Operation][ERROR]: Buffer[L0A/B/C] is Full. Possible causes: incorrect memory reuse, memory fragmentation....
- **状态**: 已关闭
- **来源**: Issue #397

#### Issue #407 - 【green light】携带validShape场景下，执行tstore指令，出现偶然的精度问题 [closed]

- **现象**: 背景： 多轮执行以下逻辑：shmemGetGm2Ub（数据从GM拷贝至UB）后，得到shap为{8, 8}的int32数据类型的ub数据，携带validShape属性，其中，validShape的rowShape为表达式结果（在本场景下值在0~3之间），colShape与rawShape保持一致；对...
- **规避方案**: 910B
- **状态**: 已关闭
- **来源**: Issue #407

#### Issue #460 - 【green light】开启"stitch_cfgcache_size"配置后执行报错 [closed]

- **现象**: 前端配置如下： runtime_options={"stitch_function_num_initial": 128, "stitch_function_outcast_memory": 1024,
- **状态**: 已关闭
- **来源**: Issue #460

#### Issue #461 - 【green light】01_beginner/compute/elementwise_ops.py用例执行失败 [closed]

- **现象**: 01_beginner/compute/elementwise_ops.py用例执行失败 代码中存在缩进问题，跑不起来。 昇腾950
- **状态**: 已关闭
- **来源**: Issue #461

#### Issue #472 - 算子shape大的时候运行时间过长 [closed]

- **现象**: gdr算子配置大shape时，比如512k的时候运行时间过长，长达20分钟多 Device: Ascend 910B CANN: 8.5.0
- **状态**: 已关闭
- **来源**: Issue #472

#### Issue #474 - gdr在新版本pypto超出内存，无法正常运行 [closed]

- **现象**: ``` torch.OutOfMemoryError: NPU out of memory. Tried to allocate 6.69 GiB (NPU 5; 60.96 GiB total capacity; 150.64 MiB already allocated; 150.64 MiB c...
- **状态**: 已关闭
- **来源**: Issue #474

#### Issue #482 - [Bug-Report|缺陷反馈]:【green light】代码执行过程产生大量warning [closed]

- **现象**: 代码执行过程产生大量warning ascend950 ```
- **状态**: 已关闭
- **来源**: Issue #482

#### Issue #485 - pypto.frontend.jit新前端标记的动态轴来自于返回的out_tensor时，报错Dynamic dimension b2 not found in symbolic_dim_value_map [closed]

- **现象**: pypto.frontend.jit新前端标记的动态轴b2来自于返回的out_tensor时，报错: File "/root/miniconda3/envs/py310/lib/python3.10/site-packages/pypto/frontend/parser/entry.py", lin...
- **状态**: 已关闭
- **来源**: Issue #485

#### Issue #487 - 【green light】代码运行过程中出现大量warning [closed]

- **现象**: 拉取今日主线master代码，安装今天的cann包与pto-isa包，跑样例时出现大量warning ascend950与cann9.0.0 cd examples/00_hello_world/
- **状态**: 已关闭
- **来源**: Issue #487

#### Issue #490 - 开启合轴优化之后AICore利用率降低 [closed]

- **现象**: 开启合轴优化：pypto.experimental.set_operation_options(combine_axis=True) 根据泳道图的显示，任务块之间间隔变大，AICore利用率降低 开启之前：
- **状态**: 已关闭
- **来源**: Issue #490

#### Issue #492 - 在2层loop外使用pypto.full(0)创建tensor，在内部对tensor进行累加，出现device报错：“Root incast read from empty address.” [closed]

- **现象**: 示例代码如下 ```python for h_idx in pypto.loop(H, name="h_loop"):
- **规避方案**: ``` if d_key_gamma_tmp is None: d_key_gamma_tmp = pypto.full(0) d_key_gamma_tmp = d_key_gamma_tmp + d_key_gamma_view ```python dgamma_tmp = pypto.full(dgamma.shape, 0.0, dtype=dgamma.dtype) for b_idx ...
- **状态**: 已关闭
- **来源**: Issue #492

#### Issue #495 - 包含两个动态轴的四维tensor调用pypto.sum()，报错“Only one dim can be inferred, func CheckAndInferShape” [open]

- **现象**: 样例如下： ```python import torch
- **状态**: 仍在处理
- **来源**: Issue #495

#### Issue #508 - [GreenLight]运行泛化场景时rtMemSet fail 无法定位 [closed]

- **现象**: 运行泛化测试时 发现场景Shape (4, 16, 2048, 1) 运行卡住  也没有报错 运行四小时之后报错rtMemset failed size=4 rc=507899 Ascend950
- **根因**: 解决方案：调大set_vec_tile_shapes，减少切块数量
- **规避方案**: 感谢反馈问题，麻烦提供下详细的case
- **状态**: 已关闭
- **来源**: Issue #508

#### Issue #529 - 静态轴与动态轴创建输出输出，上板执行结果不一致，且静态轴上板结果与精度工具比较结果不一致 [closed]

- **现象**: 示例代码如下： ```python def kernel(
- **状态**: 已关闭
- **来源**: Issue #529

#### Issue #549 - [green light] 调用x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)将int32数据转换成fp16时报错，报错信息不明确 [closed]

- **现象**: 调用x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)将int32数据转换成fp16时报错，报错信息不明确 Ascend950 kernel内部：
- **状态**: 已关闭
- **来源**: Issue #549

#### Issue #551 - 【green light】在 pypto.loop 里设置 idx_name='_' 报错 Forbid duplicate name of loop idx. It names _ [closed]

- **现象**: 在 Python 前端的 kernel 实现里这样写： ```python for _ in pypto.loop(1, name='CREATE_SHMEM_TENSOR', idx_name='_'):
- **状态**: 已关闭
- **来源**: Issue #551

#### Issue #565 - 用例执行传deviceid=11，报错RuntimeError: Unable to determine device ID: ensure all tensors support DLPack [closed]

- **现象**: **执行用例（传deviceid 11）：** python3 ops/pass_case/pass_infermemory_in_output.py 11 **报错：**
- **状态**: 已关闭
- **来源**: Issue #565

#### Issue #570 - 调大unroll后，上板执行出现ddr越界报错 [closed]

- **现象**: 如下写法不会报错： ```python unroll_list = [128]
- **规避方案**: ```python unroll_list = [128] for b_idx in pypto.loop(0, B, 1): for l_idx, tile in pypto.loop_unroll(0, L, 1, unroll_list=unroll_list): # L = 1024 * 8 ```python unroll_list = [512] for b_idx in pypto....
- **状态**: 已关闭
- **来源**: Issue #570

#### Issue #586 - 运行scatter_update该算子直接报错Segmentation fault (core dumped) [closed]

- **现象**: import pypto import torch import os
- **规避方案**: 下述新前端写法供参考: def update_scatter(a_shape, b_shape, idx_shape, out_shape): def update_scatter(a_shape, b_shape, idx_shape, out_shape): @pypto.frontend.jit() def update_kernel(
- **状态**: 已关闭
- **来源**: Issue #586

#### Issue #603 - 用例执行失败，device侧日志出现Exception Signum[11] [closed]

- **现象**: **device侧日志打印到了All schedule exited, destroy the machine后，后续出现报错Exception Signum[11]， 日志详见**： [INFO] AICPU(2086,aicpu_scheduler):2026-03-05-11:41:10.84...
- **状态**: 已关闭
- **来源**: Issue #603

#### Issue #606 - Ascend950 Conv API 22个多模态网络用例执行时间过长需要优化 [closed]

- **现象**: Ascend950 Conv API 22个多模态网络用例执行时间过长需要优化 Ascend950 无
- **状态**: 已关闭
- **来源**: Issue #606

#### Issue #609 - 重复执行单个算子，两次生成的计算图不一致 [closed]

- **现象**: 重复执行单个算子，两次生成的计算图不一致（tensor id、op magic 等）. 算子代码如下，需要patch https://gitcode.com/cann/pypto/pull/1137 ```python
- **状态**: 已关闭
- **来源**: Issue #609

#### Issue #614 -  用例执行有多余的告警信息：ld.lld: warning: cannot find entry symbol _start; not setting start address [closed]

- **现象**: python前端用例任意执行均有这个warning: ld.lld: warning: cannot find entry symbol _start; not setting start address A3
- **状态**: 已关闭
- **来源**: Issue #614

#### Issue #619 - 静态轴使用python for循环的方式，运行卡住 [closed]

- **现象**: 开发fa反向算子，循环次数block_num是固定值，使用loop循环执行没问题。但是将335行loop循环修改为for循环，执行用例会一直卡住没反应。 ``` for block_i in pypto.loop(block_num, name="LOOP_block_w", idx_name="b...
- **规避方案**: ``` for block_i in pypto.loop(block_num, name="LOOP_block_w", idx_name="block_i_w"): block_idx = block_table[b_idx, idx + block_i] block_idx_valid = block_idx.max(0)
- **状态**: 已关闭
- **来源**: Issue #619

#### Issue #642 - [Bug-Report|缺陷反馈]:【green light】kernel内部对pypto.tensor进行切分时报错 [closed]

- **现象**: ERROR:root:Record function mhc_pre_kernel failed: tensor dtype must be DT_INT32. [Compiler Monitor] Stage: Prepare(completed) | Stashed function: 2 | ...
- **状态**: 已关闭
- **来源**: Issue #642

#### Issue #650 - 【green light】expert_ids[x_active_mask == 0] = -1 报错 AttributeError: 'int' object has no attribute 'shape'，报错信息无法指导用户定位 [closed]

- **现象**: expert_ids 是 shape=[bs, topk] 的 tensor，x_active_mask 是 shape=[bs] 的 tensor，`expert_ids[x_active_mask == 0] = -1` 在 torch 上能正常运行，以下是最小可运行示例： ```python ...
- **状态**: 已关闭
- **来源**: Issue #650

#### Issue #667 - where 的 condition 和 input 的 shape 不能广播时，报错信息不友好 [closed]

- **现象**: 如题 910B 最小可复现代码：
- **状态**: 已关闭
- **来源**: Issue #667

#### Issue #699 - 有些算子对 dtype、shape 等没有拦截，到调用底层指令的时候才报错，对用户来说报错信息很难读懂，而且也看不出具体是哪一行出错，kernel 实现比较复杂时很影响定位 [closed]

- **现象**: 比如，如果 eq 算子输入的 tensor 是不支持的 dtype： ```python import torch
- **状态**: 已关闭
- **来源**: Issue #699

#### Issue #701 - 检测leaf function粒度的内存重叠，开启verbose日志和debug日志后运行用例报错 [closed]

- **现象**: gdr算子检测leaf function粒度的内存重叠，开启verbose日志和debug日志后运行用例报错 用例： 38222ddf1e3a4209a5c3a596b30d5333.zip
- **状态**: 已关闭
- **来源**: Issue #701

#### Issue #731 - https://gitcode.com/cann/pypto/blob/master/docs/invocation/examples_invocation.md快速开始样例报错 [closed]

- **现象**: 快速开始的样例报下面的错误 root@41a3a7a647cf:/workspace/pypto/examples/00_hello_world# python test_pypto.py --run_mode=npu Traceback (most recent call last):
- **规避方案**: 本问题将关闭，如有疑问，欢迎继续讨论
- **状态**: 已关闭
- **来源**: Issue #731

#### Issue #761 - pypto/examples/01_beginner/basicz下执行python3 basic_ops.py [closed]

- **现象**: ============================================================ PyPTO Basic Operations Quick-Start ======================================================...
- **状态**: 已关闭
- **来源**: Issue #761

#### Issue #808 - 搭建模型时会卡死问题 [closed]

- **现象**: 构建了模块，组合起来时会卡死 910B4，cann：8.5.0 ```python
- **根因**: 3. 搭建模型时，会分模块验证，最后拼接起来，但是分模块验证就需要有`@pypto.frontend.jit`装饰器，但是拼接起来的时候有这玩意儿又会报错。所以只能拼接的时候注释掉 4. 分模块时，子模块不能直接return，需要传参获取返回值，能改为直接用return吗 5. 目前入口处传参Tensor似乎必须分开传递，不能放到一个list或者dict里面？那如果实现一个megakernel模型...
- **状态**: 已关闭
- **来源**: Issue #808

#### Issue #842 - where算子存在多处问题，包括报错，精度，工具问题 [closed]

- **现象**: 1.where算子在大shape下报vec内存越界问题 2.where算子在和ge le组合时有精度问题 3.where在开启精度工具是报错
- **状态**: 已关闭
- **来源**: Issue #842

#### Issue #864 - 执行 python ST失败 [closed]

- **现象**: =============================================================================================================================== 4 warnings in 6.56s ==...
- **状态**: 已关闭
- **来源**: Issue #864

#### Issue #866 - Pypto使用控核配置后未生效，并且执行还会偶现卡死 [closed]

- **现象**: 1.使用控核配置aic aiv核数，实际执行观察泳道图中aic aiv使用情况，与配置不一致。配置未生效。 2.配置控核后，多step调用pypto，会偶现卡死现象，进程无法ctrl c退出，只能手动kill -9 进程。 3.配置控核生效后，连续两个step分别配置不同核数执行，会出现全aiv或a...
- **状态**: 已关闭
- **来源**: Issue #866

#### Issue #876 - 直接使用pip源上发布的pypto 包，安装之后example用例执行报错，请整改 [open]

- **现象**: **pip install pypto** Looking in indexes: http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/ Collecting pypto
- **状态**: 仍在处理
- **来源**: Issue #876

#### Issue #881 - gdr反向算子开启合轴优化后报错 [open]

- **现象**: gdr反向算子开启合轴优化后报错， `pypto.experimental.set_operation_options(combine_axis=True)` 报错日志：
- **规避方案**: 报错解决后出现精度问题，定位到是在当前 kernel 中，所有写入 2D tensor 的 `[l,1]` slice 赋值都会受影响： `db_out[bs_ofs:bs_ofs+l, nv_idx:nv_idx+1] = db_c` 建议： 1. 向 PyPTO 框架层报告此 bug：`combine_axis=True` 对 2D output tensor 的 `[l,1]` slice ...
- **状态**: 仍在处理
- **来源**: Issue #881

#### Issue #901 - 使用内存重叠检查工具，执行报错，校验两个没有依赖的task之间有内存重叠 [open]

- **现象**: a3f12a0d31ba48b9bf0bff962a9ad303.txt 使用内存重叠检查工具，执行assert报错 “memory reuse must happen for full match”。 查看报错时两个task之间内存重叠并且是包含关系，查看拓扑关系，两个task无依赖关系。
- **状态**: 仍在处理
- **来源**: Issue #901

#### Issue #922 - 局部Tensor跨loop使用，第一个loop内应当为outcast，但存在标记为incast，引发device侧断言错误 [open]

- **现象**: #  局部tensor跨loop在machine侧报Error ## 背景 算子业务实现，需要声明一个局部tensor，在一个loop内作为一个op的pretoken及Cast的输出，在第二个loop内作为GetTensorData的输入，两个loop需要串行执行；
- **状态**: 仍在处理
- **来源**: Issue #922

#### Issue #940 - LI算子在2k以下case会出现aicore error [open]

- **现象**: 1、错误信息 models/deepseek_v32_exp/deepseekv32_lightning_indexer_quant.py::test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2 ErrorTracking callback in, ta...
- **状态**: 仍在处理
- **来源**: Issue #940

#### Issue #944 - amax在使用新前端写法测试时发生aicore [open]

- **现象**: amax在更换写法后，执行[242,244,65,79]	fp32 [242,244,65,1]	fp32	ND	3	TRUE	[78,48,32,79]	[16,32,1,8]时遇到aicore A3 必现
- **规避方案**: **State**: open **Author**: Baiyi_destroyer **Created**: 2026-03-26T20:16:53+08:00 A3
- **状态**: 仍在处理
- **来源**: Issue #944

#### Issue #946 - greater 用例报错 [open]

- **现象**: rh2pid: -646605 5ppid: 645819 2026-03-26 09:57:34EBGrootgreater:45 5 | greater | Debug: input datarange = [-0.0o1 0.0o1 uniform normal:, :-1 1 uniform...
- **状态**: 仍在处理
- **来源**: Issue #946

#### Issue #947 - 泳道图统计aicore端到端的耗时不对 [open]

- **现象**: 当前泳道图IDE中所呈现的aicore端到端耗时与aicore耗时有偏差 A2,A3,A5 开启泳道图性能数据采集
- **状态**: 仍在处理
- **来源**: Issue #947

#### Issue #971 - 执行测试用例，打开日志为debug级别后，aicpu超时 [open]

- **现象**: 配置ASCEND_GLOBAL_LOG_LEVEL=0（日志级别为debug级），执行用例会必现业务失败报错 A3 Pypto版本  4a128f26cb3d5334ec3fb712e0b8b9b6d0c0feb4
- **状态**: 仍在处理
- **来源**: Issue #971

#### Issue #982 - 相同算子多次launch情况下stitch device侧动态建立依赖的管理数据残留导致后续launch使用了脏数据 [open]

- **现象**: 相同算子多次launch情况下算子二进制复用，上次launch stitch过程 device侧动态建立依赖的管理数据残留在二进制管理内存中，导致后续launch使用了脏数据，后续launch可能引用到脏数据stitch过程触发异常 NA [ERROR] AICPU(20688,aicpu_sche...
- **状态**: 仍在处理
- **来源**: Issue #982

#### Issue #996 - export DUMP_DEVICE_PERF=True下，在多个stitch场景采集的耗时数据有误 [open]

- **现象**: export DUMP_DEVICE_PERF=True时 库上python models/qwen3_next/qwen3_next_gated_delta_rule.py的test_b2_nqk4_nv8_s8k()用例下采集的AICore的End-to-End time为 但泳道图工具中的耗时...
- **状态**: 仍在处理
- **来源**: Issue #996

#### Issue #1006 - 配置infer_controlflow_shape后host侧执行异常 [open]

- **现象**: host有segment fault Pypto 0.2.0分支 CANN 330分支
- **状态**: 仍在处理
- **来源**: Issue #1006


### C. 精度问题类

**Issue 数量**: 33

#### Issue #137 - 加0.0导致的精度异常问题 [closed]

- **现象**: 去掉红框处的加0.0，精度通过 加上后精度异常 golden：
- **状态**: 已关闭
- **来源**: Issue #137

#### Issue #187 - Inplace tensor导致精度问题 [closed]

- **现象**: 情况1：eye_chunk从kernel外面传进来，进行原地修改 ``` pypto.assemble(A_block, [0, 0], eye_chunk)
- **规避方案**: ``` pypto.assemble(A_block, [0, 0], eye_chunk) A_block_inverse_aligned = inverse_pto(eye_chunk, eye, 128) ``` Tmp = pypto.full([L, L], 0.0, pypto.DT_FP32) Eyechunk = pypto.tensor([L, L], pypto.DT_FP32...
- **状态**: 已关闭
- **来源**: Issue #187

#### Issue #197 - [Bug-Report|缺陷反馈]:整网中，gdr算子非整除场景出现精度问题 [closed]

- **现象**: 对于S = 64K, chunk_size=64的场景，gdr算子**本地ST通过**，但使用整网dump下来的tensor作为输入时，会出现nan。 核心代码逻辑如下： 对于尾块，无论是否整除（actual_L <= 64），都会进入if分支。
- **规避方案**: ```txt ================pto vs torch================== npu: 910b cann: 8.5.0 略 [@poursoul](https://gitcode.com/poursoul) 样例中的两个连续的 assemble，现在已经是 ssa assemble 了吗？
- **状态**: 已关闭
- **来源**: Issue #197

#### Issue #292 - gdr算子长序列精度问题 [closed]

- **现象**: 跑gdr算子时，chunk_size为64的场景能通过，但是chunk_size=128存在精度问题，用精度工具测试遇到如下报错，但是不太清楚报错原因 ``` 2026-02-02 10:23:39.450 E | ExecuteOperation error: op RESHAPE (magic=...
- **根因**: ``` 2026-02-02 10:23:39.450 E | ExecuteOperation error: op RESHAPE (magic=10007) input[0] tensorMagic=15, shape=[1, 8, 128, 128], offset=[0, 0, 0, 0], dynValidShape=[1, 8, 128, 128], dynOffset=[] 2026...
- **状态**: 已关闭
- **来源**: Issue #292

#### Issue #316 - pypto.tensor初始化tensor数值不为0 [closed]

- **现象**: gdr算子中使用pypto.tensor初始化tensor后需要手动设置为0，不然在求逆之后精度出现误差 commit id:81814e6a25e2bf6abca105ccf544f75881122c21 patch:
- **状态**: 已关闭
- **来源**: Issue #316

#### Issue #360 - mHC_pre算子精度问题 [closed]

- **现象**: 在使用精度工具时，tensor_grpah验证的精度都pass，但是pass阶段会报错，且最终精度也出错 ``` 2026-02-06 15:34:50.191 E | VerifyPass failed for function TENSOR__LoopUnroll64_Unroll1_PATH0...
- **规避方案**: 使用的是自己的算子，现通过https://gitcode.com/cann/pypto/pull/975 这个pr报错消失。但是仍存在精度问题 b = 1
- **状态**: 已关闭
- **来源**: Issue #360

#### Issue #467 - scatter_update刷新两次时，前端正常表达的结果显示只刷新一次 [open]

- **现象**: scatter_update刷新两次后的结果有误：如果将a刷新成a1，再刷新成a2，将a2搬出去，显示a2只刷新了第二次的部分。 npu-smi 25.3.rc1 Version: 25.3.rc1 | +---------------------------+---------------+---...
- **规避方案**: 下面这种两个out的写法精度正确： ``` out1 = pypto.from_torch(out, dynamic_axis=[]) out2 = pypto.from_torch(out, dynamic_axis=[]) ``` out1 = pypto.from_torch(out, dynamic_axis=[]) out2 = pypto.from_torch(out, dynamic...
- **状态**: 仍在处理
- **来源**: Issue #467

#### Issue #468 -  pypto.frontend.jit新前端写法把output_tensor作为入参不返回时精度正确，把output_tensor不作入参带返回值时精度有误 [closed]

- **现象**: 方式1、使用新前端入参带output_tensor，不带返回，原地修改output_tensor，相同的kernel代码，执行没有精度问题； 注意：output_tensor的shape跟input_tensor_a、input_tensor_b不一样 @pypto.frontend.jit(run...
- **规避方案**: **State**: closed **Author**: lytest_asd **Created**: 2026-02-12T18:26:21+08:00 注意：output_tensor的shape跟input_tensor_a、input_tensor_b不一样 @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.N...
- **状态**: 已关闭
- **来源**: Issue #468

#### Issue #470 - 开启精度工具后日志打印不完全 [closed]

- **现象**: 开启精度工具，日志打印不完全 commit id:81814e6a25e2bf6abca105ccf544f75881122c21 patch:
- **状态**: 已关闭
- **来源**: Issue #470

#### Issue #478 - 精度工具valid_shape出现[2,0] [closed]

- **现象**: 在使用精度工具的时候，遇到 ```  = MUL ,
- **规避方案**: <2x16xFP32/2x2xFP32> = MUL <2x1xFP32/2x0xFP32>, <2x16xFP32/2x2xFP32>  这种表达 从计算图看本身是不合法的，计算结果正确也存在风险，不建议算子这样写
- **状态**: 已关闭
- **来源**: Issue #478

#### Issue #481 - Inplace tensor导致精度问题2 [closed]

- **现象**: 在此跟踪 https://gitcode.com/cann/pypto/issues/187 略 略
- **状态**: 已关闭
- **来源**: Issue #481

#### Issue #520 - 调大输入的shape后，精度工具出现core dumped [closed]

- **现象**: L为序列长度。L =32, 在校验`function_TENSOR_l_loop_LoopUnroll16_Unroll1_PATH0_hiddenfunc0_14`的第七个outcast时存在精度问题 ```shell 2026-02-26 11:40:56.242 V | function_TE...
- **状态**: 已关闭
- **来源**: Issue #520

#### Issue #521 - pypto.ne使用较大vec_tile_shapes结果有问题 [closed]

- **现象**: 代码： ``` def unique_kernel(
- **状态**: 已关闭
- **来源**: Issue #521

#### Issue #532 - 精度工具在tensorgraph阶段的assemble仿真结果异常 [closed]

- **现象**: B, L, H, D = 1, 64, 1, 16 B和L为动态轴 对于临时tensor`d_embeddings_tmp`，其shape为`[1, 64, 1, 16]`，我们通过loop循环往上assemble四个shape为`[1, 16, 1, 16]`的块（由`[16, 16]`的`d_k...
- **状态**: 已关闭
- **来源**: Issue #532

#### Issue #536 - 【green light】算子开发精度问题定位困难，缺失dump中间临时tensor功能 [closed]

- **现象**: 算子开发精度问题定位困难，缺失dump中间临时tensor功能、最终cce代码dump、kernel代码printf功能和打桩修改codegen cpp操作指导等关键功能 Ascend950 无
- **规避方案**: Ascend950 无
- **状态**: 已关闭
- **来源**: Issue #536

#### Issue #539 - 算子精度工具校验通过，但上板结果不符合预期 [closed]

- **现象**: 基于问题 https://gitcode.com/cann/pypto/issues/532 的修复patch，设置golden并开启精度工具校验，结果如下： 1. tensorgraph校验通过 ```shell
- **状态**: 已关闭
- **来源**: Issue #539

#### Issue #605 - BatchMatmul 4D场景，搬出时Reshape走入了UB [closed]

- **现象**: 在BatchMatmul 4D新优化方案中，搬出时Reshape走入了UB，定位到在Expandfunction层，引入了该错误。理论上，在新优化方案中，搬出时的reshape应当在源地址进行拆轴，不需要搬入UB进行。
- **规避方案**: 原expand_function中在遇到reshape级联assemble的场景，会验证assemble输出输入之间除第一维外的shape是否相等，若不相等则进行展开，进而导致一些场景tensor经过多余的搬入搬出。目前此段逻辑无效，需进行删除。https://gitcode.com/cann/pypto/pull/1368
- **状态**: 已修复
- **来源**: Issue #605

#### Issue #608 - 算子添加对一个outcast的assemble后，其他outcast出现精度问题，疑似出现内存踩踏 [open]

- **现象**: 代码示例如下，我们在第二个loop中，对outcast `d_embeddings `进行assemble操作。 当前现象如下： 1. 添加第二个loop后，其他的outcast出现精度问题；若将其注释掉，则其他的outcast精度正确
- **状态**: 仍在处理
- **来源**: Issue #608

#### Issue #626 -  pypto.full 创建的tensor放在loop外有精度问题，放在loop里则精度正确，通过精度校验工具发现TensorGraph阶段有问题 [closed]

- **现象**: 1、pypto.full 创建的tensor放在loop外有精度问题，放在loop里则精度正确，通过精度校验工具发现TensorGraph阶段有问题 pypto.full放在for循环loop外面： torch.full 同样放在for循环外面：
- **规避方案**: ![image.png](assets/issue-626/image.png 'image.png')
- **状态**: 已关闭
- **来源**: Issue #626

#### Issue #639 - pto算子内搬运语句影响torch计算结果 [closed]

- **现象**: gdr算子开发，按照下图的方式在前端手动pad，并且使用for循环将搬运的逻辑隔离，但是出现了很奇怪的现象，这样操作之后pto算子会影响到torch的计算结果。 ``` for _ in pypto.loop(actual_l < l, name="LOOP_PAD", idx_name="pad_...
- **规避方案**: ![image.png](assets/issue-639/image.png 'image.png')
- **状态**: 已关闭
- **来源**: Issue #639

#### Issue #653 - gdr算子开发，前端写法改变，将循环展开，会出现精度问题 [closed]

- **现象**: gdr算子开发，前端写法改变，将循环展开，会出现精度问题。具体代码段见压缩包。 使用136-166行写法精度问题 使用92-132行写法，会出现精度问题，但两者之间只是将循环展开写，且之前运行精度也没有问题
- **规避方案**: **State**: closed **Author**: ren-wenqian1 **Created**: 2026-03-13T14:35:29+08:00 使用136-166行写法精度问题 ![image.png](assets/issue-653/image.png 'image.png') 使用92-132行写法，会出现精度问题，但两者之间只是将循环展开写，且之前运行精度也没有问题 !...
- **状态**: 已关闭
- **来源**: Issue #653

#### Issue #656 - 正反向算子接入整网，同时调用正向和反向算子，反向算子会出现精度问题 [closed]

- **现象**: 我的场景上正向和反向算子没有依赖关系，正向计算结果不影响反向计算结果。 在调用正向和反向计算是，反向计算的算子编译时间过长（4min以上，单独测试反向算子不到1min编译时间），并且输出的scale_grad精度出现问题。 将正向的pypto算子换成不同torch小算子后，反向pypto算子编译时间...
- **根因**: 请联系接口人安排定位
- **规避方案**: ```python device_id = 6 torch.npu.set_device(int(device_id)) net = EmbedQuantizer(8)
- **状态**: 已关闭
- **来源**: Issue #656

#### Issue #670 - gdr算子开发，手动assemble实现尾块pad，出现精度问题，输出tensor都是0，疑似内存问题 [open]

- **现象**: gdr算子，尾块长度不确定。当actual_l=l的时候，会往一个shape为[l,1]大小的tensor里搬运一个shape为[l,1]，valid_shape=[0,1]的full 0 tensor，多次执行用例会偶现精度问题。 代码： bbd1d53441c94bd992090d92156df...
- **规避方案**: 规避手段 ![image.png](assets/issue-670/image.png 'image.png') ![image.png](assets/issue-670/image.png 'image.png') 保留[0,128]的搬运，注释[0,1]的搬运，精度跑十次都没有问题，疑似非对齐搬运问题
- **状态**: 仍在处理
- **来源**: Issue #670

#### Issue #698 - 用 = 对输出进行赋值，输出结果不符合预期 [closed]

- **现象**: 如题 910B 运行以下代码：
- **状态**: 已关闭
- **来源**: Issue #698

#### Issue #767 - Precision error with view + reshape (no computation; fails when nv/nqk > 1) [closed]

- **现象**: When using PyPTO for **view + reshape only (no numerical computation)** as a pure data-copy path, the output does not match the PyTorch golden referen...
- **状态**: 已关闭
- **来源**: Issue #767

#### Issue #788 - 更新pypto仓，原本可以跑过的算子出现精度问题 [closed]

- **现象**: 在commit id：fbf2d3b1802dbc84c1a7717e139b8550e26a6ca gdr算子可以跑过： 更新pypto为commit id：8040ed3faf9533ef9eeb7d185a7b02e544a667b8
- **状态**: 已关闭
- **来源**: Issue #788

#### Issue #820 - 设置sg_set_scope开关，出现精度问题 [closed]

- **现象**: gdr算子开发，设置代码96行 代码：83287e6078cc4724a50de5a1f4b66a1f.zip Device: Ascend 910B
- **规避方案**: - row = attn_dim1.view([1, col_num], [i, 0]) + 0.0 - row_r1 = row.reshape([size, row_num], inplace=True) # inplace设置为True row_v = row_r1.view([size, i], [0, 0]) row_t = row_v.transpose(1, 0)
- **状态**: 已关闭
- **来源**: Issue #820

#### Issue #836 - 在transpose前添加valid_shape导致精度问题 [closed]

- **现象**: 在680行使用view+valid_shape，此处valid_shape使用的是正确的valid_s，但是精度有问题。若是使用681行错误的valid_shape则精度能过。 reshape和batch_matmul也会出现丢失valid_shape的情况，跟此处问题暂时不相关 A3
- **状态**: 已关闭
- **来源**: Issue #836

#### Issue #874 - 融合算子精度失败。 [closed]

- **现象**: 融合算子精度失败，二分定位到BMM新特性的提交导致的问题。 Ascend950PR. torch_npu: 2.6.0.post5.dev20251216+Python 3.11.14
- **状态**: 已关闭
- **来源**: Issue #874

#### Issue #879 - view->reshape，精度异常 [open]

- **现象**: view->reshape，精度异常。先在外面做完reshape+inplace，然后再view，精度就通过了 A3 精度正确，reshape无拷贝
- **状态**: 仍在处理
- **来源**: Issue #879

#### Issue #904 - tile_shape设置尾轴不能被整除时，最后计算结果和设置整除的情况下不同，有精度问题 [open]

- **现象**: 当输入shape为 N 75776, M 2560,
- **规避方案**: diff = pypto.sub(rounded, clamped) abs_diff = pypto.abs(diff) out_of_bounds = pypto.clip(abs_diff, 0.0, 1.0) neg_out_of_bounds = pypto.mul(out_of_bounds, -1.0)
- **状态**: 仍在处理
- **来源**: Issue #904

#### Issue #933 - ExpandExpDif的输入尾轴为1且不连续时精度不对 [closed]

- **现象**: ExpandExpDif的输入尾轴为1且不连续时精度不对 ALL ExpandExpDif的输入尾轴为1且不连续时精度不对
- **状态**: 已关闭
- **来源**: Issue #933

#### Issue #973 - LI算子在2k以上case下出现精度错误 [closed]

- **现象**: 在b=1，s1=1，act_seq=129090时，topk出现精度错误。 环境：100.102.180.181 x86_64
- **规避方案**: https://gitcode.com/cann/pypto/pull/2149 最后定位是pass同步问题，请 [@hwwangmingjun](https://gitcode.com/hwwangmingjun) 关注
- **状态**: 已关闭
- **来源**: Issue #973


### D. 性能问题类

**Issue 数量**: 13

#### Issue #70 - 【GDN算子开发】求逆代码优化，assemble无法正确组装元素 [closed]

- **现象**: 在开发GDN算子过程中，优化下三角矩阵求逆部分代码时，发现assemble无法正确组装元素，会引入很多0。 npu: 910B cann: 8.3.0.1.200:8.3.RC1
- **状态**: 已关闭
- **来源**: Issue #70

#### Issue #110 - GDR算子中decay_mask计算添加合轴优化没有效果 [closed]

- **现象**: GDR算子中计算decay_mask时，存在[L-1]-[1,L]的广播操作，添加合轴优化配置，无变化 pypto.experimental.set_operation_config(combine_axis=True) pypto.experimental.set_operation_config...
- **根因**: 当前输入为(L,1)(1,L)，并未命中合轴优化模式，因此Pass处理广播操作仍使用EXPAND OP而非BRCB OP。 处理结论： (L,1)(1,L)模式不在当前版本的支持范围内。当前版本的合轴优化，仅针对输入为(L,1)(L,N)模式的ADD、SUB等OP采用BRCB广播。 当前输入为(L,1)(1,L)，并未命中合轴优化模式，因此Pass处理广播操作仍使用EXPAND OP而非BRCB ...
- **状态**: 已关闭
- **来源**: Issue #110

#### Issue #231 - 使用新前端@pytpo.frontend.jit, GLM算子性能下降明显 [closed]

- **现象**: 将旧前端迁移到新前端之后 pypto.jit --> pypto.frontend.jit，GLM算子性能下降明显。 首次调用时，复杂算子耗时增加100%，简单算子耗时增加0~10% ~~后续调用时，未复用编译产物，没有ControlFlowCache~~
- **状态**: 已关闭
- **来源**: Issue #231

#### Issue #307 - 新前端，第二次调用kernel时，端到端耗时过长 [closed]

- **现象**: 切换到新前端后，对kernel第二次调用时，端到端耗时过长 1. parser的创建、parser解析jit输入输出，耗时占比70% 2. _dispatch_with_run_mode上板耗时为旧前端的两倍
- **状态**: 已关闭
- **来源**: Issue #307

#### Issue #334 - Lightning_indexer_prolog_quant算子性能劣化2倍 [closed]

- **现象**: 采集性能数据时，发现库上的deepseek_lightning_indexer_prolog_quant.py算子在 case：test_b1_s1_4k_s2_64k() 1月13号的 Commit: 37a6b2f1436a4794ab1cdb29b11f8e84776e628a
- **状态**: 已关闭
- **来源**: Issue #334

#### Issue #388 - 【green light】在 GLM 网络里接入通信 Combine 算子时，每轮推理都会生成 output 目录，导致性能很差，半分钟才推理出一个 token [closed]

- **现象**: 如题 910C kernel 代码：
- **状态**: 已关闭
- **来源**: Issue #388

#### Issue #471 - 泳道图过大场景（例如泳道图文件1个G），目前无法正常打开，无法支撑性能调优，需要解决 [closed]

- **现象**: 泳道图过大场景（例如泳道图文件1个G），目前无法正常打开，无法支撑性能调优，需要解决 不涉及 生成过大泳道图，例如泳道图文件1个G
- **状态**: 已关闭
- **来源**: Issue #471

#### Issue #538 - 仿真模式单任务时间异常偏短，使得无法通过仿真估计合适的性能数据 [open]

- **现象**: 以pypto/examples/03_advanced/advanced_nn/attention/attention.py样例的npu模式和sim模式的计算结果为例。 npu模式：实际上板泳道图 sim模式：仿真环境泳道图
- **状态**: 仍在处理
- **来源**: Issue #538

#### Issue #618 - 【green light】性能优化：调度时间过久 [closed]

- **现象**: 构造了一个cube:vec == 1:1的场景，其中vec计算依赖cube，当cube计算完之后，vec等待了19us才调度起来 A2 pypto/models/glm_v4_5/glm_attention.py 代码中，设置cube_l1_reuse_setting： {0： 8}
- **状态**: 已关闭
- **来源**: Issue #618

#### Issue #843 - 整网接入时关闭AI CPU抢跑影响推理性能 [closed]

- **现象**: 在PyPTO QAT算子接入训练整网时，为避免超时问题关闭AI CPU抢跑功能，会影响推理时的性能，导致单算子性能下降20us+。 云道整网训练&推理环境 关闭AI CPU抢跑功能将影响推理时的性能，导致一个算子性能下降20us+，影响后续其他算子性能要求。
- **规避方案**: ## Comments (1)
- **状态**: 已关闭
- **来源**: Issue #843

#### Issue #872 - 合轴优化，两路输入尾轴都是1不用插brcb或者expand [closed]

- **现象**: 合轴优化，两路输入尾轴都是1不用插brcb或者expand 910b 合轴优化，两路输入尾轴都是1不用插brcb或者expand
- **状态**: 已关闭
- **来源**: Issue #872

#### Issue #938 - pypto.loop中的unroll_list功能不符合预期，cube子图种类数大大翻倍，性能调优受到阻碍 [open]

- **现象**: 库上models/qwen3_next/qwen3_next_gated_delta_rule.py算子在test_b2_nqk4_nv8_s4k用例下 使用了pypto.loop中的unroll_list=[16, 1]功能对root func进行展开，来减少root func的个数，进而使整算子...
- **状态**: 仍在处理
- **来源**: Issue #938

#### Issue #969 - [GreenLight].pypto-op-perf-autotuner内容缺乏 [open]

- **现象**: 会导致对模型依赖过高 A3 skill不全
- **规避方案**: 用什么回退？git checkout？手动撤销？ 如果优化涉及多个文件的修改呢？ NA
- **状态**: 仍在处理
- **来源**: Issue #969


### E. 功能缺失类

**Issue 数量**: 6

#### Issue #189 - 新前端pypto.forntend.jit不支持torchTensor转成ptoTensor时Format(NZ)属性的带入 [closed]

- **现象**: 新前端torchTensor转成ptoTensor时没有支持Format(NZ)属性，torchTensor是NZ格式，但是进入jit后转成ptoTensor后，格式还是ND；导致适配新前端后NZ场景出现精度报错。（适配前的老写法，NZ场景精度ok） NPU: 910B CANN : 8.5.0
- **规避方案**: <a href="https://gitcode.com/user-attachments/files/8635329/55859784500c4b5283c8f2a5e278b7f0.zip" target="_blank">55859784500c4b5283c8f2a5e278b7f0.zip</a> 新前端写法精度Fail脚本： <a href="https://gitcode.com/u...
- **状态**: 已关闭
- **来源**: Issue #189

#### Issue #265 - 不支持conv算子 [open]

- **现象**: PyPTO的算子库中没有提供卷积（Convolution）算子的实现。这使得框架无法直接用于构建或运行依赖卷积操作的神经网络模型（如CNN），极大地限制了其在计算机视觉等核心AI领域的应用。 硬件: 昇腾910B/910C 在PyPTO中尝试创建一个卷积层（例如，定义一个2D卷积操作）。导入或调用相...
- **状态**: 仍在处理
- **来源**: Issue #265

#### Issue #266 - 不支持私有格式和5hd基础格式 [open]

- **现象**: PyPTO目前不支持华为昇腾硬件上常用的5HD数据格式，无法使用为5HD格式优化的硬件指令和内存布局，难以达到峰值算力。 硬件: 昇腾910B/910C 尝试将一个使用5HD格式（NC1HWC0）的张量传入PyPTO算子进行计算
- **状态**: 仍在处理
- **来源**: Issue #266

#### Issue #752 - Batch matmul 场景 coredump，view dynOffset缺失 [closed]

- **现象**: Batch matmul 场景 coredump，view dynOffset缺失 910b Batch matmul 场景 coredump，view dynOffset缺失
- **状态**: 已关闭
- **来源**: Issue #752

#### Issue #995 -  修复 View/Assemble SourceLocation缺失问题 [open]

- **现象**: 修复 View/Assemble SourceLocation缺失问题 N/A N/A
- **状态**: 仍在处理
- **来源**: Issue #995

#### Issue #1015 - pypto.view接口不支持动态validshape [open]

- **现象**: pypto.view接口不支持前端传递的动态validshape 例如：eff_res_index = pypto.view(res_index_assemble, [1, selected_count], [dst_offset, 0], valid_shape=[1, eff_seq]) pad...
- **根因**: 1、设备环境：100.102.180.181，x86_64 2、最新主线代码
- **规避方案**: pypto.set_pass_options(pg_skip_partition=True) pypto.set_vec_tile_shapes(1, selected_count)
- **状态**: 仍在处理
- **来源**: Issue #1015


### G. 其他问题类

**Issue 数量**: 60

#### Issue #9 - PyPTO项目中没有找到卷积函数 [open]

- **现象**: 我想通过PyPTO实现cifar10图像分类模型，其中需要调用conv卷积函数，我看到项目提供了sigmoid和relu等函数，但是没有提供卷积conv函数； 想问下是否有其他方案？还是需要用户自定义实现？ 无
- **状态**: 仍在处理
- **来源**: Issue #9

#### Issue #36 - 无法简单地在已有的tensor上更新值 [closed]

- **现象**: 下面的代码中，我想构造对角线上有四个b的c。 在构造的全0的tensor上试图使用c[xx:xx]=b，运行报错 ```
- **根因**: * 选项2： 强制约束：降低易用性，前端文档需显示声明该约束，并且当前端识别到类似用法时需显式报错并引导用户修改 另外该规避方式为合法的前端表达，对于由规避引起的新问题：
- **规避方案**: ![屏幕截图 2026-01-05 141742.jpg](assets/issue-36/屏幕截图_2026-01-05_141742.jpg '屏幕截图 2026-01-05 141742.jpg') [@tsungl4](https://gitcode.com/tsungl4) 请关注，inplace依赖丢失问题 为了正面解决该问题，可以包括但不限于以下两个选项： * 选项1： 正面支持：变...
- **状态**: 已关闭
- **来源**: Issue #36

#### Issue #40 - + 0.0 消除问题 [closed]

- **现象**: 代码中对 tensor 进行 view 和 concat 时，框架无法自动识别和创建新的 raw tensor，导致需要手动 + 0.0 创建，否则会精度报错。 910B2 inverse_pto_16_16 函数中的三个 + 0.0 操作
- **状态**: 已关闭
- **来源**: Issue #40

#### Issue #44 - /tools/prepare_env.sh 中提供的 cann 版本 与 其他开源框架（比如 mindspeed-llm、mindspeed、mindspore等）依赖的商用cann 版本存在冲突 [closed]

- **现象**: 使用 /tools/prepare_env.sh 中提供的 cann 的 8.5.0 版本，可以支持 pypto 的正常功能，但是在需要融合pypto 到其他 训练推理框架的场景下，发现这个版本的cann 无法支持其他的开源框架，对比可以支持开源框架的 商用版 cann 后发现缺失一些module，...
- **规避方案**: > >[@jason_yuan_ye](https://gitcode.com/jason_yuan_ye) >经过测试，已经可以在正常训练流程中使用 pypto 融合算子并达到性能提升，感谢 [@jason_yuan_ye](https://gitcode.com/jason_yuan_ye) 经过测试，已经可以在正常训练流程中使用 pypto 融合算子并达到性能提升，感谢 你好，感谢反馈。当前...
- **状态**: 已关闭
- **来源**: Issue #44

#### Issue #60 - module 'pypto' has no attribute 'frontend' [closed]

- **现象**: (ms_training_tools) [ma-user test]$python pypto/examples/01_beginner/basic/basic_ops.py ============================================================ P...
- **状态**: 已关闭
- **来源**: Issue #60

#### Issue #90 - pto-isa安装包链接失效 [closed]

- **现象**: https://gitcode.com/cann/pypto/blob/master/docs/context/prepare_environment.md 其中pto下载安装包链接失效，无法点开 资料问题
- **状态**: 已关闭
- **来源**: Issue #90

#### Issue #95 -  [closed]

- **现象**: 使用pypto.set_vec_tile_shapes接口报错： python3 examples/03_advanced/advanced_nn/arctic_lstm/sum_lstm.py Compiling Kernel (Mode: npu)...
- **状态**: 已关闭
- **来源**: Issue #95

#### Issue #126 - 非整除场景切片操作和pypto.view效果不一致 [closed]

- **现象**: attn_inv_list[1] = attn[:2, :]     #得到tensor是[0, 0]，应该是[2, 14] 换成view就正确： attn_inv_list[1] = attn.view([2, min_length], [0, 0], valid_shape=[2, actual...
- **状态**: 已关闭
- **来源**: Issue #126

#### Issue #132 - error: no matching function for call to 'BinaryPlusInstr' [closed]

- **现象**: 执行链接中的算子：https://gitcode.com/xzero72/pypto/blob/work_branch/examples/03_advanced/advanced_nn/arctic_lstm/sum_lstm.py 当把tile_cfg.h_tile = 512修改成tile_cf...
- **规避方案**: python3 examples/03_advanced/advanced_nn/arctic_lstm/sum_lstm.py Compiling Kernel (Mode: npu)... args.test_type performance
- **状态**: 已关闭
- **来源**: Issue #132

#### Issue #150 - 调度的时间长：端到端打点时间和泳道图算子时间 [closed]

- **现象**: 调度的时间长：端到端打点时间和泳道图算子时间 端到端打点时间：1.8785 ms python3 examples/03_advanced/advanced_nn/arctic_lstm/sum_lstm.py
- **状态**: 已关闭
- **来源**: Issue #150

#### Issue #166 - AddressSanitizer: attempting free on address which was not malloc()-ed: 0x12c100000000 in thread T0 [closed]

- **现象**: Note: Google Test filter = ViewType.dequant_test_bf16_2_int8 [==========] Running 1 test from 1 test suite. [----------] Global test environment set-u...
- **状态**: 已关闭
- **来源**: Issue #166

#### Issue #167 - ERROR: LeakSanitizer: detected memory leaks ，framework/tests/st/machine/src/ops/test_dynamic_binding.cpp [closed]

- **现象**: Note: Google Test filter = DynamicBindingTest.TestDeviceCompute [==========] Running 1 test from 1 test suite. [----------] Global test environment se...
- **状态**: 已关闭
- **来源**: Issue #167

#### Issue #168 - Direct leak of   OnBoardTest.test_sin_dim2_float32 [closed]

- **现象**: 执行 ST 出现了内存泄漏。 A3 CLANG 12 执行对应 ST用例。
- **状态**: 已关闭
- **来源**: Issue #168

#### Issue #179 - 动态循环的依赖建立有误 [closed]

- **现象**: 动态更新tensor的依赖建立有误 +------------------------------------------------------------------------------------------------+ | npu-smi 25.3.rc1               ...
- **规避方案**: ![image.png](assets/issue-179/image.png 'image.png') [@tsungl4](https://gitcode.com/tsungl4)  [@seu_chang](https://gitcode.com/seu_chang) 我们可以在这里讨论下这个问题。
- **状态**: 已关闭
- **来源**: Issue #179

#### Issue #180 - scatter_update相关的依赖建立有误 [closed]

- **现象**: scatter_update相关的依赖建立有误，不管是两个update OP之间、还是它与其他OP之间都没看到正常的依赖边。 root:/home/# npu-smi info +------------------------------------------------------------...
- **状态**: 已关闭
- **来源**: Issue #180

#### Issue #195 - 新版pypto需要适配的cann包版本，接入sglang整网会出现问题 [closed]

- **现象**: 新版的pypto需要适配的cann包版本为： 但sglang整网环境使用该版本cann包运行会报错： 并且source该cann包后，重新编译安装sgl-kernel会报错：
- **状态**: 已关闭
- **来源**: Issue #195

#### Issue #208 - 新前端@pypto.frontend.jit, tensor赋值，消除[:]问题 [closed]

- **现象**: 如果将输出tensor放到输入中，对tensor的赋值时，需要加[:] out[:] = a + b 如果输出是通过return返回，则不需要加[:]
- **状态**: 已关闭
- **来源**: Issue #208

#### Issue #215 -  [closed]

- **现象**: 1、报错信息： 流同步失败 2、用例：
- **状态**: 已关闭
- **来源**: Issue #215

#### Issue #217 - Profiling can't run successfully on some tests(cpp前端泳道图) [closed]

- **现象**: When running test(for example single MatMul test) without profiling with cpp it passes, but when profiling is initiated the error occures. CANN 8.5.0(...
- **状态**: 已关闭
- **来源**: Issue #217

#### Issue #254 - [Bug-Report|缺陷反馈]:Assemble入参校验逻辑问题 [closed]

- **现象**: 最新开源仓运行gdr算子时报错 ```Python last_state[:] = cur_state
- **状态**: 已关闭
- **来源**: Issue #254

#### Issue #261 - pto-isa使用社区最新版本，出现大量 warming 显示使用C++20 [closed]

- **现象**: 当前 pypto 最新主线需要使用最新版本的 pto-isa 包，更换后，发现出现大量的 warming 告警，但不影响精度 初步分析发现是c++20 需要使用-std=c++20，但是源码中写的都是-std=c++17 A2/A3
- **状态**: 已关闭
- **来源**: Issue #261

#### Issue #268 -  scatter_update刷新两次时，和其他op间的依赖建立有误 [open]

- **现象**: scatter_update相关的依赖建立有误：如果将a刷新成a1，再刷新成a2，则a1作为输入的OP无法和第二次刷新建立起保证顺序的依赖关系，导致计算结果错误。 npu-smi 25.3.rc1                 Version: 25.3.rc1                  ...
- **规避方案**: ``` index1 = pypto.view(index, [1, 8], [0, 0]) a = pypto.scatter_update(a, -2, index1, b) out1[:] = a+1.0
- **状态**: 仍在处理
- **来源**: Issue #268

#### Issue #308 - Unroll loops cause variables to be out of scope [closed]

- **现象**: ```python @pypto.frontend.jit( runtime_options={"run_mode": mode,
- **状态**: 已关闭
- **来源**: Issue #308

#### Issue #313 - 【green light】pypto.unsqueeze没有生效 [closed]

- **现象**: pypto.unsqueeze没有生效，shape没有变化 Ascend950 python3 pypto_scatter_nd_sub.py
- **状态**: 已关闭
- **来源**: Issue #313

#### Issue #331 - `loop_unroll` doesn't retain actual indexes [closed]

- **现象**: Using `loop_unroll` doesn't maintain actual indexes. Using the following code: ```python
- **状态**: 已关闭
- **来源**: Issue #331

#### Issue #332 - tilefwk_prof_data_parser.py无法使用 [closed]

- **现象**: tilefwk_prof_data_parser.py运行会报错 运行指令为： python3 tools/profiling/tilefwk_prof_data_parser.py -p /home/j00911874/pypto/build/PROF_000001_202602041638092...
- **状态**: 已关闭
- **来源**: Issue #332

#### Issue #335 - 不同task之间间隙较大 [closed]

- **现象**: 执行sum_lstm算子后通过泳道图发现部分task的间隔时间较长，影响了整体性能，最长间隔有6us，如下图 存在其他task之间间隔仅有2us（如下图） 是否可以整体减少task之间的间隔，这样算子性能可以得到大幅度提升
- **状态**: 已关闭
- **来源**: Issue #335

#### Issue #361 - pypto.view MTE Out-of-Range Error: Memory Offset Corruption for Non-Zero Indices in Multi-Dimensional Tensors [closed]

- **现象**: The pypto.view function triggers a hardware-level memory access violation when a non-zero offset is applied to an intermediate dimension of a multi-di...
- **状态**: 已关闭
- **来源**: Issue #361

#### Issue #381 - [green light] pypto内怎么通过一个list/tuple数据构建新的pyto tensor [closed]

- **现象**: 在@pypto.frontend.jit内部，怎么通过一个list/tuple，比如 strides = [1,2,3,4,5]， 创建一个pypto tensor，内部数据是[1,2,3,4,5] 任意昇腾硬件 NA
- **状态**: 已关闭
- **来源**: Issue #381

#### Issue #428 - 空 tensor 非法计算问题拦截 [closed]

- **现象**: 开发gdr非整除求逆算法时，存在取valid_shape外的空tensor之后mul的操作，该操作框架会识别出最小valid_shape来运算。但开启精度工具之后，会在这里进行拦截。该操作的合理性，以及工具是否应该拦截该问题，前端、pass、运行时如何拦截【空 tensor 非法计算】问题，希望给出...
- **规避方案**: 框架会识别出最小valid_shape来运算   是指框架会按(0,0)的shape来计算吗？mul的两个输入的validshape要相同吧
- **状态**: 已关闭
- **来源**: Issue #428

#### Issue #466 - 新前端pypto.is_loop_begin的问题 [closed]

- **现象**: 在使用新前端的pypto.is_loop_begin的时候，我写在循环里不会出错，如下： ``` def mytest(a, b, c):
- **状态**: 已关闭
- **来源**: Issue #466

#### Issue #473 - [green light] 动态shape场景shape获取错误 [closed]

- **现象**: tile_indices = 32 loop_indices = num_batch // tile_indices for idx in pypto.loop(loop_indices):
- **状态**: 已关闭
- **来源**: Issue #473

#### Issue #484 - 【green light】循环调用kernel只有第一次调用成功 [closed]

- **现象**: pypto代码中通过for循环调用kernel，通过打印输出结果看只有第一次循环调用成功，后续循环均没有调用成功 ascend950 ```
- **状态**: 已关闭
- **来源**: Issue #484

#### Issue #489 - loop和loop间的依赖边建立有误 [open]

- **现象**: 在loop和loop之间有tensor传输时，如果前者是assemble产生的这个tensor，则泳道图上的依赖关系有误，某些场景存在偶现精度错误 npu-smi info +--------------------------------------------------------------...
- **状态**: 仍在处理
- **来源**: Issue #489

#### Issue #561 - 【green light】test_common.h 里 std::abs(diff / expVal) 似乎有除 0 问题？ [closed]

- **现象**: framework/tests/st/utils/include/test_common.h ```cpp template
- **状态**: 已关闭
- **来源**: Issue #561

#### Issue #575 - [green light] 使用pypto.round接口时，fp32数据类型的63.5四舍五入为63，torch.round四舍五入为64，与torch.round行为不一致，请分析 [closed]

- **现象**: 创建了一个float32的pypto.tensor，再使用pypto.round接口进行四舍五入操作时，fp32数据类型的63.5转为63，而torch.round操作转为64，与torch.round行为不一致，请分析 Ascend950 kernel代码：
- **状态**: 已关闭
- **来源**: Issue #575

#### Issue #596 -  基础 view/assemble M/N 分块场景触发内部异常: bad incast tensor addr [open]

- **现象**: 在 origin/master 上，使用纯基础算子（view + assemble + loop + 普通子函数调用）构造 M/N 分块后，SIM 模式稳定触发内部异常： Caught exception: 'bad incast tensor addr: ...'。 同一脚本在 NPU 模式下可正...
- **根因**: 初步看是PvModel相关代码触发异常，需要仿真同事进一步分析
- **状态**: 仍在处理
- **来源**: Issue #596

#### Issue #620 - [GreenLight]在使用官方agent辅助编程时,对于多输出算子无法正确找到写法样例 [closed]

- **现象**: 在用pypto提供的agent工具开发ApplyRMSProp算子时,agent根据我的提示词 在搜索多输出算子写法的时候 无法正确找到库上的多输出算子写法  agent怀疑我设计有问题 910b 用库上的配置 要求agent实现一个多输出的算子
- **规避方案**: **State**: closed **Author**: zhang-song-rui **Created**: 2026-03-10T17:32:18+08:00 910b opencode没有记录留存
- **状态**: 已关闭
- **来源**: Issue #620

#### Issue #621 - Pypto对于valid_shape使用存在约束，view和reshape等场景无法自动推导，需要在文档里汇总说明 [closed]

- **现象**: mla_prolog改进算子开发过程中识别到，Pypto对于valid_shape使用存在约束，view和reshape等场景无法自动推导，需要在文档里汇总并显著说明 Ascend910B 参见问题描述
- **状态**: 已关闭
- **来源**: Issue #621

#### Issue #632 - 【greenlight】pypto.sum功能存在问题 [closed]

- **现象**: pypto.sum(temp, 0) 调用pypto.sum功能存在问题 ascend950 在kernel内调用pypto.sum(temp, 0)，累加结果错误
- **状态**: 已关闭
- **来源**: Issue #632

#### Issue #637 - 框架需保持对确定性计算的支持能力 [closed]

- **现象**: ``` 为避免遗漏，这里使用单独的 issue 统一跟踪对**框架支持确定性计算**的诉求 ```
- **状态**: 已关闭
- **来源**: Issue #637

#### Issue #703 - where+eq 输出不符合预期 [closed]

- **现象**: 如题 910B 运行以下代码：
- **状态**: 已关闭
- **来源**: Issue #703

#### Issue #707 - A5 IsFinite 算子初始化异常 [closed]

- **现象**: A5 平台 isfinite 算子运行时异常 A5 芯片 A5平台合轴场景下执行 isfinite 算子
- **状态**: 已关闭
- **来源**: Issue #707

#### Issue #753 - qat对称正向算子，开启日志打印workspacesize=0 [closed]

- **现象**: qat对称正向算子，开启日志打印workspaceSize=0 算子代码：a25c51f875a94f8a898f2e2f520f6c27.zip Device: Ascend 910B
- **状态**: 已关闭
- **来源**: Issue #753

#### Issue #775 - PyPTO QAT算子接入整网后拉起失败 [closed]

- **现象**: 在云道训练环境上，将PyPTO QAT算子接入整网后，训练模型拉起失败 910B，pytorch_2.6.0，cann_8.5.1.b010，py_3.11，aarch64 在云道训练环境下替换PyPTO QAT算子并拉起训练任务。
- **状态**: 已关闭
- **来源**: Issue #775

#### Issue #809 - 通信算子接入整网，开启stitch_cfgcache_size配置，会在emulation阶段挂掉 [closed]

- **现象**: 通信算子接入整网，开启stitch_cfgcache_size配置，会在emulation阶段挂掉 910B 无
- **状态**: 已关闭
- **来源**: Issue #809

#### Issue #817 - 64卡整网中接入pypto算子出现host内存泄露问题 [closed]

- **现象**: 64卡整网训练中，接入pypto算子后，物理内存线性增加，可能出现内存泄漏 NPU 910B，CANN 8.5.0，PyPTO 0.1.1，HDK 25.5.1，Python 3.11 在云道开发环境下用基于PyPTO的QAT算子替换整网中torch小算子之后，拉起整网训练任务后物理内存一直增长
- **规避方案**: 请分析
- **状态**: 已关闭
- **来源**: Issue #817

#### Issue #823 - IsFinite 算子输出类型为 uint8 而非 bool 类型，导致 Assemble 类型校验错误 [closed]

- **现象**: 在运行 IsFinite 算子时输出类型为 uint8 而非 bool，导致 Assemble 算子运行时异常 A2/A3/A5 均可复现 PyPTO: d9b369faf4f7fefbdd4f4645ef92c32221aba891
- **状态**: 已关闭
- **来源**: Issue #823

#### Issue #828 - 【green light】Machine首轮任务下发时间不规整和任务间调度时间长问题 [closed]

- **现象**: 测试用例：matmul+allreduce+addrmnorm，测试结果及用例代码见附件 问题1：首轮任务下发时间不规整（见测试结果1,4,8），首轮任务前序无依赖，但是会偶发的下发时间不规整，相差间隔能有2us 问题2：任务间调度时间长问题。allreduce中的shmem_put操作仅依赖mat...
- **根因**: 问题2已安排 [@chai-hw](https://gitcode.com/chai-hw) 继续分析
- **状态**: 已关闭
- **来源**: Issue #828

#### Issue #838 - 使用reshape inplace方法输出在泳道图中的问题 [open]

- **现象**: 使用方法1去将输出reshape inplace回3d的形状，在泳道图上有很大一段空白不知道在做什么。目前发现对整网性能无影响，使用"stitch_cfgcache_size": 100000000时会复现该情况。 0da63764de05423dbf8edb665af324af.zip A3
- **规避方案**: 当前结论：1、泳道图呈现的统计端到端耗时长是否应该基于AIcore执行完成的时间，图中红框部分应该不统计 ![image.png](assets/issue-838/image.png 'image.png') 2、pass同事帮忙分析一下为什么计算结果直接reshape给output tensor会一直产生hub节点
- **状态**: 仍在处理
- **来源**: Issue #838

#### Issue #858 - RemoveRedundantOp插入冗余节点导致view直连outcast形成非法图 [closed]

- **现象**: 图非法，产生了冗余的view节点，同时该节点链接了outcast，非法行为 A2/3 运行算子代码 python3 ops/pass_splitlangefanouttensor.py 15
- **状态**: 已关闭
- **来源**: Issue #858

#### Issue #861 - master branch runs ~15% slower now than 3 weeks ago [closed]

- **现象**: The performance of PyPTO decreased severely (around ~15%) at some point between committs: f05429969dc98809b1c7d181b9d24039886ca161 and 82e7eacfc0c0d13...
- **状态**: 已关闭
- **来源**: Issue #861

#### Issue #873 - 大量创建pypto Tensor后产生内存泄露 [closed]

- **现象**: ``` import torch import psutil
- **状态**: 已关闭
- **来源**: Issue #873

#### Issue #875 -  example示例case npu模式和sim模式均有用例失败，请整改 [open]

- **现象**: example示例case npu模式和sim模式均有用例失败，请整改 NPU模式失败2个： SIM模式失败20个：
- **状态**: 仍在处理
- **来源**: Issue #875

#### Issue #890 - l1size计算修复 [closed]

- **现象**: fmap占用l1空间计算修复 Device: Ascend 950 CANN: 9.0.0
- **状态**: 已关闭
- **来源**: Issue #890

#### Issue #906 - 当前operation开合轴的条件没有对齐，在白名单内的operation才能开合轴 [closed]

- **现象**: 当前operation开合轴的条件没有对齐，在白名单内的operation才能开合轴 A3 白名单以外的operation不增加Brc，当前运行相关operation的时候，会出现插入Brc，从而出现预期之外的结果
- **状态**: 已关闭
- **来源**: Issue #906

#### Issue #909 - 依赖函数未定义 [open]

- **现象**: 文件位置：version.cmake 行 9-15 问题描述：set_package、set_build_dependencies、set_run_dependencies 函数未在文件中定义，也未说明来自哪个 CMake 模块，可能导致构建失败。 # -----------------------...
- **规避方案**: 暂无日志
- **状态**: 仍在处理
- **来源**: Issue #909

#### Issue #910 - 修复部分rawtensorindex没有正确encode [open]

- **现象**: 当前encode了incast/outcast， 还有部分非incast/outcast tensor需要做RawTensorIndex替换 NA NA
- **状态**: 仍在处理
- **来源**: Issue #910

#### Issue #991 - mla_prolog算子改进，输入query_start_loc是INT64且涉及动态轴，需要cast为INT32，使用cast之后的Tensor进行GetTensorData操作并叠加减法op，发生内存越界 [open]

- **现象**: mla_prolog算子改进，输入query_start_loc是INT64且涉及动态轴，需要cast为INT32，使用cast之后的Tensor进行GetTensorData操作并叠加减法op，发生内存越界 以下测试代码中，query_start_loc作为外部输入已转为INT32，代码中quer...
- **规避方案**: 代码片段如下 ``` B = query_start_loc.shape[0] - 1
- **状态**: 仍在处理
- **来源**: Issue #991

#### Issue #1011 - 删除docs中对于老前端pypto.jit的描述，防止agent生成了错误的代码 [open]

- **现象**: 文档中存在对于老前端pypto.jit的描述，一定概率导致agent生成了错误的代码，使用了老前端的装饰器 不涉及 不涉及
- **状态**: 仍在处理
- **来源**: Issue #1011

