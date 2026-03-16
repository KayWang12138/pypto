# PyPTO Pass 模块知识库

本文档详细介绍了 PyPTO 中常见的 Pass 模块及其功能、常见错误和可能原因。

## 常见 Pass 模块及错误模式

### 1. RemoveRedundantReshape

**功能描述**：删除冗余的reshape操作（当所有消费者都是reshape时）

**常见错误**：
- 输入输出操作数无效
- 消费者为空指针
- 操作已删除标记错误

**可能原因**：
- 输入输出操作数数量不为1
- 输入输出tensor为空指针
- 消费者为空指针
- 操作已被其他pass标记为删除

---

### 2. AutoCast

**功能描述**：自动插入类型转换操作（BF16/FP16/INT32-FP16）并优化冗余CAST链

**常见错误**：
- 获取InOutCast连接tensor失败
- 插入BF16 CAST失败
- 插入FP16 CAST失败
- 插入INT32-FP16 CAST失败
- 冗余CAST链消除失败

**可能原因**：
- 架构不支持某些数据类型
- INCAST/OUTCAST节点连接问题
- CAST链优化过程中图结构问题
- 类型转换路径不合法

---

### 3. InferMemoryConflict

**功能描述**：推断内存冲突并插入register copy

**常见错误**：
- 初始化失败
- 前向传播失败
- 后向传播失败
- 插入Copy失败
- 更新前向tensor失败
- 更新后向tensor失败
- 设置默认shape失败
- 推断tile shape失败
- 获取reshape tile失败
- 插入前置copy失败
- 插入后置copy失败
- 获取tile shape失败
- ViewType处理失败
- 检查原始shape冲突失败
- 检查冲突失败
- 检查传输失败
- 验证tile shape失败

**可能原因**：
- ViewType输入输出tensor的数据类型不在viewType表中
- vecTypeTile的tile维度n不是偶数
- 输入输出tensor的raw shape中有负数（动态shape）
- 输入输出tensor的raw size不相等且不匹配特定的reshape pattern（4D<->2D, 3D<->2D的batch matmul优化pattern）
- tile shape维度与输入shape维度不等
- tile shape大小为0
- 无法推导tile shape
- 推导reshape tile shape失败

---

### 4. RemoveUndrivenView

**功能描述**：删除无驱动的view操作（将OP_ASSEMBLE_SSA降级为OP_ASSEMBLE并删除无驱动的OP_VIEW）

**常见错误**：
- 缺少INPLACE_IDX属性错误
- 生产者数量无效错误
- 生产者类型错误

**可能原因**：
- OP_ASSEMBLE_SSA缺少INPLACE_IDX属性
- 输入操作数的生产者数量不为1
- 生产者类型不是OP_VIEW

---

### 5. ExpandFunction

**功能描述**：将tensor graph扩展为tile graph

**常见错误**：
- 操作验证失败
- 更新IO操作数失败
- 操作展开失败
- Scope混合错误

**可能原因**：
- 操作验证失败
- 输入输出操作数为空指针
- 操作不支持展开
- scope配置错误
- 在CV分离平台上混合Cube和Vector操作

---

### 6. MergeViewAssemble

**功能描述**：合并连续的view/assemble操作链

**常见错误**：
- 初始化失败
- 处理操作失败
- 清理失败
- 追加合并的view操作失败
- 追加合并的assemble操作失败
- 合并view链失败
- 合并assemble链失败
- 处理消费者链失败
- 处理assemble消费者失败
- 处理链尾失败
- 处理assemble链尾失败
- 计算合并offset失败
- 计算assemble offset失败
- 删除冗余assemble失败

**可能原因**：
- 创建ViewOpAttribute/AssembleOpAttribute失败
- 获取view/assemble属性失败
- 操作链中存在空指针
- 链首操作没有输入操作数
- 链尾操作没有输出操作数
- 输入/输出tensor为空指针
- offset计算失败
- tensor offset添加失败
- 获取view valid shape失败

---

### 7. SplitReshape

**功能描述**：拆分reshape操作（将一个reshape拆分为多个reshape并添加view/assemble）

**常见错误**：
- 初始化失败
- 收集CopyOut失败
- 检查CopyIn失败
- 添加操作失败
- 删除reshape失败
- 消除死操作失败
- 设置内存类型失败
- 获取变化轴失败
- 检查动态状态失败
- 更新动态shape失败
- 分组reshape偏移失败

**可能原因**：
- 对齐shape不正确（alignedShape中有0）
- 输出维度与动态shape维度不匹配
- 动态shape涉及变化的轴（合轴、分轴、部分合轴）
- offset与dynShape维度不同
- 偏移值维度不一致

---

### 8. SplitRawTensor

**功能描述**：拆分原始tensor

**常见错误**：
- Tensor拆分失败
- 内存分配失败
- Offset计算错误

**可能原因**：
- Tensor size过大
- 内存不足
- offset边界错误

---

### 9. SplitLargeFanoutTensor

**功能描述**：拆分大扇出tensor

**常见错误**：
- LCM shape计算失败
- Offset生成失败
- 操作创建失败

**可能原因**：
- Shape维度不匹配
- GCD/LCM计算溢出
- 内存类型冲突

---

### 10. DuplicateOp

**功能描述**：复制操作

**常见错误**：
- 操作复制失败
- Tensor克隆失败
- 消费者替换失败

**可能原因**：
- 内存不足
- 属性复制不完整
- 消费者链断裂

---

### 11. AssignMemoryType

**功能描述**：为tensor分配内存类型（original和tobe），处理特殊操作（reshape/view/assemble/nop/reduce_acc/shmem_wait_until），处理cube级联场景，插入convert操作

**常见错误**：
- 插入convert失败
- 处理未知内存类型失败
- 处理小tile到大tile失败
- 处理大tile到小tile失败

**可能原因**：
- 操作属性获取失败
- offset与rawshape维度不匹配
- offset计算失败
- 内存对齐检查失败（32字节对齐）
- 内存大小超限（UB/L1阈值）
- 维度倍数检查失败
- tobeMap更新失败
- 视图属性为空
- 输入输出操作数索引越界

---

### 12. InferDiscontinuousInput

**功能描述**：推断不连续输入并插入copy操作

**常见错误**：
- 从INCAST推断失败
- 插入copy操作失败

**可能原因**：
- 图结构复杂
- 内存类型不匹配
- 动态shape处理
- offset重叠检查失败
- view冲突检测失败

---

### 13. RemoveRedundantOp

**功能描述**：删除冗余的view-assemble操作对和reshape操作

**常见错误**：
- 处理DummyOps失败
- 处理DummyOp失败
- 处理Reshape失败
- 处理ViewAssemble失败
- 合并ViewAssemble失败

**可能原因**：
- 输入输出shape和dynshape不相等
- ASSEMBLE输出有多个生产者
- View输入非同源
- Assemble数据重排（view offset与assemble offset不匹配）
- 图连接更新失败

---

### 14. InsertOpForViewAssemble

**功能描述**：为view-assemble插入操作

**常见错误**：
- 操作插入失败
- 属性设置失败
- 图连接更新失败

**可能原因**：
- 内存类型不匹配
- offset计算错误
- 操作顺序依赖

---

### 15. SplitK

**功能描述**：拆分K维度（消除Reduce_Acc操作）

**常见错误**：
- Loop检查失败
- Reduce_Acc消除失败
- 操作验证失败

**可能原因**：
- 存在Loop结构
- OP_A_MUL_B/OP_A_MULACC_B（输出数量不为1、输出tensor不是L0C类型或消费者数量不为1）
- OP_REDUCE_ACC（输入数量小于1、输出数量不为1、输入输出tensor不是DDR类型）

---

### 16. GraphPartition

**功能描述**：图切分（将图划分为多个子图以支持并行执行）

**常见错误**：
- 设置参数失败
- 图切分失败
- 构建操作图失败
- 构建超级节点图失败
- 构建hash值失败
- 构建同构组失败
- 合并非同构组失败
- 合并同构组失败
- 更新切分结果失败

**可能原因**：
- 切分参数未初始化
- 图结构复杂
- 循环依赖
- 资源约束冲突
- 同构组扩展失败

---

### 17. ReduceCopyMerge

**功能描述**：减少copy合并（在DAV_3510平台上处理CV混合图并优化copy操作）

**常见错误**：
- ReduceCopy失败
- 内部子图ID检查失败
- 图构建失败
- 合并策略应用失败
- 循环检测失败
- 混合图验证失败

**可能原因**：
- 操作不属于任何内部子图
- 图结构复杂
- CV混合图权重计算失败
- 合并阈值不满足
- 存在循环依赖
- 混合图比例不合法

---

### 18. NBufferMerge

**功能描述**：向量子图合并

**常见错误**：
- 合并策略失败

**可能原因**：
- 参数配置错误
- 子图成环

---

### 19. L1CopyInReuseMerge

**功能描述**：L1拷贝输入重用合并

**常见错误**：
- 重用分析失败
- L1缓存策略失败
- 依赖关系错误

**可能原因**：
- 缓存容量限制
- 访问模式不规律
- 数据依赖复杂

---

### 20. IntraSubgraphAdapter

**功能描述**：处理边界tensor，插入view/assemble打通子图间数据通路

**常见错误**：
- 内存类型冲突
- 无效内存类型
- 生产者类型错误
- 生产者输入错误
- 跨核move操作错误

**可能原因**：
- 边界tensor的original和tobe内存类型不一致
- 内存类型不在[UB, L1, DDR, L0C]范围内
- tensor有多个生产者但不是OP_ASSEMBLE/OP_COPY_OUT
- OP_ASSEMBLE输入操作数不为1
- 跨核move操作出现在子图开头

---

### 21. GenerateMoveOp

**功能描述**：将view/assemble/convert/duplicate操作转换为具体的移动操作

**常见错误**：
- 内存路径查找失败
- UB2L1转换失败
- L0C2L1属性设置失败

**可能原因**：
- 内存路径不支持
- ND到NZ格式转换失败
- 内存层次结构复杂

---

### 22. CommonOperationEliminate

**功能描述**：通用操作消除（通过hash检测等价操作并消除冗余操作）

**常见错误**：
- 操作等价判断错误
- hash计算失败
- 图连接更新失败
- 死操作消除失败

**可能原因**：
- 操作属性不一致
- 输入输出tensor不匹配
- shape或dtype不匹配
- tensor生产者或消费者为空
- OUTCAST节点不支持消除
- View操作不支持消除
- dontTouch属性操作不支持消除
- 操作序列排序失败
- 消费者连接更新失败
- offset计算失败（view/copy操作）

---

### 23. AxisCombine

**功能描述**：轴组合（处理广播对齐，对尾轴做对齐处理，插入BRCB/EXPAND操作）

**常见错误**：
- 广播对齐失败
- 轴组合失败
- Shape变换失败
- 内存布局错误

**可能原因**：
- 广播操作输入shape不匹配
- 尾轴对齐值计算失败
- BRCB/EXPAND操作插入失败
- 轴组合标记器运行失败
- 配置未启用combineAxis

---

### 24. PadLocalBuffer

**功能描述**：填充本地缓冲区（对尾轴做32B/16B/64B对齐，处理matmul、vector、broadcast、transpose等场景）

**常见错误**：
- 填充计算失败
- 缓冲区重分配失败
- 对齐验证失败
- 数据类型检查失败
- offset计算失败
- 属性设置失败

**可能原因**：
- 对齐要求复杂
- 缓冲区大小限制
- 填充策略冲突
- matmul输入shape维度不足
- L1转换场景转换失败
- B8/B4数据类型对齐失败
- broadcast轴处理失败
- transpose处理失败
- Cube/Vector类型判断错误
- 内存类型不匹配

---

### 25. RemoveUnalignedReshape

**功能描述**：删除未对齐的reshape（移除尾轴非对齐的reshape，插入copy_out/copy_in操作）

**常见错误**：
- 对齐检测失败
- Reshape优化失败
- 边界处理错误
- 动态reshape替换失败
- copy操作插入失败
- offset计算错误
- tensor插入失败
- 消费者更新失败
- 动态状态复制失败
- shape检查失败

**可能原因**：
- 对齐要求不满足
- 边界条件复杂
- 内存布局冲突
- 输入输出shape维度不匹配
- rawshape与shape不一致
- 内存类型不是UB
- 动态shape变化轴不支持
- offset维度不一致

---

### 26. ReplaceTensor

**功能描述**：替换tensor（处理inplace操作，替换tensor引用，更新offset和连接）

**常见错误**：
- Tensor替换失败
- 引用更新失败
- 内存一致性错误
- 地址冲突检查失败
- assemble冲突检查失败
- index_outcast冲突检查失败
- reshape冲突检查失败
- a_mulacc_b冲突检查失败
- inplace检查失败
- 前向/后向处理失败
- offset调整失败
- copy操作插入失败

**可能原因**：
- Tensor类型不匹配款
- 引用链断裂
- 内存类型冲突
- 输入输出tensor地址冲突
- rawmagic/memoryId不匹配
- index_outcast输入输出冲突
- reshape shape首轴外不一致
- a_mulacc_b inplace关系不匹配
- tensor生产者消费者为空
- offset计算失败

---

### 27. PreGraphProcess

**功能描述**：预图处理（包括颜色图预处理、设置tensor边界、设置copy属性、删除冗余assemble、更新cube属性、合并view/assemble）

**常见错误**：
- 预处理初始化失败
- 状态设置失败
- 依赖分析错误
- 颜色图预处理失败
- tensor颜色初始化失败
- copy操作cube属性更新失败
- 边界设置失败
- 特殊copy操作处理失败
- 冗余assemble删除失败
- cube操作更新失败
- view/assemble合并失败

**可能原因**：
- 初始化顺序错误
- 状态冲突
- 依赖关系不完整
- subgraphID设置失败
- cube属性获取失败
- copy操作类型判断错误

---

### 28. InferDynShape

**功能描述**：推断动态shape（按照入度解依赖顺序遍历每个op调用对应的infershape函数）

**常见错误**：
- 动态shape推断失败
- 符号表达式错误
- 约束求解失败
- 拓扑排序失败
- 有效shape重置失败
- 参数索引更新失败

**可能原因**：
- 动态维度关系复杂
- 符号计算溢出
- 约束不一致
- 操作序列构建失败
- 输入输出图构建失败
- view/assemble/reshape有效shape重置失败
- 地址到有效shape映射更新失败

---

### 29. SubgraphToFunction

**功能描述**：子图转函数（将子图转换为独立函数，构建incast/outcast连接，处理参数映射和符号化）

**常见错误**：
- 子图转换失败
- 函数创建失败
- 接口映射错误
- view转copy_in失败
- 数据依赖插入失败
- 数据依赖清除失败
- copy_in转view恢复失败
- 图构建失败
- incast/outcast记录失败
- 参数映射构建失败
- island转换失败
- nList构造失败
- 参数插入失败
- 符号化失败
- 符号名称查找失败

**可能原因**：
- 子图结构复杂
- 接口不匹配
- 调用约定错误
- 静态流程处理器设置失败
- 函数类型不支持
- subgraphID设置失败
- tensor数据依赖记录失败
- producer连接记录失败
- inplace语义处理失败
- 子图边界判断失败
- 内存类型不匹配
- producer为空

---

### 30. InferParamIndex

**功能描述**：推断参数索引（推断参数索引和更新有效shape，处理view/assemble/reshape的动态有效shape）

**常见错误**：
- 参数索引推断失败
- 访问模式分析失败
- 边界检查错误
- 有效shape重置失败
- 参数索引更新失败
- shape推断失败
- dump失败

**可能原因**：
- 参数数量不匹配
- 索引越界
- 访问模式不规则
- 操作序列构建失败
- 输入输出图构建失败
- view/assemble/reshape有效shape重置失败
- 地址到有效shape映射更新失败
- 子函数有效shape设置失败

---

### 31. SrcDstBufferMerge

**功能描述**：源目标缓冲区合并（处理inplace重用、L0内存重用，合并源目标缓冲区）

**常见错误**：
- 缓冲区分析失败
- 合并策略失败
- 传输优化错误
- 初始化失败
- tensor最大大小初始化失败
- 操作有效性检查失败
- inplace检查失败
- 可重用查找失败
- L0内存重用处理失败
- inplace重用处理失败

**可能原因**：
- 缓冲区大小不匹配
- 传输方向冲突
- 内存重叠
- 操作为空
- 输入输出数量不匹配
- inplaceInfo属性获取失败
- 输入输出索引越界
- L0到L0C/L0C到L1传输判断失败
- assemble重用检查
- 内存类型不匹配
- consumer集合插入失败
- tensor大小计算失败
- opList为空
- memoryrange初始化失败
- excludeBufferReuse属性存在

---

### 32. AddAlloc

**功能描述**：添加分配操作（按color判断是否需要插入alloc，生成各类alloc节点：L0A_ALLOC/UB_ALLOC/REG_ALLOC/L0B_ALLOC/L0C_ALLOC/L1_ALLOC/BT_ALLOC/FIX_ALLOC）

**常见错误**：
- 分配操作插入失败
- 内存大小计算失败
- 生命周期分析失败
- alloc操作生成失败
- tensor分配消息更新失败
- 分配消息查找失败
- alloc操作码生成失败
- tensor分配消息映射生成失败
- 分配消息设置失败

**可能原因**：
- 内存不足
- 分配策略错误
- 生命周期管理复杂
- 内存类型不支持
- opcode映射失败
- program为空
- alloc节点创建失败
- 属性设置失败

---

### 33. OoOSchedule

**功能描述**：OOO调度（执行乱序调度，支持非混合图和混合图调度，处理依赖关系和资源分配）

**常见错误**：
- 调度分析失败
- 依赖关系处理失败
- 资源分配冲突
- 调度失败
- 任务列表排序失败
- 延迟估计失败
- 核心调度失败
- 任务合并失败
- 边界顺序修改失败
- advance alloc失败
- AICPU程序检测失败
- workspace大小设置失败
- tensor消费者生产者更新失败
- 死操作消除失败

**可能原因**：
- 依赖关系复杂
- 资源约束严格
- 调度窗口限制
- 混合图判断错误
- 任务图构建失败
- 任务节点处理失败
- 目标核心类型映射错误
- 边界操作判断错误
- alloc操作查找失败
- 操作旋转失败

---

### 34. TuneTileOpSeqForVF

**功能描述**：为VF调整tile操作序列（DAV_3510平台专用，优化向量融合场景的操作序列）

**常见错误**：
- 序列分析失败
- 调整策略失败
- 性能预测错误
- 操作序列调整失败
- pipe vector索引查找失败
- 操作合并性判断失败
- 操作移动失败

**可能原因**：
- 操作依赖复杂
- 向量资源限制
- 性能模型不准确
- 不支持的平台
- group合并判断失败
- set_flag/wait_flag调整失败
- pipe操作映射构建失败

---

### 35. RemoveAlloc

**功能描述**：删除分配操作（删除alloc节点）

**常见错误**：
- 分配操作分析失败
- 冗余检测失败
- 生命周期验证错误

**可能原因**：
- 分配操作链复杂
- 生命周期重叠
- 使用分析不完整

---

### 36. CopyOutResolve

**功能描述**：Copy输出解析（解析copy_out操作，处理outcast后的copy_out合并）

**常见错误**：
- Copy-out分析失败
- 传输路径优化失败
- 输出处理错误
- copy_out resolve插入失败
- outcast最后copy_out查找失败
- outcast生产者检查失败

**可能原因**：
- 输出依赖复杂
- 传输路径阻塞
- 内存类型转换
- leaf函数处理失败
- copy_out resolve coalescing参数错误

---

### 37. InsertSync

**功能描述**：插入同步操作（插入set_flag/wait_flag同步操作，处理数据依赖和管道依赖）

**常见错误**：
- 同步点分析失败
- 同步操作插入失败
- 一致性验证失败
- 管道同步失败
- 操作序列调整失败
- view/assemble顺序处理失败
- 同步操作生成失败
- 依赖关系添加失败
- 事件ID获取失败
- 数据依赖搜索失败
- 死锁检测失败
- 同步操作发射失败
- oplog生成失败
- 依赖信息获取失败
- 假数据依赖松弛失败
- 数据依赖检查失败
- 范围搜索树插入失败
- 区间树操作失败
- tensor范围构建失败

**可能原因**：
- 同步依赖复杂
- 一致性要求严格
- 性能开销过大
- 管道类型判断错误
- 依赖操作dump失败
- 管道调度失败
- issue队列处理失败
- 事件ID用尽
- 内存缓冲区重叠检查
- WAW/WAR/RAW依赖检查失败
- issue op检查失败
- 管道阶段映射失败
- set_flag/wait_flag生成失败
- 同步源日志索引获取失败
- 最大事件ID获取失败
- view/assemble重排序失败
- 管道依赖map更新失败

---

### 38. TuneSyncForVF

**功能描述**：为VF调整同步（DAV_3510平台专用，优化向量融合场景的同步操作）

**常见错误**：
- 同步分析失败
- 优化策略失败
- 正确性验证失败
- 操作序列调整失败
- set_flag/wait_flag调整失败
- pipe操作映射生成失败
- pipe vector索引查找失败
- 操作合并性判断失败
- 操作移动失败
- pipe vector时间更新失败
- set pipe时间更新失败
- wait pipe时间更新失败
- maxMoveBackDist计算失败

**可能原因**：
- 同步粒度不当
- 依赖关系遗漏
- 性能-正确性权衡
- 不支持的平台
- group数量计算错误
- vector tileop索引越界
- set_flag/wait_flag列表处理失败
- mergedOps为空
- pipe操作map构建失败

---

### 39. MixSubgraphSplit

**功能描述**：混合子图拆分（将Mix子图拆分为多个独立的Cube和Vector子图，重新分配subgraphID）

**常见错误**：
- 子图拆分失败
- 混合模式分析失败
- 资源分配错误
- 子图信息收集失败
- 拆分计算失败
- 拆分执行失败
- 新函数生成失败
- 拆分结果应用失败
- 原始Mix callOp删除失败
- leaf函数处理失败
- 组件信息记录失败
- 最终依赖应用失败
- incast依赖应用失败
- outcast依赖应用失败

**可能原因**：
- 子图耦合紧密
- 资源竞争
- 拆分边界不当
- Mix子图判断失败
- 资源类型获取失败
- programID映射更新失败
- mix子图新ID记录失败
- callOp删除失败
- incast/outcast参数记录失败
- producer连接记录失败
- tensor到函数映射失败
- 有效shape设置失败
- 子函数invoke信息记录失败
- 静态处理器设置失败
- split结果记录失败
- split记录验证失败
- 组件信息为空
- split函数数量与组件数量不匹配

---

### 40. GlobalMemoryReuse

**功能描述**：全局内存重用（处理root casts初始化、操作处理、tensor收集、连接矩阵构建、存储分配、leaf函数全局内存重用）

**常见错误**：
- 内存使用分析失败
- 重用策略失败
- 生命周期冲突
- tensor bucket更新失败
- tensor group添加失败
- topology依赖检查失败
- raw tensor资格检查失败
- leaf函数输出输入重用映射获取失败
- parent op扫描失败
- 重用op检查失败
- 可重用输入查找失败
- raw shape计算失败
- stride存储offset计算失败
- 存储offset获取失败
- 存储更新失败
- best fit bucket查找失败
- 新bucket处理失败
- tensor magic到bucket索引更新失败
- bucket size到索引更新失败
- bucket索引到size更新失败
- tensor consumer无重叠标记失败
- tensor bucket获取失败
- 可复用L0 tensor查找失败
- L0内存重用处理失败
- 存储需要分配预处理失败
- 存储ID更新失败
- 非重叠consumer tensor标记失败
- leaf函数全局内存重用初始化失败
- 输出tensor收集失败
- 输入tensor收集失败
- leaf函数全局内存重用处理失败
- 输出全局内存重用处理失败
- incast outcast更新失败

**可能原因**：
- 内存访问模式复杂
- 生命周期重叠
- 重用机会识别困难
- tensor组为空
- tensor大小计算失败
- raw data size不匹配
- 数据类型不匹配
- 维度数量不匹配
- 非最高维度值不匹配
- leaf函数未找到
- copy_in操作判断失败
- candidate未找到
- count/size/usage/rawReuseCompatible检查失败
- copy_in输入为空
- parent op遍历失败
- tensor已访问
- producer为空
- storage offset计算失败
- raw shape/offset/valid shape计算失败
- tensor size为0
- bucket大小不足
- consumer op检查失败
- bucket map为空
- tensor magic不存在

---

### 41. LoopaxesProc

**功能描述**：循环轴处理（DAV_3510平台专用，处理循环轴分组和标记，支持向量融合）

**常见错误**：
- 循环分析失败
- 轴处理失败
- 边界条件错误
- 函数循环轴更新失败
- 操作循环轴更新失败
- 状态清除失败
- 循环轴一致性检查失败
- 属性设置失败
- 循环组结束标记失败

**可能原因**：
- 循环结构复杂
- 边界条件不规则
- 依赖关系嵌套
- 不支持的平台
- VF未启用
- VF mark for未启用
- rootFunc为空
- subProgram为空
- shape维度小于等于2
- EXPAND轴判断失败
- loopAxes属性获取失败
- groupIdx递增溢出
- previousOutputMagic与input magic不匹配
- lastOpInLoop为空
- 属性获取失败
- 输入输出为空
- 符号表达式构建失败
- 动态参数表查找失败
- 参数信息replacedSymbol为空
- 符号表达式不相等

---

### 42. CodegenPreproc

**功能描述**：代码生成预处理（更新alloc节点operand、强制轴组合、处理axis combine、设置needAlloc属性、修复expand dim）

**常见错误**：
- 预处理验证失败
- 代码生成准备失败
- 最终一致性检查失败
- GM tensor参数索引保存失败
- 强制轴组合失败
- 轴组合处理失败
- 属性设置失败
- 轴组合失败
- 尾轴处理失败
- last轴处理失败
- 操作轴处理失败
- needAlloc属性设置失败
- expand dim修复失败
- op list dump失败

**可能原因**：
- 图状态不一致
- 资源分配冲突
- 生成约束不足
- 非动态函数
- copy in/out操作判断失败
- gather/load操作处理
- op属性获取失败
- 属性大小与操作数不匹配
- shape size不足
- shape处理失败
- combine axis配置未启用
- forceCombineAxis配置未启用
- input/output_combine_axis属性获取失败
- process axis失败
- copy in/out属性设置失败
- rawshape处理失败
- UB copy判断失败
- reduce操作判断失败
- reduce axis属性获取失败
- expand dim属性获取失败
- expand dim调整失败
- DAV_3510平台判断
- skip操作判断
- input/output shape判断
- rawshape尾轴判断
- reduce轴判断失败
- memoryrange.memId获取失败
- needAlloc属性获取失败
- appearedMemId查找失败
- opCode转换失败

---

## 使用说明

本文档作为 PyPTO Pass 模块错误定位的参考手册，当遇到 Pass 模块错误时：

1. 根据错误日志中的 Pass 模块名称查找对应的条目
2. 查看"常见错误"部分，了解可能出现的错误类型
3. 查看"可能原因"部分，分析错误产生的根本原因
4. 结合具体错误信息和堆栈跟踪，定位问题代码位置
5. 参考可能原因进行问题修复

## 注意事项

- 不同版本的 PyPTO 可能存在 Pass 模块差异
- 部分错误可能由多个 Pass 模块共同导致
- 错误信息中的行号和文件路径是定位问题的关键
- 建议结合代码审查和单元测试进行问题修复
