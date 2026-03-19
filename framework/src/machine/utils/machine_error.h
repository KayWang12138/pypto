/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file machine_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
/** 顶层阶段：自 HOST_BACKEND=0x70000 起每项 +0x1000，与下方同名 *Err 子枚举码段对齐 */
enum class MachineError : uint32_t {
    HOST_BACKEND         = 0x70000U, // Host 后端（总段标识）
    HOST_LAUNCHER        = 0x71000U, // Host 启动器（总段标识）
    SCHEDULE             = 0x72000U, // 调度链路
    CONTROL_FLOW         = 0x73000U, // 控制流执行
    WORKSPACE            = 0x74000U, // Workspace / Slab
    DUMP_DFX             = 0x75000U, // Dump / DFX / Profiling
    PROGRAM_ENCODE       = 0x76000U, // 编解码与一致性
    TENSOR_META          = 0x77000U, // 张量元信息
    SERVER_KERNEL        = 0x78000U, // AICPU server / kernel
    THREAD_MACHINE       = 0x79000U, // 线程/机器级
    DATA_STRUCTURE       = 0x7A000U, // 内部数据结构
    UNKNOWN              = 0x7F000U, // 未知/预留
};

/** Host 编译/链接/控制 bin/函数缓存等；子码 0x700xx，与 MachineError::HOST_BACKEND（0x70000）同段。 */
enum class HostBackEndErr : uint32_t {
    COMPILE_AICORE_CMD_FAILED         = 0x70001U, // 生成/执行 AICore 编译命令失败
    COMPILE_CCEC_FAILED               = 0x70002U, // ccec 编译失败
    OPEN_CALL_HEADER_FAILED           = 0x70003U, // 打开 call header 失败
    LINK_CORE_MACHINE_CMD_FAILED      = 0x70004U, // 链接 core machine 命令失败
    LINK_KERNEL_FAILED                = 0x70005U, // 链接 kernel 失败
    NO_CCE_TOOLCHAIN_PATH             = 0x70006U, // 未找到 CCE 工具链路径
    GENERATE_AICORE_SRC_FAILED        = 0x70007U, // 生成 AICore 源码失败
    CONSTRUCT_CUSTOM_OP_JSON_FAILED   = 0x70008U, // 构造 custom op json 失败
    PRECOMPILE_FILE_FAILED            = 0x70009U, // 预编译文件失败
    RUN_DEVICE_MACHINE_COMPILE_FAILED = 0x7000AU, // device machine 编译流程失败
    GEN_DYNAMIC_OP_FAILED             = 0x7000BU, // 生成动态算子失败
    OP_PRECOMPILE_FAILED              = 0x7000CU, // 算子预编译失败
    MACHINE_TASK_OR_FUNCTION_NULL     = 0x7000DU, // machine task 或 function 为空
    FUNCTION_CACHE_HASH_MISS          = 0x7000EU, // 函数缓存 hash 未命中/不一致
    LOOP_OPERATION_COUNT_LIMIT_EXCEEDED = 0x7000FU, // 循环内 operation 数超限
    CREATE_AICPU_COMPILE_DIR_FAILED   = 0x70010U, // 创建 AICPU 编译目录失败
    DUPLICATE_LEAF_FUNC_HASH          = 0x70011U, // leaf 函数 hash 重复
    COMPILE_DYNAMIC_AICORE_OBJECT_FAILED = 0x70012U, // 动态 AICore 目标编译失败
    ENV_HOME_MISSING_OR_EMPTY         = 0x70013U, // 环境变量 HOME 缺失或为空
    CACHE_DIR_CREATE_FAILED           = 0x70014U, // 缓存目录创建失败
    FUNC_CACHE_BIN_PATH_NOT_FOUND     = 0x70015U, // 函数缓存 bin 路径不存在
    UNKNOWN                           = 0x70099U  // 未归类/兜底
};

/** Host 运行期：RT/ACL、launcher、runner、设备内存池、AICPU json 等；子码 0x710xx，与 MachineError::HOST_LAUNCHER（0x71000）同段。 */
enum class HostLauncherErr : uint32_t {
    RT_MEMSET_FAILED                    = 0x71001U, // rtMemset 失败
    ALLOC_DEV_DFX_INFO_FAILED           = 0x71002U, // 分配设备 DFX 信息区失败
    READ_BACKEND_SERVER_SO_FAILED       = 0x71003U, // 读取 backend server 动态库失败
    AICPU_SERVER_INIT_FAILED            = 0x71004U, // AICPU server 初始化失败
    ACLRT_CREATE_EVENT_FAILED           = 0x71005U, // aclrtCreateEvent 失败
    ACLRT_RECORD_EVENT_FAILED           = 0x71006U, // aclrtRecordEvent 失败
    ACLRT_STREAM_WAIT_EVENT_FAILED      = 0x71007U, // stream 等待 event 失败
    LAUNCH_AICPU_FAILED                 = 0x71008U, // 启动 AICPU 任务失败
    LAUNCH_PREPARE_FAILED               = 0x71009U, // launch 准备阶段失败
    LAUNCH_CUSTOM_AICPU_FAILED          = 0x7100AU, // 启动 custom AICPU 失败
    LAUNCH_AICORE_FAILED                = 0x7100BU, // 启动 AICore 失败
    LAUNCH_BUILTIN_OP_NULL_FAILED       = 0x7100CU, // 内置算子句柄/函数为空
    DEVICE_RUNNER_PREPARE_FAILED        = 0x7100DU, // device runner 准备失败
    REGISTER_KERNEL_BIN_FAILED          = 0x7100EU, // 注册 kernel bin 失败
    GET_BUILTIN_OP_FUNC_HANDLE_FAILED   = 0x7100FU, // 获取内置算子函数句柄失败
    PREPARE_ARGS_FAILED                 = 0x71010U, // kernel 参数准备失败
    ACLMDL_RI_CAPTURE_GET_INFO_FAILED   = 0x71011U, // 捕获流信息获取失败
    STREAM_CAPTURE_STATUS_UNSUPPORTED   = 0x71012U, // 当前流捕获状态不支持
    RT_MODEL_NULL                       = 0x71013U, // RT model 为空
    RT_STREAM_ADD_TO_MODEL_FAILED       = 0x71014U, // stream 加入 model 失败
    ADX_DUMP_SERVER_INIT_SYMBOL_MISSING = 0x71015U, // ADX dump server 初始化符号缺失
    ADX_DUMP_SERVER_INIT_FAILED         = 0x71016U, // ADX dump server 初始化失败
    ADX_DUMP_SERVER_UNINIT_SYMBOL_MISSING = 0x71017U, // ADX dump server 反初始化符号缺失
    ADX_DUMP_SERVER_UNINIT_FAILED       = 0x71018U, // ADX dump server 反初始化失败
    CTRL_FLOW_CACHE_MALLOC_FAILED       = 0x71019U, // 控制流 cache 设备内存分配失败
    CTRL_FLOW_CACHE_MEMCPY_FAILED       = 0x7101AU, // 控制流 cache 拷贝失败
    GET_CAPTURE_INFO_FAILED             = 0x7101BU, // 获取 capture 信息失败
    REGISTER_KERNEL_FAILED              = 0x7101CU, // 注册 kernel 失败
    UNREGISTER_KERNEL_FAILED            = 0x7101DU, // 注销 kernel 失败
    TRIPLE_STREAM_NR_AICPU_ADJUSTED     = 0x7101EU, // 三流场景下 AICPU 数量被调整（告警类）
    STREAM_SYNC_FAILED                  = 0x7101FU, // stream 同步失败
    GET_DEVICE_ID_FAILED                = 0x71020U, // 获取 device id 失败
    ALLOC_DEV_ADDR_IN_POOL_FAILED       = 0x71021U, // 内存池中分配设备地址失败
    HAL_FUNCTION_NOT_FOUND              = 0x71022U, // HAL 函数未找到
    MAP_REG_ADDR_FAILED                 = 0x71023U, // 寄存器地址映射失败
    RT_MALLOC_FAILED                    = 0x71024U, // rtMalloc 失败
    RT_MEMCPY_FAILED                    = 0x71025U, // rtMemcpy 失败
    MEM_POOL_2MB_BLOCK_ALLOC_FAILED     = 0x71026U, // 内存池 2MB 块分配失败
    MEM_POOL_2MB_FREE_NOT_ALLOWED       = 0x71027U, // 不允许释放 2MB 池块
    MEM_POOL_DEVADDR_NULL               = 0x71028U, // 内存池设备地址为空
    MEM_POOL_ALLOCATE_FAILED            = 0x71029U, // 内存池分配失败
    MEM_POOL_FREE_NULLPTR               = 0x7102AU, // 释放空指针
    MEM_POOL_FREE_UNKNOWN_PTR           = 0x7102BU, // 释放未托管指针
    MEM_POOL_CHECK_ALL_SENTINELS_FAILED = 0x7102CU, // 哨兵校验失败
    MEM_POOL_CORRUPTION_DETAIL          = 0x7102DU, // 内存池损坏详情
    MEM_POOL_BASE_ADDR_NOT_FOUND        = 0x7102EU, // 未找到基址对应块
    MEM_POOL_SENTINEL_VERIFY_FAILED     = 0x7102FU, // 哨兵验证失败
    MEM_POOL_ALL_STRATEGIES_FAILED      = 0x71030U, // 各分配策略均失败
    LOAD_AICPU_CUSTOM_OP_JSON_FAILED    = 0x71031U, // 加载 AICPU custom op json 失败
    CUSTOM_OP_JSON_PATH_EMPTY           = 0x71032U, // custom op json 路径为空
    LOAD_AICPU_JSON_FAILED              = 0x71033U, // 加载 AICPU json 失败
    GET_OP_TYPE_FUNC_HANDLE_FAILED      = 0x71034U, // 按 op type 取函数句柄失败
    GET_BUILTIN_BIN_HANDLE_FAILED       = 0x71035U, // 获取内置 bin 句柄失败
    GET_BUILTIN_FUNC_HANDLE_FAILED      = 0x71036U, // 获取内置函数句柄失败
    INVALID_CUSTOM_FUNC_NAME            = 0x71037U, // 非法 custom 函数名
    DEPRECATED_USER_DYNAMIC_WORKSPACE   = 0x71038U, // 已废弃的用户动态 workspace 用法
    TEST_ALLOC_DEV_MALLOC_FAILED        = 0x71039U, // 测试路径设备 malloc 失败
    TEST_ACL_INIT_FAILED                = 0x7103AU, // 测试路径 ACL 初始化失败
    TEST_DUMP_TENSOR_DEBUG              = 0x7103BU, // 测试/调试 tensor dump
    UNKNOWN                             = 0x71099U  // 未归类/兜底
};

/** 调度链路：prefetch、多核任务等待、握手、队列与异常复位；子码 0x720xx，与 MachineError::SCHEDULE 同段。 */
enum class SchedErr : uint32_t {
    PREFETCH_CHECK_FAILED     = 0x72001U, // prefetch 检查失败
    AIC_TASK_WAIT_TIMEOUT     = 0x72002U, // AIC 任务等待超时
    AIV_TASK_WAIT_TIMEOUT     = 0x72003U, // AIV 任务等待超时
    TAIL_TASK_WAIT_TIMEOUT    = 0x72004U, // tail 任务等待超时
    ALL_AICORE_SYNC_TIMEOUT   = 0x72005U, // 全 AICore 同步超时
    HANDSHAKE_TIMEOUT         = 0x72006U, // 线程握手超时
    READY_QUEUE_OVERFLOW      = 0x72007U, // 就绪队列溢出
    SIGNAL_QUEUE_OVERFLOW     = 0x72008U, // 信号队列溢出
    QUEUE_DEQUEUE_WHEN_EMPTY  = 0x72009U, // 空队列出队
    CORE_TASK_PROCESS_FAILED  = 0x72010U, // core 任务处理失败
    AICPU_TASK_SYNC_TIMEOUT   = 0x72011U, // AICPU 任务同步超时
    EXCEPTION_RESET_TRIGGERED = 0x72012U, // 异常触发复位流程
    EXCEPTION_SIGNAL_RECEIVED = 0x72013U, // 收到异常信号
    THREAD_INIT_ARGS_INVALID  = 0x72014U, // 线程初始化参数非法
    UNKNOWN                   = 0x72099U  // 未归类/兜底
};

/** 控制流执行：root ctx、device task 构建、就绪队列与依赖 dump；子码 0x730xx。 */
enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED     = 0x73001U, // 控制流执行失败
    ROOT_ALLOC_CTX_NULL       = 0x73002U, // root alloc 上下文为空
    ROOT_STITCH_CTX_NULL      = 0x73003U, // root stitch 上下文为空
    SYNC_FLAG_WAIT_TIMEOUT    = 0x73004U, // 同步标志等待超时
    DEVICE_TASK_BUILD_FAILED  = 0x73005U, // device task 构建失败
    READY_QUEUE_INIT_FAILED   = 0x73006U, // 就绪队列初始化失败
    DEP_DUMP_FAILED           = 0x73007U, // 依赖关系 dump 失败
    READY_QUEUE_DUMP_FAILED   = 0x73008U, // 就绪队列 dump 失败
    TASK_STATS_ABNORMAL       = 0x73009U, // 任务统计异常
    UNKNOWN                   = 0x73099U  // 未归类/兜底
};

/** Workspace / Slab：缓存、类型、初始化与地址范围；子码 0x740xx。 */
enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED       = 0x74001U, // slab 添加 cache 失败
    SLAB_STAGE_LIST_INCONSISTENT = 0x74002U, // slab stage 列表不一致
    SLAB_TYPE_INVALID           = 0x74003U, // slab 类型非法
    WORKSPACE_INIT_RESOURCE_ERROR = 0x74004U, // workspace 初始化资源错误
    WORKSPACE_INIT_PARAM_INVALID  = 0x74005U, // workspace 初始化参数非法
    WS_TENSOR_ADDRESS_OUT_OF_RANGE = 0x74006U, // tensor 地址越界
    SLAB_CAPACITY_CALC_INVALID  = 0x74007U, // slab 容量计算非法
    UNKNOWN                     = 0x74099U  // 未归类/兜底
};

/** Dump / DFX / Profiling：拷贝、tensor 信息、指标与 trace；子码 0x750xx。 */
enum class DumpDfxErr : uint32_t {
    DUMP_MEMCPY_FAILED            = 0x75001U, // dump 过程 memcpy 失败
    DUMP_TENSOR_INFO_FAILED       = 0x75002U, // dump tensor 元信息失败
    DUMP_TENSOR_DATA_FAILED       = 0x75003U, // dump tensor 数据失败
    METRIC_ALLOC_OR_WAIT_TIMEOUT  = 0x75004U, // 指标缓冲分配或等待超时
    PERF_TRACE_FORMAT_ERROR       = 0x75005U, // 性能 trace 格式错误
    PERF_TRACE_DUMP_ERROR         = 0x75006U, // 性能 trace 落盘错误
    DFX_AICPU_TIMEOUT            = 0x75007U, // DFX 路径 AICPU 超时
    UNKNOWN                       = 0x75099U  // 未归类/兜底
};

/** Program 编解码：stitch、reloc、program range、cell match；子码 0x760xx。 */
enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = 0x76001U, // 动态函数数据对齐错误
    FUNC_OP_SIZE_MISMATCH        = 0x76002U, // func op 大小不一致
    STITCH_PRED_SUCC_MISMATCH     = 0x76003U, // stitch 前驱后继不一致
    STITCH_LIST_TOO_LARGE         = 0x76004U, // stitch 列表过大
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = 0x76005U, // stitch handle 索引越界
    CELL_MATCH_PARAM_INVALID     = 0x76006U, // cell match 参数非法
    PROGRAM_RANGE_VERIFY_FAILED   = 0x76007U, // program 范围校验失败
    CACHE_RELOC_KIND_INVALID      = 0x76008U, // cache reloc 类型非法
    ADDR_OFFSET_RAW_MAGIC_MISMATCH = 0x76009U, // 地址偏移与 raw magic 不匹配
    CALL_OP_COUNT_EXCEEDS_UINT16_MAX = 0x7600AU, // call op 数量超过 uint16 上限
    CELL_MATCH_DIM_ZERO          = 0x7600BU, // cell match 维度为 0
    ASSEMBLE_STITCH_MEMORY_EXCESS = 0x7600CU, // assemble stitch 内存超限
    LEAF_CALLEE_ATTR_NULL        = 0x7600DU, // leaf 被调属性为空
    UNKNOWN                       = 0x76099U  // 未归类/兜底
};

/** 张量元信息：维度、编码指针、索引与 shape；子码 0x770xx。 */
enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED   = 0x77001U, // 维度个数超限
    TENSOR_ENCODE_PTR_MISMATCH  = 0x77002U, // 编码指针不一致
    RAW_TENSOR_INDEX_OUT_OF_RANGE = 0x77003U, // raw tensor 索引越界
    SHAPE_VALUE_MISMATCH        = 0x77004U, // shape 数值不一致
    TENSOR_DUMP_INFO_INCONSISTENT = 0x77005U, // dump 信息与 tensor 不一致
    UNKNOWN                      = 0x77099U  // 未归类/兜底
};

/** AICPU 动态 server / kernel：so、符号加载与执行；子码 0x780xx。 */
enum class ServerKernelErr : uint32_t {
    DYN_SERVER_ARGS_NULL      = 0x78001U, // 动态 server 参数为空
    DYN_SERVER_SAVE_SO_FAILED  = 0x78002U, // 保存 so 失败
    KERNEL_EXEC_FUNC_FAILED    = 0x78003U, // kernel 执行函数失败
    KERNEL_SO_OR_FUNC_LOAD_FAILED = 0x78004U, // so 或函数加载失败
    DYN_SERVER_RUN_FAILED      = 0x78005U, // 动态 server 运行失败
    DYN_SERVER_INIT_FAILED     = 0x78006U, // 动态 server 初始化失败
    UNKNOWN                    = 0x78099U  // 未归类/兜底
};

/** 线程/设备机级：参数、信号处理、全局 reset；子码 0x790xx。 */
enum class ThreadErr : uint32_t {
    DEVICE_ARGS_INVALID     = 0x79001U, // 设备参数非法
    SIGNAL_HANDLER_ABNORMAL = 0x79002U, // 信号处理异常
    RESET_REG_ALL_TRIGGERED = 0x79003U, // 触发全量寄存器复位
    UNKNOWN                 = 0x79099U  // 未归类/兜底
};

/** 内部数据结构：reloc 向量与小数组边界；子码 0x7A0xx。 */
enum class DataStructErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = 0x7A001U, // 设备 reloc 向量索引越界
    SMALL_ARRAY_RESIZE_OOB     = 0x7A002U, // 小数组 resize 越界
    UNKNOWN                    = 0x7A099U  // 未归类/兜底
};

/** Machine 与 Function 交集错误码：当前仅占位 RESERVED。 */
enum class MachineFunctionErr : uint32_t {
    RESERVED = 0x87000U  // 预留
};

/** Machine 与 Pass 交集错误码：当前仅占位 RESERVED。 */
enum class MachinePassErr : uint32_t {
    RESERVED = 0x87500U  // 预留
};

/** Machine 与 Codegen 交集错误码：当前仅占位 RESERVED。 */
enum class MachineCodegenErr : uint32_t {
    RESERVED = 0x88000U  // 预留
};

/** Machine 与 Simulation 交集：仿真设备内存与控制流 cache 分配等。 */
enum class MachineSimulationErr : uint32_t {
    SIM_ALLOC_DEVICE_MEMORY_FAILED   = 0x88501U, // 仿真路径分配设备内存失败
    SIM_ALLOC_CTRL_FLOW_CACHE_FAILED = 0x88502U, // 仿真路径分配控制流 cache 失败
    UNKNOWN                          = 0x88599U  // 未归类/兜底
};

/** Machine 与 Distributed 交集错误码：当前仅占位 RESERVED。 */
enum class MachineDistributedErr : uint32_t {
    RESERVED = 0x89000U  // 预留
};

/** Machine 与 Operation 交集错误码：当前仅占位 RESERVED。 */
enum class MachineOperationErr : uint32_t {
    RESERVED = 0x89500U  // 预留
};

}  // namespace npu::tile_fwk
