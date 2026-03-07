# MACHINE 组件错误码

MACHINE 错误码的范围、Category/Scene 定义及具体码值均以头文件为准，本文档不再重复罗列。

**头文件（权威定义）**：`framework/src/machine/utils/machine_error.h`

头文件内包含：`MachineErrorCategory`、各 `*ErrorScene` 枚举（如 `SchedErrorScene`、`WorkspaceErrorScene`）、`ToMachineErrorCode`、`GetMachineErrorCategory`、`IsMachineIntersectionErrorCode` 等，详见源码注释与实现。

---

## F70006 HANDSHAKE_TIMEOUT 排查说明

**含义**：AICPU 与 AICore/AIV 握手超时。调度线程在设备模式下调用了 `HandShake()`，在约定时间内未完成握手即报此错。

**典型报错位置**：`framework/src/machine/device/dynamic/aicore_manager.h` 中调度 run 初始化阶段（`HandShake()` 返回非 OK 时打印 `"hand shake timeout."`）。

**建议排查步骤**：

1. **确认设备与驱动**：NPU 设备可用、驱动正常，`npu-smi info` 无异常。
2. **确认资源与负载**：当前进程/容器内 NPU 占用是否过高，是否存在多进程争用同一设备。
3. **确认超时配置**：若存在握手/同步超时配置项，检查是否过短或与环境不符。
4. **查日志上下文**：结合同线程前后日志（如 “Schedule run init succ” 之后、AbnormalStop 相关）确认是首次握手失败还是运行中异常。
