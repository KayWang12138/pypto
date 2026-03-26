# MACHINE 组件错误码

- **范围**：F7-F8XXXX
- 本文档说明 MACHINE 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义与使用说明

相关错误码的统一定义，参见 [machine_error.h](../../framework/src/machine/utils/machine_error.h) 文件。

## 排查建议

### 怀疑和MACHINE内存处理有关的精度问题

1. **检查输入初始化**：
保证输入/输出初始化

2. **检查 Tensor 连续性**：
部分 MACHINE 相关算子或接口要求输入/输出在指定内存布局下**连续**（`tensor.is_contiguous()` 为 True）。非连续 tensor（如某些 view、transpose、slice 结果）可能导致精度异常或运行错误。
eg：reshape/view、交换维度、索引/切片后的 tensor 可能非连续；若前序是 COPY_IN 或尾轴 reduce，需在前端或调用侧保证传入的 tensor 在对应格式下连续。

3. **扩大 workspace 大小**:
`python/pypto/frontend/parser/entry.py`
```python
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
```

```python
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```
如果问题不复现，则是workspace计算问题

4. **workspace从torch管理，改为内部自管理**
`framework/src/machine/runtime/device_launcher.cpp`
```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```

```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (0) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```
如果问题不复现，则是workspace使用问题，存在内存踩踏等

5. **leaf function粒度的内存重叠检测**
（1）打开VERBOSE日志
`framework/src/machine/utils/device_switch.h`
```cpp
#define ENABLE_COMPILE_VERBOSE_LOG 1
```

（2）打开DEBUG日志，指定日志落盘路径
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

（3）重新编pypto whl包并安装

（4）启动性能数据采集功能，保证生成 dyn_topo.txt

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

（5）执行用例

（6）执行内存检测脚本

命令格式：
```bash
python3 tools/schema/schema_memory_check.py -d <device_log_dir_absolute_path> -t <topo_file_absolute_path>
```

示例（请替换为实际绝对路径）：
```bash
python3 tools/schema/schema_memory_check.py -d /path/to/my_log/debug/device-8/ -t /path/to/output/output_20260314_112655_964781_3025352/dyn_topo.txt
```

如无异常，提示device task 无内存重叠；
如存在异常，提示内存重叠的 device task 以及 leaf function。

注：
（1）如果报错 emory reuse must happen for full match. 则两个需要内存复用的 rawtensor 范围不一致；
（2）如果报错 memory reuse must happen for same dimension. 则两个内存复用的 rawtensor 的 shape 不一致；
上述两种情况非内存重叠，脚本内存检查依赖不会发生上述情况 ，因此脚本会断言，直接提示日志信息错误。

6. **复杂特性排除**：
关闭unroll_list、合轴特性、配置submit_before_loop=True使loop串行执行、确定valid_shape配置正确性、+0.0等，缩小定位范围


### encode 阶段 actualRawMagic 断言触发

**问题特征**：运行用例时在 encode 阶段触发断言，错误信息包含 `Shape size mismatch` 或 `Data size mismatch`，并伴随 `rawTensor->actualRawmagic`、`rawShape`、`actualrawShape` 等字段输出，报错位置为 `framework/src/machine/utils/dynamic/dev_encode.cpp` 中 `InitRawTensorAndMemoryRequirement` 函数。

**问题背景**：encode 阶段会对所有带有 `actualRawmagic` 的 RawTensor 进行内存复用一致性校验——即要求复用链两端（当前 rawTensor 与其 `actualRawmagic` 指向的 actualRaw）在 rawShapeSize（元素数乘积）或 rawDataSize（字节大小）上严格一致。涉及 `reshape`、`assemble` 等内存复用操作时，若 pass 侧未同步更新相关 tensor 的特征信息，就会触发此类断言。

**定位步骤**：

1. **从报错第一现场获取基础信息**：

   触发断言时，错误日志中会包含以下关键字段（对应 `dev_encode.cpp` 第 412~420 行）：

   ```
   Shape size mismatch: <rawShapeSize> != <actualRawShapeSize>,
   rootMagic=<...>, rootHash=<...>,
   rawShape=<...>, actualrawShape=<...>,
   rawTensor->rawMagic=<...>, rawTensor->actualRawmagic=<...>, actualRaw->rawMagic=<...>
   ```

   记录其中的 `rawMagic`、`actualRawmagic`、`rawShape`、`actualrawShape`，作为后续计算图定位的依据。

   若当前报错信息不够完整（缺少 rawMagic 等字段），可参考 `dev_encode.cpp` 中该 ASSERT 的上下文，在断言前增加相关字段打印，重编 whl 包后复现。

2. **打开 pass 计算图 dump 开关**：

   修改 `framework/src/interface/configs/tile_fwk_config.json`，将 `global.pass.default_pass_configs` 下的 `dump_graph` 改为 `true`：

   ```json
   "default_pass_configs": {
       "print_graph": false,
       "print_program": false,
       "dump_graph": true,
       ...
   }
   ```

   重新编译 whl 包并安装。

3. **复现用例，获取 pass 计算图**：

   再次执行用例，在 `build/output/pass/` 目录下会生成各 pass 阶段的计算图文件。

4. **定位目标 pass 范围**：

   建议重点排查 **pass4 ~ pass27** 之间的 tilegraph：
   - pass4 ExpandFunction： 之前为 tensorgraph 阶段，尚未进入 pass 的tile展开 粒度分析
   - pass27 SubgraphToFunction: 之后框架开始切分 root function 为 leaf function，图结构较为分散，不便于整体定位

   在该范围内，根据步骤 1 获取的 `rawMagic` / `actualRawmagic`，在计算图中定位对应 tensor 节点。

5. **结合计算图分析问题原因**：

   确定 tensor 节点后，沿数据流向分析其上下游操作，重点关注以下场景：

   - **reshape 操作**：reshape 会为输出 tensor 设置 `actualRawmagic`，使其复用输入 tensor 的内存地址
   - **assemble 操作**：assemble 在图优化阶段会将输入 tensor 的 raw 指针替换为目标大 tensor 的 raw，若 reshape 在此之前已设置 `actualRawmagic`，该字段可能未随 raw 替换同步更新
   - **其他内存复用操作**：view、inplace 等操作同样涉及 `actualRawmagic` 的设置与传递

   若复用链两端的 rawShapeSize 在某个 pass 之后出现不一致，说明该 pass 对其中一侧的 rawshape 进行了改写，但未同步另一侧。

6. **判断问题归属**：

   （1）**用例写法问题**：检查用例中 reshape、assemble 等操作的组合方式是否满足 API 约束（如 `inplace=True` 的限制、valid_shape 与 rawshape 的一致性要求等）。若存在不满足约束的写法，调整用例。

   （2）**框架侧问题**：若用例写法无误，联系 **pass 同事**进行进一步分析，排查框架在处理 actualRawmagic 传递时是否存在遗漏或错误更新的场景。

注：
- actualRawmagic 断言失败本质是编译期一致性校验，去掉断言后若运行精度仍正确，说明该路径的运行时读写并未越界，属于校验口径过严或 pass 侧信息同步遗漏问题，需与 pass 同事联合分析
- 此类问题若在动态 shape（含负数维度）场景下触发，`dev_encode.cpp` 会跳过动态维度的校验（`isDynamicShape` 分支），排查时需注意区分静态与动态 shape 场景

---

### F70006 HANDSHAKE_TIMEOUT

1. **确认设备与驱动**：NPU 设备可用、驱动正常，`npu-smi info` 无异常。
2. **确认资源与负载**：当前进程/容器内 NPU 占用是否过高，是否存在多进程争用同一设备。
3. **确认超时配置**：若存在握手/同步超时配置项，检查是否过短或与环境不符。
4. **查日志上下文**：结合同线程前后日志（如 “Schedule run init succ” 之后、AbnormalStop 相关）确认是首次握手失败还是运行中异常。

**关联 Skill**：[pypto-environment-setup](../../.opencode/skills/pypto-environment-setup/SKILL.md)（环境与 NPU 设备诊断、`npu-smi`、驱动与编译运行）

---

## 泳道图相关问题指导

### output 目录产物说明

当前 `output` 目录中涉及泳道图相关的文件如下：
`machine_runtime_operator_trace*.json`、`machine_trace_perf_data*.json`、`merged_swimlane.json`、`tilefwk_L1_prof_data_*.json`。

其中 `machine_trace_perf_data*.json` 和 `tilefwk_L1_prof_data_*.json` 是 Machine 组件提供的原始 Profiling 数据，可以通过查看这些文件内容是否为空，来判断是否成功采集到信息。其余 JSON 文件则是 IDE 展示所需的格式化文件，如涉及 IDE 展示异常等问题，建议优先联系 IDE 对应负责人咨询解决。

### IDE 参数含义解释

在生成和查看泳道图时，IDE 工具中会显示多个性能参数和事件标签。以下是常见参数的含义说明：

**1. AICore 泳道图**

| 参数/事件名称 | 含义解释 |
| --- | --- |
| **Task Time** | AI Core 端到端执行时间 |
| **AICore Time** | 所有 AICore 任务时间之和 |
| **AICore 利用率** | `AICore Time` / (`泳道数` × `泳道时长`) |

**2. AICPU 泳道图**

| 模块 | 参数/事件名称 | 含义解释 | 线程信息 |
| --- | --- | --- | --- |
| **AICPU** | **DEV_TASK_BUILD** | Ctrl AICPU 构建 devTask 的耗时（即 stitch 耗时统计） | Ctrl |
| **AICPU** | **DEV_TASK_RCV** | Sched AICPU 接收到 Ctrl AICPU 构建的 devTask 的耗时 | Sched |
| **AICore** | **RCV_MODEL** | AICore 接收到 AICPU 发送的 devTask 的耗时 | Sched |
| **AICore** | **ALL_CALLOP_TASK_EXEC** | 当前 AICore 的 devTask 中所有 leafTask 执行完成的耗时 | Sched |


### 常见异常排查

#### 1. 未生成泳道图文件
**现象**：算子运行正常，但 `output` 目录下未生成 `merged_swimlane.json` 文件。
**原因与排查**：通常是因为未启动性能数据采集功能。请检查代码中是否已正确将 `runtime_debug_mode` 设置为 `1`。

#### 2. 泳道图文件为空（无任何数据）
**现象**：成功生成了 `merged_swimlane.json` 文件，但文件内容为空。
**原因与排查**：通常是 Profiling 功能未能成功使能。需要开启 DEBUG 日志打印进行进一步排查：
   - 按照上文说明打开 DEBUG 日志并指定日志落盘路径。
   - **Device 侧排查**：检查日志文件 `log/debug/device*/device*.log`。若包含 `aicore profiling is opened, level is %d.`，表示成功使能；若包含 `aicore profiling is closed..`，则表示未能成功使能。

#### 3. 泳道图中某些核首任务启动时间过长
**现象**：从泳道图看，部分核并没有前序任务依赖，但第一个任务的启动时间却很长。
**原因与排查**：这种情况通常是因为 AICPU 启动较慢，导致 AICore 接收任务的时间被整体延后。在泳道图中表现为首任务启动前存在等待 AICPU 启动的时间。

#### 4. ACL Graph 模式下采集不到泳道图数据
**现象**：当算子运行在 ACL Graph 模式时，启动泳道图性能采集后，`output` 目录下没有生成泳道图文件。
**原因与排查**：当前 PyPTO 尚未支持 ACL Graph 场景的泳道图性能数据采集。在 ACL Graph 模式中，执行流程分为 Capture 和 Replay 两个阶段，当前 Capture 阶段未开启 Profiling，而是在 Replay 阶段开启性能采集；但 Task 的下发实际发生在 Capture 阶段，由于此时 Profiling 开关是关闭的，所以不会上报 OP 相关信息。目前请暂时规避该场景，后续版本将支持 ACL Graph 模式下的泳道图性能数据采集。

#### 5. Profiling 泳道图数据与 msprof 采集的结果差距较大
**现象**：`msprof` 采集到的 AICore 耗时远大于泳道图中的 AICore 端到端耗时，二者数据无法对齐。
**原因与排查**：`msprof` 所采集到的 AICore 耗时不能真实代表 AICore 内部端到端的执行耗时，因为它实际上还包含了 **AICore 启动等待 AICPU 下发 devTask 的时间**，以及 **AICore 执行完任务后的退出时间**。为了获取更准确的时间，当前已实现对 AICore 端到端执行时间的打屏输出，可以在执行算子前设置环境变量 `export DUMP_DEVICE_PERF=true`，即可在终端中直接获取当前准确的性能统计数据。