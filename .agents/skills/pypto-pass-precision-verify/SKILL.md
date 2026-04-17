---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass精度问题，定位精度问题出现在哪个Pass，并尝试修复
trigger: 验证Pass精度问题、pass verify、Pass精度调试、定位Pass精度问题、Pass精度报错、Pass精度不一致
---

## 快速参考

| 场景 | 错误码 | 处理方法 |
|-----|-------|---------|
| 前端问题 | `0xB4001U` | 调用 `pypto-precision-compare` 技能 |
| OP 报错 | `0xB200FU` | 检查 IR 图，对比 Before/After |
| 精度问题 | `0xB4001U` | 使用 PreCheck/PostCheck → pass_compare.py → 上板结果比对 |
| 工具误报 | 无固定码 | 对比 IR 图确认 shape 匹配 |
| 所有验证通过但精度异常 | 无报错 | 二分前端或二分 CCE |
| Reshape Pass验证报错 | EXCEPTION | 查阅ERROR-CASES.md案例01 |
| 精度通过但Pass报错 | EXCEPTION | 查阅ERROR-CASES.md案例02 |

---

## 常见场景速查表

| 场景 | 关键步骤 | 工具/命令 | 预期时间 |
|-----|---------|----------|---------|
| 快速定位前端问题 | 查看tensor_graph验证结果 → 调用pypto-precision-compare技能 | `grep "tensor_graph Verify" log/*.log` | 5分钟 |
| OP报错排查 | 检查IR图 → 对比Before/After → 查询OP详情 | `get_op_info.py --op-magic <ID>` | 10分钟 |
| Pass精度问题定位 | PreCheck/PostCheck → pass_compare → 上板比对 | 3步骤流程 | 20-40分钟 |
| 工具误报识别 | 对比IR图shape → 确认实际匹配 | IR分析工具 | 5分钟 |
| Codegen问题排查 | 二分前端 → 二分CCE → 打印验证 | `binary_cce.py --print-idx` | 30-60分钟 |
| 多CCE场景定位 | 先二分前端 → 确定范围 → 目标定位或二分搜索 | 方式一/方式二 | 40-80分钟 |
| 单CCE精确定位 | 初始化 → 列出CCE → 单CCE二分或手动修改 | 方式三/方式四 | 30-60分钟 |
| Reshape报错规避 | 查阅ERROR-CASES.md案例01 → 添加 `+ 0.0` → 验证 | ERROR-CASES.md | 10分钟 |
| Pass误报确认 | 查阅ERROR-CASES.md案例02 → 精度对比为准 → 确认 | ERROR-CASES.md | 5分钟 |

**快速诊断流程**：
```
出现精度问题
    ↓
查看验证日志
    ↓
是否有报错码？
    ├─ 有 0xB4001U → tensor_graph FAIL → 前端问题（调用pypto-precision-compare）
    ├─ 有 0xB200FU → OP报错 → IR图分析
    ├─ 有 0xB4001U + Pass名称 → Pass精度问题 → PreCheck/PostCheck流程
    ├─ 有 EXCEPTION + RESHAPE → reshape报错 → 查阅ERROR-CASES.md案例01
    ├─ 无报错但精度异常 → 二分前端或二分CCE
    └─ 精度PASS + Pass EXCEPTION → Pass误报 → 查阅ERROR-CASES.md案例02
```

---

## 目录

1. [简介](#简介)
2. [环境依赖检查](#环境依赖检查)
3. [环境变量配置](#环境变量配置)
4. [配置说明](#配置说明)
5. [配置备份与恢复](#配置备份与恢复)
6. [操作步骤](#操作步骤)
7. [错误码速查表](#错误码速查表)
8. [问题处理流程](#问题处理流程)
9. [IR 图对比分析方法](#ir-图对比分析方法)
10. [工具误报识别与处理](#工具误报识别与处理)
11. [二分 CCE 使用方法](#二分-cce-使用方法)
12. [多CCE场景处理](#多cce场景处理)
13. [单CCE场景处理](#单cce场景处理)
14. [IR 图分析方法](#ir-图分析方法)
15. [IR分析实战示例](#ir分析实战示例)
16. [常见错误案例库](./ERROR-CASES.md)
17. [测试规模优化建议](#测试规模优化建议)
18. [常见错误处理指南](#常见错误处理指南)
19. [故障排查流程](#故障排查流程)
20. [注意事项](#注意事项)

---

## 简介

本技能用于验证 PyPTO Pass 侧的精度问题。当 PyPTO 算子上板执行后输出数据与 torch 输出不一致时，用于排查问题是否出现在 Pass 处理阶段。

> **前端问题处理**：若日志显示 `tensor_graph Verify FAIL`，说明问题在前端代码，请直接调用 `pypto-precision-compare` 技能进行定位。

---

## 环境依赖检查

在开始使用本技能前，请检查以下环境依赖：

### 必需依赖

- [ ] CANN 工具链已安装
- [ ] PyPTO 已正确安装
- [ ] 编译环境（g++, make）可用
- [ ] Python 环境版本 >= 3.8
- [ ] torch 版本 >= 2.1.0
- [ ] 头文件 `pto/comm/pto_comm_inst.hpp` 存在

### 可选依赖

- [ ] NPU 硬件（用于 NPU 模式测试）
- [ ] SIM 模式环境（用于模拟测试）

### 依赖检查方法

```bash
# 检查 Python 版本
python3 --version

# 检查 torch 版本
python3 -c "import torch; print(f'torch version: {torch.__version__}')"

# 检查 PyPTO 安装
python3 -c "import pypto; print('PyPTO installed')"

# 检查编译环境
g++ --version
make --version

# 检查头文件
find /usr/include -name "pto_comm_inst.hpp" 2>/dev/null
find ~/.local -name "pto_comm_inst.hpp" 2>/dev/null
```

### 依赖缺失处理

**如果缺少 CANN 工具链**：
1. 参考 CANN 安装文档进行安装
2. 设置 CANN 相关环境变量

**如果缺少头文件**：
1. 检查 CANN 安装是否完整
2. 重新安装 CANN 或 PyPTO
3. 检查编译器的 include 路径设置

---

## 环境变量配置

以下环境变量必须在使用本技能前设置：

### 必需环境变量

```bash
# 必需：设置工作目录
export ASCEND_WORK_PATH="/path/to/work/directory"

# 必需：设置日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=0
```

### 可选环境变量

```bash
# 可选：设置设备 ID（NPU 模式必需）
export TILE_FWK_DEVICE_ID=0

# 可选：设置 PyPTO 日志级别
export PYPTO_LOG_LEVEL=0
```

### 验证环境变量设置

```bash
# 验证必需环境变量
echo "ASCEND_WORK_PATH: $ASCEND_WORK_PATH"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"

# 验证可选环境变量
echo "TILE_FWK_DEVICE_ID: $TILE_FWK_DEVICE_ID"
echo "PYPTO_LOG_LEVEL: $PYPTO_LOG_LEVEL"
```

### 环境变量设置时机

- **ASCEND_WORK_PATH**：必须在运行测试前设置
- **ASCEND_GLOBAL_LOG_LEVEL**：建议在运行测试前设置为 0（DEBUG）
- **TILE_FWK_DEVICE_ID**：NPU 模式运行前必须设置

---

## 配置说明

### 配置对比总览

| 配置类型 | 配置文件/参数 | 关键配置项 | 精度问题时设置 | 说明 |
|---------|-------------|----------|--------------|------|
| **verify_options** | 算子实现文件 | `enable_pass_verify` | **True** | 启用Pass验证（必须） |
| | | `pass_verify_pass_filter` | "all" | 验证所有Pass |
| | | `pass_verify_save_tensor` | **True** | 保存中间数据 |
| **tile_fwk_config.json** | framework配置文件 | `pre_check` | **true** | Pass前校验 |
| | | `post_check` | **true** | Pass后校验 |
| | | `dump_graph` | true | 保存IR图 |
| | | `print_graph` | true | 打印IR图 |
| **环境变量** | Shell环境 | `ASCEND_WORK_PATH` | 必须设置 | 工作目录路径 |
| | | `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为0 | DEBUG级别日志 |

### verify_options 配置

在 PyPTO 算子实现文件中配置：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_pass_filter": "all",      # 可选：Pass 侧有精度问题时设置
    "pass_verify_save_tensor": True,      # 可选：Pass 侧存在精度问题时设置
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    output: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1
```

**配置说明**：

| 配置项 | 说明 | 使用时机 |
|-------|------|---------|
| `enable_pass_verify` | 启用 Pass 验证 | 必须设置 |
| `pass_verify_pass_filter` | 过滤要验证的 Pass | Pass 侧有精度问题时设置为 `"all"` |
| `pass_verify_save_tensor` | 保存 Pass 中间数据 | Pass 侧存在精度问题时设置为 `True` |

**注意事项**：
- 必须使用具体的 shape 值，不能使用占位符
- 返回类型不能使用 `-> pypto.Tensor(...)` 语法，必须使用输出参数

### tile_fwk_config.json 配置

用于启用 PreCheck 和 PostCheck：

```json
{
    "pass": {
        "enable_binary_cache": false,
        "enable_pass_configs": true,
        "enable_cv_fuse": false,
        "pass_thread_num": 1,
        "vf_opt_mark_for": false,
        "enable_vf": true,
        "default_pass_configs": {
            "print_graph": true,
            "print_program": false,
            "dump_graph": true,
            "dump_pass_time_cost": true,
            "pre_check": false,     # Pass 精度问题时设为 true
            "post_check": false,    # Pass 精度问题时设为 true
            "expected_value_check": false,
            "disable_pass": false,
            "health_check": false,
            "use_max_freq_label": false
        }
    }
}
```

**配置文件位置**：`framework/src/interface/configs/tile_fwk_config.json`

---

## 配置备份与恢复

### 配置备份

在修改配置前，建议先备份原始配置：

```bash
# 备份 tile_fwk_config.json
cp framework/src/interface/configs/tile_fwk_config.json \
   framework/src/interface/configs/tile_fwk_config.json.backup

# 验证备份文件
ls -lh framework/src/interface/configs/tile_fwk_config.json.backup
```

### 配置恢复

调试完成后，恢复原始配置：

```bash
# 恢复 tile_fwk_config.json
cp framework/src/interface/configs/tile_fwk_config.json.backup \
   framework/src/interface/configs/tile_fwk_config.json

# 验证恢复
diff framework/src/interface/configs/tile_fwk_config.json.backup \
     framework/src/interface/configs/tile_fwk_config.json
```

### 配置修改建议

1. **每次修改前备份**：避免配置丢失
2. **记录修改内容**：便于问题排查
3. **测试完成后恢复**：避免影响后续工作
4. **使用版本控制**：推荐使用 git 管理配置文件

---

## 操作步骤

### 步骤一：配置校验开关

> **开始前**：请评估当前测试用例规模。如果数据量较大（如 Shape 参数 T>1000），建议：
> - 缩小 shape（如 T=64）以加快验证
> - 确认是否有最小可用用例
> - 防止编译/运行时间过长导致卡死

1. 配置 `verify_options`（见[配置说明](#配置说明)）
2. 设置 Golden 数据（**必须在算子运行前设置**）：

```python
# 计算 golden（必须是 CPU 上的 tensor）
torch_output = torch.add(input_data0, input_data1)

# 【重要】golden tensor 必须在 CPU 上，必须用额外变量接收 .cpu() 结果
# 错误写法：pypto.set_verify_golden_data(goldens=[None, None, torch_output.cpu()])
#         链式调用 .cpu() 可能导致内部校验失败
golden_cpu = torch_output.cpu()  # 用额外变量接收，确保 tensor 确实在 CPU 上
pypto.set_verify_golden_data(goldens=[None, None, golden_cpu])

# 执行 pypto 算子
your_kernel(input_data0, input_data1, output_data)
```

**关键注意事项**：
- `pypto.set_verify_golden_data()` 必须在 `your_kernel()` 调用**之前**执行
- 如果放在 pypto 算子运行之后设置，仅能校验 Pass 阶段，无法校验 Tensor Graph 阶段
- goldens 列表顺序对应算子的输入输出数量：`[input0, input1, ..., output]`
- goldens 列表中的输入项不参与校验，统一设置为 `None`

### 步骤二：编译并运行

```bash
# 编译安装（加 --no-build-isolation）
python3 -m pip install . --verbose --no-build-isolation

# 运行测试
python3 your_test_case.py
```

**输出目录位置说明**：

验证数据输出目录分为两类，需要区分：

| 目录类型 | 位置 | 内容 | 查看方法 |
|---------|------|------|---------|
| **验证数据目录** | 当前目录 `./output/output_*` | Pass 验证结果、tensor 数据、IR 图 | `ls ./output/` |
| **组件日志目录** | `ASCEND_WORK_PATH` 环境变量设置的工作目录 | Pass 侧日志、Machine 上板日志、编译日志 | `$ASCEND_WORK_PATH/log/` |

> **重要**：验证数据目录 `./output/output_*` **不依赖** `ASCEND_WORK_PATH` 环境变量，无论是否设置日志输出级别，都会在当前目录下生成。

**验证数据目录查找方法**：

```bash
# 查找当前目录下的验证输出目录
ls -la ./output/

# 查找最新的验证结果目录（按时间排序）
ls -lt ./output/ | head -5

# 查找验证数据目录（verify_*）
find ./output -name "verify_*" -type d

# 查看最新验证结果
latest_verify=$(ls -td ./output/*/verify_* 2>/dev/null | head -1)
echo "最新验证目录: $latest_verify"
ls -la "$latest_verify"
```

**组件日志目录结构**（在 `ASCEND_WORK_PATH` 下）：

```
log/
├── debug/
│   ├── device/       # Machine 上板日志（DumpAicoreLog*）
│   └── plog/
│       └── pypto-log-*  # Pass 侧日志
├── run/              # 运行日志
└── security/         # 安全日志
```

> **注意**：如果未设置 `ASCEND_WORK_PATH`，组件日志可能不会生成，但验证数据目录 `./output/output_*` 仍会正常生成。

### 步骤三：分析验证结果与定位问题

运行后会打印验证结果，错误码统一定义于 `framework/src/interface/interpreter/verify_error.h` 与 `framework/src/interface/interpreter/calculator/calc_error.h` 文件。

执行结束后，在 `{work_path}/output/output_*/verify_*/` 目录下生成验证数据：

```
├── tensor_graph/                  # 前端初始计算图数据
│   ├── *.data
├── Pass_XX_Name/                  # 各 Pass 中间数据（格式：Pass_XX_Name）
│   ├── *.data
├── verify_graph_result_brief.csv # 验证结果报告（简化版）
├── verify_graph_result_brief.log # 验证日志
└── verify_graph_data_metainfo.csv# 元信息
```

根据日志中的错误码和验证阶段，参考[问题处理流程](#问题处理流程)进行处理。

> **Pass 精度判断标准**：
> 
> Pass 侧精度是否通过，**只看最后一个 Pass（CodegenPreproc）是否正确**。
> 
> **判断依据**：
> - 如果 CodegenPreproc Pass 验证结果为 PASS → Pass 侧整体通过
> - 如果中间 Pass 报错但 CodegenPreproc PASS → 中间 Pass 错误可忽略（可能是工具误报）
> - 如果 CodegenPreproc FAIL → 需要定位具体 Pass 问题
>
> **判断流程**：
> ```
> 查看验证日志
>     ↓
> 找到 CodegenPreproc Pass 的验证结果
>     ↓
> 结果是什么？
>     ├─ PASS → Pass 侧整体通过 → 问题在 Codegen/Machine 阶段 → 进入二分 CCE
>     ├─ FAIL 且有其他 Pass FAIL → 定位第一个失败的 Pass → IR 分析
>     └─ FAIL 但其他 Pass PASS → CodegenPreproc 特定问题 → IR 分析 + CCE 二分
> ```

---

## 错误码速查表

| 错误码 | 名称 | 阶段 | 处理方法 |
|-------|------|------|---------|
| `0xB4001U` | VERIFY_RESULT_MISMATCH | 前端/Pass | 参考[问题处理流程](#问题处理流程) |
| `0xB200FU` | RUNTIME_EXCEPTION | Pass | 检查 OP 属性，参考 IR 图 |
| `0xB0001U` | VERIFY_NOT_ENABLE | 环境 | 检查 `torch >= 2.1.0` |
| 其他 | — | 未知 | 联系开发人员 |

---

## 问题处理流程

### 情况一：tensor_graph Verify FAIL（前端问题）

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **原因** | 前端代码书写问题 |
| **处理** | 调用 `pypto-precision-compare` 技能，发送指令："使用 pypto-precision-compare 技能，定位 xxx.py 的精度问题" |

---

### 情况二：Pass 级别 Verify FAIL

#### 2.1 OP 报错

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB200FU`（RUNTIME_EXCEPTION） |
| **日志特征** | `[operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB ...` 日志会输出出错 OP 的 `<shape/validshape>` 及属性信息 |
| **原因** | Operation 模拟执行失败，OP 上存在错误或缺失的属性 |
| **处理** | 查看对应 Pass 的 IR 图，分析该 OP 是否缺失属性或属性值不正确。分析 IR 图的方法参考 pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md，查找对应的 OpMagic 进行分析 |

> **重要**：遇到 OP 报错时，请先**对比 Before 和 After 两个 IR 文件**，确认是否为工具误报。如果 IR 图显示 shape 实际匹配，则可能是验证了具的误报，此时应按照"情况三"继续排查。

#### 2.2 精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! pass_06_SplitReshape Verify result FAILED` |
| **原因** | Pass 处理阶段引入的精度偏差 |

**Pass 侧精度问题定位流程**：

1. **首先使用 PreCheck/PostCheck 方法**：打开 `tile_fwk_config.json` 中对应 Pass 的 PreCheck 及 PostCheck 开关，关闭精度工具配置后重新编译运行，观察是否存在报错。有报错则分析，无报错再往下用其他方法

2. **其次利用 pass_compare.py**：对比失败 Pass 与前置 Pass 的每个 OP 节点，定位第一个出错的节点

3. **最后使用上板结果比对**：将 Machine 上板后的结果与 verify_result 中各 Pass 保存的输出逐一比对，第一个输出与上板输出一致的 Pass 即为出错的 Pass

**pass_compare.py 使用方法**：

pass_compare.py 位于 `tools/verifier/pass_compare.py`，使用方法：

```bash
python3 tools/verifier/pass_compare.py --p <FailedPass> <GoldenPass> --verify_path=/path/to/verify_data
```

- `--p` 参数后是对比的两个 Pass，空格隔开，前者为精度对比失败的 Pass，后者为作为 golden 的 Pass
- `--verify_path` 参数为精度工具 dump 数据文件目录的绝对路径
- 对比结果会生成类似 `verify_pass@SplitK@ExpandFunction@1773821696834386.csv` 的文件，记录每个 OP 节点的对比结果，未能匹配的节点会标注 skip

---

### 情况三：所有验证都 PASS 但仍有精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | 无报错，但上板结果与 golden 不一致 |
| **日志特征** | 所有 Pass 验证显示 PASS，但最终输出含 NaN 或与 golden 差异大 |
| **原因** | 问题可能在 Codegen 或 Machine 执行阶段，也可能 Pass 侧未能覆盖校验 |

**处理方式**：

1. **先回顾常见 Pass 错误类型**，逐一排除（参见[Pass 常见错误类型及修复](#pass-常见错误类型及修复)）

2. **如果 Pass 验证全部通过但存在工具误报**（如 IR 图显示 shape 实际匹配，但验证仍报错）：
   - 这通常是验证工具的误报，无需针对该 Pass 进行修复
   - 应继续按照以下排查方向定位真正的问题

3. **排查方向选择**（根据实际情况选择）：

   | 排查方向 | 适用场景 | 操作方法 |
   |---------|---------|---------|
   | **二分前端** | 利用二分前段方法定位哪些Op出现错误 | 调用 `pypto-precision-compare` 技能，在前端代码中插入检查点，定位产生精度问题的代码行 |
   | **二分 CCE** | 利用二分 CCE 方法定位哪些Op出现错误 | 使用 `binary_cce.py` 工具，在生成的 CCE 代码中添加打印，定位具体出错的 Op |

4. **若仍未定位**，向用户确认是否继续排查

---

## 二分 CCE 使用方法

当问题定位到 Codegen/Machine 阶段时，使用二分法或目标定位法定位具体出错的 Op。

### 场景判断标准

| 判断条件 | 多CCE场景 | 单CCE场景 |
|---------|-----------|-----------|
| CCE文件数量 | > 10个 | = 1个 |
| 函数复杂度 | 包含多个子图 | 单一计算图 |
| 问题范围 | 跨多个CCE | 单个CCE内 |
| 推荐方法 | 方式一/二 | 方式三/四 |

### 场景判断方法

```bash
# 1. 统计CCE文件数量
cce_count=$(ls output/output_*/kernel_aicore/*.cpp 2>/dev/null | wc -l)
echo "CCE文件数量: $cce_count"

# 2. 分析CCE复杂度
echo "CCE文件复杂度分析:"
for cce_file in output/output_*/kernel_aicore/*.cpp; do
    if [ -f "$cce_file" ]; then
        func_count=$(grep -c "void.*kernel" "$cce_file" 2>/dev/null || echo "0")
        line_count=$(wc -l < "$cce_file")
        echo "  $(basename $cce_file): $func_count 个函数, $line_count 行"
    fi
done

# 3. 确定场景
if [ "$cce_count" -gt 10 ]; then
    echo "当前场景: 多CCE场景"
    echo "推荐方法: 方式一（目标定位）或 方式二（二分搜索）"
elif [ "$cce_count" -eq 1 ]; then
    echo "当前场景: 单CCE场景"
    echo "推荐方法: 方式三（单CCE直接二分）或 方式四（手动修改CCE）"
else
    echo "当前场景: 中等规模CCE场景"
    echo "推荐方法: 根据问题复杂度选择方式一或方式三"
fi
```

### 决策树

```
开始二分CCE
       ↓
判断CCE场景
       ↓
  ┌─────────────────────────┐
  │                         │
多CCE场景               单CCE场景
  │                         │
  ↓                         ↓
能否直接定位？        方式三：单CCE直接二分
  │                         │
  是                        方式四：手动修改CCE
  ↓                         │
方式一：目标定位             │
  ↓                         │
方式二：二分搜索             │
  │                         │
  └─────────────────────────┘
```

### 前提条件

CCE 文件查找范围在 `kernel_aicore/*.cpp`，每个 cpp 文件对应一张子图。可用以下命令列出：

```bash
# 统计 CCE 数量
ls output/output_*/kernel_aicore/*.cpp 2>/dev/null | wc -l

# 查看前20个
ls output/output_*/kernel_aicore/*.cpp 2>/dev/null | head -20

# 查看CCE文件详情
for cce_file in output/output_*/kernel_aicore/*.cpp; do
    echo "=== $(basename $cce_file) ==="
    echo "文件大小: $(wc -c < $cce_file) bytes"
    echo "函数数量: $(grep -c "void.*kernel" $cce_file)"
    echo "行数: $(wc -l < $cce_file)"
done
```

---

## 多CCE场景处理

### 方式一：目标定位（推荐多CCE场景）

根据错误日志或输出分析，定位可疑的 CCE 文件，直接打印验证：

#### 1. 分析可疑 CCE

- 查看验证日志中的函数名（如 `TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11`）
- 或根据输出差异的 pattern 推测可能出问题的算子
- 检查 IR 图中的异常操作（参考[IR 图分析方法](#ir-图分析方法)）

#### 2. 找到对应 CCE 文件

```bash
# 根据函数名搜索
ls output/output_*/kernel_aicore/*TENSOR_Loop_S_Unroll1*.cpp

# 或根据 funcHash 搜索
grep -l "funcHash: 11899060959657268680" output/output_*/kernel_aicore/*.cpp

# 根据操作类型搜索
grep -l "TILE_MATMUL" output/output_*/kernel_aicore/*.cpp
```

#### 3. 手动添加打印

在可疑 cpp 文件中添加：
```cpp
#include "tilefwk/aicore_print.h"

// 在 kernel 函数对应的位置打印输入
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)gmTensor_X.GetAddr(), element_count, 0);
```

#### 4. 运行验证

```bash
python3 your_test.py
# 查看 log/debug/device-*/DumpAicoreLog* 获取打印结果
```

---

### 方式二：二分搜索（多CCE场景）

当存在多个 CCE 文件，且无法直接定位可疑文件时，推荐**先使用 pypto-precision-compare 二分前端**，定位到问题范围后再在该范围内二分 CCE。

#### 第一步：先用 pypto-precision-compare 定位问题范围

调用 `pypto-precision-compare` 技能，在前端代码中插入检查点，二分定位问题出现在哪个检查点区间。根据检查点的名称可以推断出问题出现在哪个 Pass 或哪个计算阶段。

#### 第二步：在问题范围内的 CCE 文件中二分

```bash
# 列出所有 CCE 文件，确认问题范围
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --list-cce
```

找到问题范围对应的 CCE 索引（如 20-50），然后在该范围内二分：

```bash
# 从范围中间开始打印
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --print-idx 35 \
    --pos kernel_start \
    --rebuild
```

#### 第三步：根据结果判断方向

运行后检查输出：
- 输出正常或差异变小 → 问题在后半部分 → 继续打印后半区间的 CCE
- 输出仍然异常 → 问题在前半部分 → 继续打印前半区间的 CCE

重复二分，逐渐缩小范围。

---

## 单CCE场景处理

### 方式三：单CCE直接二分

当确定问题在单个 CCE 文件中时，直接在该 CCE 内二分查找具体 Op：

#### 第一步：初始化配置（仅需一次）

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --init \
    --work-path /path/to/output/output_latest/
```

#### 第二步：列出CCE文件信息

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --list-cce
```

输出示例：
```
找到 186 个 CCE 文件:
  [0] TENSOR_Loop_S_TND_Unroll1_PATH0_xxx.cpp
  [1] TENSOR_Loop_S_TND_Unroll1_PATH1_xxx.cpp
  ...
```

#### 第三步：选择打印位置

基于以下原则选择：
- 如果知道大概范围（如 0-93），从中间开始
- 如果完全不确定，从约 1/3 位置开始

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --print-idx 60 \
    --pos kernel_start \
    --rebuild
```

---

### 方式四：手动修改 CCE（高精度定位）

当二分定位到单个 CCE 后，需要精确定位具体 Op：

#### 1. 查看 CCE 内容

```bash
# 查看 CCE 文件
cat output/output_*/kernel_aicore/XXX.cpp

# 查看CCE中的关键操作
grep -n "TILE_MATMUL\|TILE_ADD\|TILE_MUL" output/output_*/kernel_aicore/XXX.cpp
```

#### 2. 在可疑操作前后添加打印

```cpp
#include "tilefwk/aicore_print.h"

// 在 TILE_MATMUL 之前
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)input_tensor.GetAddr(), 1024, 0);

// 执行操作
...

// 在 TILE_MATMUL 之后
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)output_tensor.GetAddr(), 1024, 0);
```

#### 3. 基于IR分析的精确打印

参考详细IR分析指南：`pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

```bash
# 1. 分析IR文件，定位问题操作
python3 .agents/skills/pypto-pass-error-locator/scripts/analyze_ir_operation.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --operation-id 10003

# 2. 根据IR分析结果，在CCE中定位对应操作
# IR分析会提供操作类型、输入输出、内存类型等信息

# 3. 在CCE中添加精确的打印语句
# 根据IR分析的内存类型和shape信息，选择合适的打印方法
```

---

## IR 图分析方法

用于判断是否为工具误报或真实错误，以及辅助CCE二分定位。

参考详细IR分析指南：`pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

### IR分析检查清单

基于ir-analysis-guide.md的分析要点：

#### 1. 文件完整性检查

- [ ] 文件头格式正确
- [ ] 所有RAWTENSOR索引唯一
- [ ] 所有INCAST/OUTCAST索引唯一
- [ ] 所有操作节点op_id唯一
- [ ] 所有变量定义和使用匹配

#### 2. 数据流分析

- [ ] 追踪数据从输入到输出的完整路径
- [ ] 验证数据依赖关系的正确性
- [ ] 检查是否存在悬空数据或数据泄露

#### 3. 内存访问分析

- [ ] 检查内存访问的合法性
- [ ] 验证内存类型转换的正确性
- [ ] 检查是否存在内存冲突

#### 4. Shape一致性检查

- [ ] 操作输入输出shape匹配
- [ ] 张量shape传播正确
- [ ] 动态shape处理正确

### IR分析工具使用

```bash
# 使用计算图分析工具分析IR文件
python3 .agents/skills/pypto-pass-error-locator/scripts/computation_graph_analyzer.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr

# 查询指定 OP 的详细信息（精度问题定位时使用）
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic 10003

# 列出所有 OP（用于查找报错的 OP）
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --list-ops
```

### 常见IR问题模式

| 问题模式 | IR特征 | 可能原因 | 处理方法 |
|---------|---------|---------|---------|
| 操作丢失 | After中缺少某些操作 | Pass错误删除了必要操作 | 对比操作ID和操作名 |
| 变量未定义 | 使用了未定义的变量 | Pass删除了变量定义 | 检查变量定义和使用 |
| shape不匹配 | 操作输入输出shape不一致 | Pass错误修改了shape | 检查操作节点的shape |
| 内存类型错误 | 内存类型转换不正确 | Pass错误分配了内存类型 | 检查内存类型转换链 |

### 基于IR分析的问题定位流程

```
开始IR分析
       ↓
检查文件完整性
       ↓
   完整？
       ↓ 是
分析数据流
       ↓
   正常？
       ↓ 是
检查内存访问
       ↓
   正常？
       ↓ 是
验证shape一致性
       ↓
   一致？
       ↓ 是
IR分析完成，问题不在Pass
       ↓
进入CCE二分
       ↓
多CCE场景？
       ↓ 是
方式一：目标定位
       ↓ 否
方式三：单CCE直接二分
```

### IR分析与CCE二分的结合

#### 步骤1：IR分析确定问题范围

```bash
# 使用计算图分析工具分析Before和After IR文件
python3 .agents/skills/pypto-pass-error-locator/scripts/computation_graph_analyzer.py \
    --before-file output/output_*/Pass_XX_Name/Before_XXXX_PassName_funcname.tifwkgr \
    --after-file output/output_*/Pass_XX_Name/After_XXXX_PassName_funcname.tifwkgr \
    --output-dir ir_analysis_result

# 查看生成的分析报告
cat ir_analysis_result/analysis_report.txt
```

#### 步骤2：根据IR分析结果选择CCE二分策略

| IR分析结果 | 问题类型 | 推荐CCE二分策略 |
|-----------|---------|----------------|
| 操作丢失 | 特定操作缺失 | 方式一：目标定位 |
| shape不匹配 | shape计算错误 | 方式二：二分搜索 |
| 内存类型错误 | 内存操作错误 | 方式三：单CCE直接二分 |
| 数据流断裂 | 数据流问题 | 方式四：手动修改CCE |
| 无明显异常 | 未定位到具体问题 | 方式二：二分搜索 |

#### 步骤3：结合IR信息进行CCE二分

**多CCE场景 - 基于IR分析的目标定位**

```bash
# 1. 从IR分析报告中获取问题函数名或操作信息
# 查看 ir_analysis_result/analysis_report.txt

# 2. 根据函数名定位对应CCE
cce_file=$(ls output/output_*/kernel_aicore/*函数名*.cpp 2>/dev/null | head -1)
echo "对应CCE文件: $cce_file"

# 3. 在CCE中添加打印
# 参考方式一的打印方法
```

**单CCE场景 - 基于IR分析的精确二分**

```bash
# 1. 从IR分析报告中获取问题操作信息
# 查看 ir_analysis_result/analysis_report.txt

# 2. 在CCE中定位对应操作
grep -n "操作关键字" output/output_*/kernel_aicore/*.cpp

# 3. 在操作前后添加打印
# 参考方式四的打印方法
```

---

## IR分析实战示例

### 示例1：操作丢失问题

**场景**：Pass验证报错，IR对比显示After中缺少TILE_ADD操作

**IR对比结果**：
```bash
# Before IR
!10003 TILE_ADD(g:-1, s:-1) %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %8@12#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR

# After IR - 缺少TILE_ADD操作
# !10003 TILE_ADD 操作不存在
```

**问题定位**：
1. 确认操作丢失：对比Before和After IR的操作ID
2. 查找影响该操作的Pass：检查Pass日志
3. 定位对应CCE：根据函数名查找CCE文件

**解决方法**：
```bash
# 1. 找到包含TILE_ADD的CCE文件
cce_file=$(grep -l "TILE_ADD" output/output_*/kernel_aicore/*.cpp 2>/dev/null | head -1)
echo "问题CCE: $cce_file"

# 2. 在CCE中添加打印验证
# 在TILE_ADD操作前后添加打印
```

### 示例2：shape不匹配问题

**场景**：Pass验证报shape错误，IR分析显示shape计算错误

**IR分析结果**：
```bash
# Before IR
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR

# After IR - shape不匹配
<8 x 128 x DT_FP32 / 8 x 128 x DT_FP32> %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR
# 注意：shape从 16x128 变成了 8x128
```

**问题定位**：
1. 确认shape不匹配：对比Before和After IR的shape
2. 分析shape变化原因：检查shape计算相关操作
3. 定位对应CCE：根据shape变化定位CCE

**解决方法**：
```bash
# 1. 使用 get_op_info.py 查询报错 OP 的详细信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json

# 2. 对比 Before 和 After IR 中的 OP 信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/Before_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json > before_op.json

python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json > after_op.json

# 3. 比对两个 JSON 文件的差异
diff before_op.json after_op.json

# 4. 根据分析结果定位CCE
# 查看生成的分析报告
cat ir_analysis_result/analysis_report.txt

# 3. 在相关CCE中添加打印
```

### 示例3：内存类型错误问题

**场景**：Pass验证报内存错误，IR分析显示内存类型转换错误

**IR分析结果**：
```bash
# 错误的内存类型转换
%84@50#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR = !10022 TILE_VIEW(g:-1, s:-1) %12@14#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR
# 问题：从DDR直接到DDR，没有经过UB
```

**问题定位**：
1. 确认内存类型错误：检查内存类型转换链
2. 分析内存访问模式：检查是否需要经过UB
3. 定位对应CCE：根据内存操作定位CCE

**解决方法**：
```bash
# 1. 使用 get_op_info.py 查询 OP 信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic 10022 \
    --format json

# 2. 检查输出中的内存类型
# 查看 "memory_type" 字段，确认 read 和 write 类型是否正确

# 3. 在相关CCE中添加打印
```

---

## 常见错误案例库

遇到常见错误时，请查阅 [ERROR-CASES.md](./ERROR-CASES.md) 查看具体案例的诊断与解决方案。

**案例索引**：
- **案例01**：Reshape操作导致Pass验证报错 → reshape后添加 `+ 0.0` 规避
- **案例02**：精度对比通过但Pass验证报错 → 以精度对比结果为准

**案例结构**：每个案例包含：
- 场景描述：问题发生的典型场景
- 问题特征：错误码、日志特征、现象描述
- 诊断步骤：如何定位问题（包含具体命令）
- 解决方案：修复方法（包含代码示例）
- 验证结果：预期结果与判断标准
- 注意事项：特殊情况和限制条件

---

## 注意事项

1. **CCE 文件是 .cpp 格式**：每个 cpp 对应一张子图（kernel）
2. **打印会重新编译**：修改 cpp 后需要重新运行测试
3. **查看日志位置**：`output/output_*/log/debug/device-*/DumpAicoreLog*`
4. **恢复原文件**：调试完成后记得删除打印语句
5. **配置修改**：初始化时会将 `codegen.parallel_compile` 设为 1 以控制编译并发
6. **IR分析辅助**：充分利用IR分析指南，提高定位精度
7. **场景判断**：根据CCE数量和复杂度选择合适的二分策略
8. **结合使用**：IR分析和CCE二分结合使用，提高效率
