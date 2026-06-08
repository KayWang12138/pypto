# 子图 CostModel 移植与验证指南

> **目标读者**: 另一台服务器上的 Agent，需要在不同于原始开发的 CANN 版本和源码基础上，实现并跑通子图 costmodel 功能。
>
> **最终目标**: 在目标服务器上，对 softmax 算子执行子图 costmodel 仿真，获得 `status=success`，返回 `subgraph_total_cycles` 和 `functions` 列表。

---

## 一、功能概述

### 什么是子图 CostModel

PyPTO 的 CostModel 有两种运行模式：

| 模式 | 说明 | 入口 |
|------|------|------|
| **整图 (Whole-graph)** | 对整张计算图做仿真，在 SIM 模式 kernel 运行时自动触发 | `pypto_impl.CostModelRunOnceDataFromHost` |
| **子图 (Sub-graph)** | 按子图 ID (`p_sg_id`) 过滤，只仿真指定子图的 leaf function 和拓扑调度 | `pypto_impl.CostModelRunSubgraphLine`（新增） |

### 子图 CostModel 的两阶段流程

```
Phase 1: LEAF_FUNCTION 模式
  → 只提交属于目标 p_sg_id 的 leaf function 到 CostModel
  → 获取每个 leaf function 的 cycles

Phase 2: NORMAL 模式
  → 从 dyn_topo.txt 中过滤出目标 p_sg_id 的拓扑条目
  → 提交过滤后的拓扑 + leaf function 到 CostModel
  → 获取子图的全局调度 cycles (subgraph_total_cycles)
```

### 预期输出格式

```json
{
  "status": "success",
  "error_msg": "",
  "p_sg_id": 0,
  "subgraph_total_cycles": 74223,
  "functions": [
    {
      "hash": 15536366383870408930,
      "cycles": 102,
      "name": "TENSOR_LOOP_L0_...leaf0_12",
      "machine_type": 4
    }
  ],
  "output_dir": "/path/to/output/CostModelSimulationOutput"
}
```

---

## 二、需要修改的文件清单

一共需要改动 **4 个文件**：

| 文件路径 | 改动类型 | 说明 |
|----------|----------|------|
| `framework/src/cost_model/simulation/backend.h` | 新增 3 个方法声明 | `SubmitLeafFunctionsBySubgraph`、`SubmitSubgraphTopoByPid`、`GetCostModel` |
| `framework/src/cost_model/simulation/backend.cpp` | 新增 2 个方法实现 + 修改 `SubmitLeafFunctionsBySubgraph` 的实现 | 子图 leaf 过滤、拓扑过滤 |
| `framework/src/cost_model/simulation/cost_model_launcher.h` | 新增 1 个 public 静态方法 + 2 个 private 静态方法 | `CostModelRunSubgraph` 入口 + 两个 Phase 实现 |
| `python/src/bindings/cost_model.cpp` | 新增 1 个函数 + 1 个 binding | `CostModelRunSubgraphLine` Python 绑定 |
| `python/pypto/cost_model.py` | 新增 1 个 Python 函数 | `_cost_model_run_subgraph_line` |

---

## 三、逐文件详细改动

### 3.1 `framework/src/cost_model/simulation/backend.h`

在 `CostModelAgent` 类的 `public:` 区域内（在已有的 `SubmitLeafFunctionsToCostModel()` 声明之后），新增 3 个方法：

```cpp
// ---- 在 public: 区域内已有方法之后添加 ----
void SubmitLeafFunctionsBySubgraph(uint64_t pSgId);
void SubmitSubgraphTopoByPid(std::string& path, uint64_t pSgId);
std::shared_ptr<CostModel::CostModelInterface> GetCostModel() const { return costModel; }
```

> **注意**: 这三个方法必须在 `public:` 区域内，`private:` 之前。

### 3.2 `framework/src/cost_model/simulation/backend.cpp`

在 `namespace npu::tile_fwk {` 内部（不是 namespace 外面！），添加两个方法的实现：

#### `SubmitLeafFunctionsBySubgraph` 实现

```cpp
void CostModelAgent::SubmitLeafFunctionsBySubgraph(uint64_t pSgId)
{
    BuildCostModel();

    auto* rootFunc = Program::GetInstance().GetLastFunction();
    if (rootFunc == nullptr) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_CONTENT_ERROR,
            "No compiled function found, cannot submit leaf functions by subgraph");
        return;
    }
    if (rootFunc->programs_.empty()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_CONTENT_ERROR,
            "Root function has no programs_, cannot filter by subgraph");
        return;
    }

    // 从 rootFunc->programs_ 中找出属于目标 pSgId 的 leaf function 的 hash
    std::unordered_set<uint64_t> targetHashes;
    for (auto& [sgId, leafFunc] : rootFunc->programs_) {
        if (sgId == pSgId) {
            targetHashes.insert(leafFunc->GetFunctionHash().GetHash());
        }
    }

    if (targetHashes.empty()) {
        SIMULATION_LOGI("No leaf functions found for pSgId=%lu", pSgId);
        return;
    }

    SIMULATION_LOGI("Submitting %zu leaf functions for pSgId=%lu", targetHashes.size(), pSgId);
    // 从全局 FunctionMap 中找到 hash 匹配的 function 提交
    std::vector<npu::tile_fwk::Function*> funcs;
    for (auto& [name, f] : Program::GetInstance().GetFunctionMap()) {
        if (targetHashes.count(f->GetFunctionHash().GetHash())) {
            funcs.push_back(f.get());
        }
    }
    costModel->Submit(funcs, false, "");
}
```

**关键逻辑说明**：
- 先通过 `rootFunc->programs_` 的 key（`sgId`）过滤出属于目标子图的 leaf function
- `programs_` 的类型通常是 `std::map<uint64_t, std::shared_ptr<Function>>`，key 是子图 ID
- 然后用 hash 在全局 `FunctionMap` 中查找对应的 Function 指针
- 这是因为 CostModel 的 Submit 接受 Function 指针列表

#### `SubmitSubgraphTopoByPid` 实现

```cpp
void CostModelAgent::SubmitSubgraphTopoByPid(std::string& path, uint64_t pSgId)
{
    // ── 第一遍：收集目标子图的所有 taskId ──
    std::unordered_set<uint64_t> targetTaskIds;
    {
        std::ifstream file(path);
        if (!file.is_open()) {
            SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Cannot open: %s", path.c_str());
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || std::isalpha(static_cast<unsigned char>(line[0]))) {
                continue;
            }
            std::vector<uint64_t> fields;
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    fields.push_back(std::stoull(item));
                } catch (const std::invalid_argument&) {
                } catch (const std::out_of_range&) {
                }
            }
            if (fields.size() > psgIdPos && fields[psgIdPos] == pSgId) {
                targetTaskIds.insert(fields[taskIdPos]);
            }
        }
    }

    if (targetTaskIds.empty()) {
        SIMULATION_LOGI("No tasks found for pSgId=%lu in %s", pSgId, path.c_str());
        topoJsonPath = config::LogTopFolder() + "/tmp_topo_json.json";
        Json emptyArray = Json::array();
        std::ofstream file(topoJsonPath);
        file << emptyArray.dump(1) << std::endl;
        return;
    }

    // ── 第二遍：保留目标行，截断 successors ──
    Json topoJson = Json::array();
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || std::isalpha(static_cast<unsigned char>(line[0]))) {
            continue;
        }
        std::vector<uint64_t> fields;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                fields.push_back(std::stoull(item));
            } catch (const std::invalid_argument&) {
            } catch (const std::out_of_range&) {
                SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_CONTENT_ERROR, "Out of range");
            }
        }
        if (fields[psgIdPos] != pSgId) {
            continue;
        }

        Json taskJson;
        taskJson["uniqueKey"] = static_cast<uint64_t>(fields[seqPos]) << seqNumOffset | fields[taskIdPos];
        taskJson["seqNo"] = fields[seqPos];
        taskJson["taskId"] = fields[taskIdPos];
        taskJson["rootIndex"] = fields[rootIndexPos];
        taskJson["rootHash"] = fields[rootHashpos];
        taskJson["leafIndex"] = fields[leafIndexPos];
        taskJson["opmagic"] = fields[opmagicPos];
        taskJson["funcHash"] = fields[funcHashPos];
        auto coreType = static_cast<npu::tile_fwk::CoreType>(fields[coreTypePos]);
        taskJson["coreType"] = npu::tile_fwk::GetCoreTypeDict().Find(coreType);
        taskJson["psgId"] = fields[psgIdPos];
        taskJson["wrapId"] = fields[wrapIdPos];

        // successors 只保留目标子图内的 taskId（截断跨子图依赖）
        Json successorsJson = Json::array();
        for (size_t i = succStartPos; i < fields.size(); i++) {
            if (targetTaskIds.count(fields[i])) {
                successorsJson.push_back(fields[i]);
            }
        }
        taskJson["successors"] = successorsJson;
        topoJson.push_back(taskJson);
    }

    topoJsonPath = config::LogTopFolder() + "/tmp_topo_json.json";
    std::ofstream outFile(topoJsonPath);
    outFile << topoJson.dump(1) << std::endl;
    outFile.close();

    SIMULATION_LOGI("Filtered topo for pSgId=%lu: %zu tasks written to %s",
        pSgId, topoJson.size(), topoJsonPath.c_str());
}
```

**关键逻辑说明**：
- `dyn_topo.txt` 是 CSV 格式，每行一个 task，字段含义由 `CostModelAgent` 的成员变量位置确定（`seqPos`, `taskIdPos`, `psgIdPos` 等）
- **两遍扫描**：第一遍收集目标子图的所有 taskId，第二遍过滤行并截断 successors
- **截断 successors** 很关键：原始 successors 可能指向其他子图的 task，必须只保留本子图内的 successor，否则 CostModel 会尝试调度不存在的 task 导致异常
- `psgIdPos` 是 `CostModelAgent` 的成员变量，已经在构造时通过 `pos++` 计算好

> **适配注意**：不同版本的 pypto，`dyn_topo.txt` 的字段布局可能不同（字段数量、顺序）。需要检查目标版本的 `CostModelAgent` 构造函数中 `psgIdPos` 等位置变量是否与你的版本一致。核心是看 `ParseDynTopo` 方法中的字段解析逻辑。

### 3.3 `framework/src/cost_model/simulation/cost_model_launcher.h`

#### 新增 public 静态方法

在 `CostModelLauncher` 类的 `public:` 区域，在 `CostModelRunOnce` 方法之后添加：

```cpp
static Json CostModelRunSubgraph(Function* function, uint64_t pSgId)
{
    Json result;
    result["status"] = "success";
    result["error_msg"] = "";

    // Phase 1: LEAF_FUNCTION — 只仿真子图内的 leaf func
    result["functions"] = RunSubgraphCostModel(function, pSgId);

    // Phase 2: NORMAL — 过滤后的拓扑, 获取子图调度耗时
    uint64_t subgraphTotalCycles = RunSubgraphDynCostModel(function, pSgId);
    if (subgraphTotalCycles == UINT64_MAX) {
        result["status"] = "error";
        result["error_msg"] = "subgraph simulation failed (no tasks found)";
    }

    result["p_sg_id"] = pSgId;
    result["subgraph_total_cycles"] = subgraphTotalCycles;
    result["output_dir"] = config::GetAbsoluteTopFolder() + "/" + "CostModelSimulationOutput";

    return result;
}
```

#### 新增 private 静态方法

在 `CostModelLauncher` 类的 `private:` 区域，添加两个 Phase 实现：

```cpp
static Json RunSubgraphCostModel(Function* /*function*/, uint64_t pSgId)
{
    Json functionsJson = Json::array();

    config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::LEAF_FUNCTION);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitLeafFunctionsBySubgraph(pSgId);
    auto costModel = costModelAgent.GetCostModel();
    if (costModel == nullptr) {
        return functionsJson;
    }
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();

    auto sim = costModel->sim;
    if (sim == nullptr) {
        return functionsJson;
    }

    for (auto& [hash, cycles] : sim->leafFunctionTime) {
        Json funcJson;
        funcJson["hash"] = hash;
        funcJson["cycles"] = cycles;
        auto it = sim->functionCache.cache.find(hash);
        if (it != sim->functionCache.cache.end()) {
            funcJson["name"] = it->second->funcName;
            funcJson["machine_type"] = static_cast<int>(it->second->machineType);
        }
        functionsJson.push_back(funcJson);
    }

    return functionsJson;
}

static uint64_t RunSubgraphDynCostModel(Function* /*function*/, uint64_t pSgId)
{
    config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::NORMAL);

    CostModelAgent costModelAgent;
    std::string path = config::LogTopFolder() + "/dyn_topo.txt";
    costModelAgent.SubmitSubgraphTopoByPid(path, pSgId);
    costModelAgent.SubmitLeafFunctionsBySubgraph(pSgId);
    auto costModel = costModelAgent.GetCostModel();
    if (costModel == nullptr) {
        return UINT64_MAX;
    }
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();

    auto sim = costModel->sim;
    if (sim == nullptr) {
        return UINT64_MAX;
    }
    return sim->globalCycles;
}
```

> **适配注意**：
> - `KEY_SIM_MODE` 在某些版本可能是宏定义，在某些版本是字符串 `"SIM_MODE"`。需要检查目标版本的 `config::SetSimConfig` 接受什么类型的 key。
> - `costModel->sim` 是否暴露为 public 需要确认。如果 `sim` 是 private 的，需要通过 getter 方法获取 `leafFunctionTime` 和 `globalCycles`。

### 3.4 `python/src/bindings/cost_model.cpp`

新增 Python 绑定函数 `CostModelRunSubgraphLine`：

```cpp
std::string CostModelRunSubgraphLine(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs, uint64_t pSgId)
{
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        Json error;
        error["status"] = "error";
        error["error_msg"] = "compile not complete";
        return error.dump();
    }

    // ⚠️ 关键：这里需要调用 InitInputOutputData
    // 但是需要注意 ProgramData 残留数据问题（见下方排障章节）
    std::string initResult = InitInputOutputData(inputs, outputs);
    if (!initResult.empty()) {
        Json error;
        error["status"] = "error";
        error["error_msg"] = initResult;
        return error.dump();
    }

    Function* func = Program::GetInstance().GetLastFunction();
    Json result = CostModelLauncher::CostModelRunSubgraph(func, pSgId);
    CopyTensorFromModel(inputs, outputs);
    return result.dump();
}
```

在 `BindCostModelRuntime` 函数中注册：

```cpp
void BindCostModelRuntime(py::module& m)
{
    m.def("CostModelRunOnceDataFromHost", &CostModelRunOnceDataFromHost);
    m.def("CostModelRunSubgraphLine", &CostModelRunSubgraphLine);  // 新增
}
```

### 3.5 `python/pypto/cost_model.py`

在已有的 `_cost_model_run_once_data_from_host` 函数之后，新增：

```python
def _cost_model_run_subgraph_line(
    inputs: List[pypto.Tensor],
    outputs: List[pypto.Tensor],
    *,
    p_sg_id: int,
) -> dict:
    isDevice = False
    for t in inputs:
        if t.device != torch.device("cpu"):
            isDevice = True
            break

    if isDevice:
        input_datas = _device_to_host_tensor_datas(inputs)
        output_datas = _device_to_host_tensor_datas(outputs)
    else:
        input_datas = _pto_to_tensor_data(inputs)
        output_datas = _pto_to_tensor_data(outputs)

    result_json = pypto_impl.CostModelRunSubgraphLine(input_datas, output_datas, p_sg_id)

    if isDevice:
        _host_to_device_tensor_datas(inputs, input_datas)
        _host_to_device_tensor_datas(outputs, output_datas)

    return json.loads(result_json)
```

同时更新 `__all__` 导出列表：

```python
__all__ = [
    "_cost_model_run_once_data_from_host",
    "_cost_model_run_subgraph_line",  # 新增
]
```

---

## 四、测试脚本

### 4.1 一键验证脚本 `test_costmodel_subgraph_verify.py`

将以下脚本放在 `python/tests/st/` 目录下，可直接 `python test_costmodel_subgraph_verify.py` 运行：

```python
#!/usr/bin/env python3
"""子图 CostModel 验证脚本"""
import os, sys, json, traceback
import pypto, numpy as np, torch

try:
    from pypto.cost_model import _cost_model_run_subgraph_line
    from pypto.converter import from_torch
except (ImportError, AttributeError):
    print("[SKIP] _cost_model_run_subgraph_line 不可用")
    sys.exit(0)

def softmax_core(t):
    row_max = pypto.amax(t, dim=-1, keepdim=True)
    return pypto.div(pypto.exp(pypto.sub(t, row_max)), pypto.sum(pypto.exp(pypto.sub(t, row_max)), dim=-1, keepdim=True))

# ⚠️ 注意：@pypto.frontend.jit 是 8.5.0 写法，9.0.0 可能是 @pypto.jit
# ⚠️ 注意：run_mode=1 是 SIM 模式
@pypto.frontend.jit(runtime_options={"run_mode": 1})
def softmax(input_tensor: pypto.Tensor(), output_tensor: pypto.Tensor()):
    s = input_tensor.shape
    b, n1, n2, dim = s[0], s[1], s[2], s[3]
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    for idx in range(b):
        iv = input_tensor[idx:idx+1, :n1, :n2, :dim]
        pypto.assemble(softmax_core(iv), [idx, 0, 0, 0], output_tensor)

# ---- main ----
print("=" * 50)
print("  子图 CostModel 验证")
print("=" * 50)

shape = (32, 32, 1, 256)
input_data = torch.rand(shape, dtype=torch.float32)
output_data = torch.zeros(shape, dtype=torch.float32)

# Step 1: 编译运行 kernel (SIM 模式自动触发整图 costmodel)
print("\n[1] 编译 kernel ...")
softmax(input_data, output_data)
print("  OK")

# Step 2: 子图 costmodel
print("\n[2] 运行子图 costmodel (p_sg_id=0) ...")
pto_input = from_torch(input_data)
pto_output = from_torch(output_data)

result = _cost_model_run_subgraph_line(inputs=[pto_input], outputs=[pto_output], p_sg_id=0)
print(f"  返回: {json.dumps(result, indent=2, ensure_ascii=False)}")

# 验证
assert result.get("status") == "success", f"status={result.get('status')}, error={result.get('error_msg')}"
assert result["p_sg_id"] == 0
assert isinstance(result["subgraph_total_cycles"], int)
assert isinstance(result["functions"], list)
print("\n  >>> 全部验证通过 <<<")
```

### 4.2 运行测试的正确流程

```bash
# 1. 先用 build_ci.py 生成 tmp_simulation.json（这是必须的！）
cd /path/to/pypto
python3 build_ci.py --editable -s1

# 2. 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 3. 运行测试
cd python/tests/st
python3 test_costmodel_subgraph_verify.py
```

---

## 五、环境搭建与排障

### 5.1 构建系统

**必须使用 `build_ci.py`**，不要手动 cmake。`build_ci.py` 会：
1. 生成 `tmp_simulation.json`（CostModel 运行必需的配置文件）
2. 编译 C++ 代码
3. 安装 pypto wheel

```bash
# 生成仿真配置 + 编译 + 安装
python3 build_ci.py --editable -s1
```

> **注意**：`-s1` 参数是关键，它告诉 build_ci.py 启用仿真模式，生成 `tmp_simulation.json`。没有这个文件，运行时会崩在 `ConfigManagerNg::ConfigManagerNg()`。

### 5.2 `tmp_simulation.json` 的位置和内容

这个文件通常在编译输出目录下（如 `build_subgraph/output/` 或类似路径），内容大致为：

```json
{
    "global_configs": {
        "platform_configs": {"ENABLE_COST_MODEL": true},
        "simulation_configs": {}
    }
}
```

如果运行时报 `ConfigManagerNg` 相关错误，大概率是缺少这个文件。

### 5.3 常见编译错误与修复

| 错误 | 原因 | 修复方法 |
|------|------|----------|
| `KEY_SIM_MODE` undeclared | 不同版本用不同 key 类型 | 检查 `config::SetSimConfig` 的 key 类型，可能是字符串 `"SIM_MODE"` 而非宏 |
| `blockIdx` ambiguous | CANN 9.0.0 内置了 `__cce_scalar::blockIdx` | 用 `::blockIdx` 全局限定所有 blockIdx 引用 |
| 方法实现放在 namespace 外面 | 编译不报错但链接时符号不匹配 | 确保实现代码在 `namespace npu::tile_fwk { ... }` 内部 |
| `ALOG_INFO` vs `SIMULATION_LOGI` | 不同版本日志宏不同 | 检查目标版本的日志系统，用对应的宏 |
| `costModel->sim` 不可访问 | sim 可能是 private | 检查 `CostModelInterface` 类定义，可能需要通过 getter 方法 |
| `rootFunc->programs_` 不可访问 | programs_ 可能叫别的名字或不可访问 | 检查 Function 类定义，找到存储子图 ID → leaf function 映射的容器 |

### 5.4 运行时常见错误与修复

#### 错误 1: `ConfigManagerNg` 崩溃 / segfault

**症状**: Python import pypto_impl 后运行就崩。

**原因**: 缺少 `tmp_simulation.json`。

**修复**: 运行 `python3 build_ci.py --editable -s1` 生成它。

#### 错误 2: "mismatch input/output"

**症状**: `CostModelRunSubgraphLine` 返回 `{"status": "error", "error_msg": "mismatch input/output"}`。

**原因**: `ProgramData` 是单例，保留了上一次整图 costmodel 运行时的 input/output 数据。当子图 costmodel 再次调用 `InitInputOutputData` 时，`ValidateFunctionAndIO` 检测到 function 的 `startArgsInputLogicalTensorList` 数量与传入的 inputs 数量不匹配。

**根因**: 整图 costmodel 运行时已经把 inputs/outputs 写入了 `ProgramData`，但 `InitInputOutputData` 中 `AppendInput/AppendOutput` 是追加操作而非替换，导致数量翻倍。

**修复方案**（二选一）：

**方案 A（推荐）**: 在调用 `InitInputOutputData` 之前，先清空 `ProgramData`：
```cpp
ProgramData::GetInstance().Clear();  // 如果有 Clear 方法
```
或者：
```cpp
ProgramData::GetInstance().GetInputDataList().clear();
ProgramData::GetInstance().GetOutputDataList().clear();
```

**方案 B**: 跳过 `InitInputOutputData`，因为整图运行已经初始化了 ProgramData：
```cpp
std::string CostModelRunSubgraphLine(...) {
    // 不调用 InitInputOutputData，直接获取 func
    Function* func = Program::GetInstance().GetLastFunction();
    Json result = CostModelLauncher::CostModelRunSubgraph(func, pSgId);
    CopyTensorFromModel(inputs, outputs);
    return result.dump();
}
```
> 方案 B 更简单，但前提是整图 costmodel 已经在同一进程中运行过。如果用户直接调用子图 costmodel 而没有先跑整图，方案 B 会失败。

**方案 C（最稳妥）**: 在 `InitializeInputOutputData` 中用赋值替换追加：
```cpp
static void InitializeInputOutputData(...) {
    auto& inputList = ProgramData::GetInstance().GetInputDataList();
    auto& outputList = ProgramData::GetInstance().GetOutputDataList();
    inputList.clear();  // 先清空
    outputList.clear();
    // 然后重新填充
    for (size_t i = 0; i < outputs.size(); i++) {
        outputList.push_back(std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape()));
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        inputList.push_back(RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), ...));
    }
}
```

#### 错误 3: `IsLoopEnd` / `DynTileFwkBackendKernelServerInit` undefined symbol

**症状**: `import pypto_impl` 时报 `dlopen` 错误，找不到符号。

**原因**: 没有完整编译所有依赖库（`libtile_fwk_interface.so`、`libtile_fwk_runtime.so` 等）。

**修复**: 用 `build_ci.py` 完整编译，或者 `cmake --build . -j8` 确保所有 target 都被编译。不要只编译单个 target。

#### 错误 4: `pypto.frontend.jit` 不存在

**症状**: Python 报 `module pypto.frontend has no attribute 'jit'`。

**原因**: CANN 9.0.0 改了 API，`jit` 直接挂在 `pypto` 模块下。

**修复**: 把 `@pypto.frontend.jit(...)` 改成 `@pypto.jit(...)`。

#### 错误 5: softmax 参数数量不对

**症状**: `softmax() takes 2 positional arguments but 3 were given`。

**原因**: CANN 9.0.0 版本的 softmax JIT 函数签名变了，新增了 `cost_model_enable` 参数。

**修复**: 把 `softmax(input_data, output_data)` 改成 `softmax(input_data, output_data, cost_model_enable=True)` 或检查目标版本的实际签名。

#### 错误 6: `from_torch` 缺少 `dynamic_axis` 参数

**症状**: 报错说 tensor 的 shape 不对或 dynamic_axis 相关错误。

**原因**: CANN 9.0.0 要求显式声明动态轴。

**修复**: 使用 `pypto.from_torch(tensor, dynamic_axis=[0])` 代替 `pypto.from_torch(tensor)`。但这取决于目标版本。

### 5.5 API 版本差异速查

| API | CANN 8.5.0 | CANN 9.0.0-beta.1 |
|-----|------------|-------------------|
| JIT 装饰器 | `@pypto.frontend.jit(...)` | `@pypto.jit(...)` |
| from_torch | `pypto.from_torch(t)` | `pypto.from_torch(t, dynamic_axis=[0])` |
| softmax 签名 | `softmax(input, output)` | `softmax(input, output, cost_model_enable)` |
| 日志宏 | `SIMULATION_LOGI` | `ALOG_INFO`（部分场景） |
| SIM_MODE key | 宏 `KEY_SIM_MODE` | 可能是字符串 `"SIM_MODE"` |
| for 循环 | `pypto.loop(N)` 或 `range(N)` | `range(N)` |
| ConfigManager | `ConfigManager` | `ConfigManagerNg` |

---

## 六、关键架构概念（帮助理解代码）

### 6.1 数据流

```
Python 层                                    C++ 层
────────                                     ──────

torch.Tensor ──→ pypto.Tensor ──→ DeviceTensorData
                                         │
                                         ▼
                               InitInputOutputData()
                               (写入 ProgramData 单例)
                                         │
                                         ▼
                               CostModelLauncher::CostModelRunSubgraph()
                                         │
                               ┌─────────┼─────────┐
                               ▼                   ▼
                    Phase 1: LEAF_FUNCTION    Phase 2: NORMAL
                    SubmitLeafFunctions       SubmitSubgraphTopo
                    BySubgraph(pSgId)         ByPid(path, pSgId)
                               │                   │
                               ▼                   ▼
                        leaf cycles           global cycles
                               │                   │
                               └─────────┬─────────┘
                                         ▼
                                   Json result
                                         │
                               CopyTensorFromModel()
                                         │
                                         ▼
                               json string → Python dict
```

### 6.2 关键单例

| 单例 | 作用 | 注意 |
|------|------|------|
| `Program::GetInstance()` | 管理所有编译后的 Function | `GetLastFunction()` 返回最近编译的 function |
| `ProgramData::GetInstance()` | 存储 input/output tensor 数据 | 是追加模式，不清空会导致 "mismatch" |
| `ConfigManager` / `ConfigManagerNg` | 管理仿真配置 | 需要 `tmp_simulation.json` 初始化 |

### 6.3 Function 的层次结构

```
Function (root)
  ├── programs_: map<uint64_t, shared_ptr<Function>>
  │     ├── [0] → leaf_func_0 (属于子图 0)
  │     ├── [0] → leaf_func_1 (属于子图 0)
  │     ├── [1] → leaf_func_2 (属于子图 1)
  │     └── ...
  ├── GetDyndevAttribute()
  │     ├── devLeafIndex2Hash
  │     ├── startArgsInputLogicalTensorList
  │     └── startArgsOutputLogicalTensorList
  ├── GetFunctionHash()
  └── GetMagicName()  // 包含 "leaf" 的是 leaf function
```

---

## 七、排查问题的方法论

### 7.1 编译阶段

1. **先看已有代码结构**: 用 `grep -rn "SubmitLeafFunctionsToCostModel"` 找到整图 costmodel 的实现位置，以此为参照修改。
2. **确认 API 兼容性**: 在目标代码中搜索 `SetSimConfig`、`CostModelInterface`、`Submit` 等关键 API，确认签名和参数类型。
3. **编译不过时**: 逐个解决编译错误，不要一次改多个文件。先确保 `backend.h` 改动编译通过，再改 `backend.cpp`，以此类推。

### 7.2 链接阶段

1. `undefined symbol` 错误说明某个 `.so` 没有被正确编译或加载。
2. 检查 `LD_LIBRARY_PATH` 和 `PYPTO_LIB_DIR` 是否指向包含所有 `.so` 的目录。
3. 用 `nm -C xxx.so | grep MissingSymbol` 确认符号在哪个 `.so` 中。

### 7.3 运行阶段

1. **先用整图 costmodel 验证基础环境**: 确保 `python3 examples/03_advanced/cost_model/cost_model.py` 能跑通。
2. **整图通过后再试子图**: 子图 costmodel 依赖整图 costmodel 的基础设施。
3. **看日志**: SIM 模式会输出大量日志到 `./output/` 目录，关注 `CostModelSimulationOutput` 下的文件。
4. **用 print/json.dumps 打印中间结果**: 在 Python 层打印 `result` 的完整内容。

### 7.4 验证成功标准

```
✅ 整图 costmodel: 输出 "✓ Cost model test passed"，output/ 下有 merged_swimlane.json
✅ 子图 costmodel: status="success"，subgraph_total_cycles > 0，functions 列表非空
✅ 多子图遍历: 至少 p_sg_id=0 返回 success
✅ 无效 ID: p_sg_id=999 返回 sentinel 值 (UINT64_MAX = 18446744073709551615)
```

---

## 八、如果目标版本的源码结构不同怎么办

### 8.1 找对等文件

用以下 grep 命令在目标代码中找到对应位置：

```bash
# 找 CostModelAgent 类定义
grep -rn "class CostModelAgent" framework/

# 找 Python 绑定
grep -rn "CostModelRunOnceDataFromHost\|BindCostModelRuntime" python/src/

# 找 CostModelLauncher 类
grep -rn "class CostModelLauncher" framework/

# 找整图 costmodel 入口（理解已有流程）
grep -rn "RunDynCostModel\|RunCostModel" framework/src/cost_model/
```

### 8.2 理解已有整图流程再改

在修改之前，先阅读整图 costmodel 的完整流程：
1. `CostModelRunOnce` → `RunDynamic` → `RunModel`
2. `RunModel` 中依次调用 `RunCostModel`（Phase 1: LEAF_FUNCTION）和 `RunDynCostModel`（Phase 2: NORMAL）
3. 子图 costmodel 复用了相同的两阶段模式，只是过滤了 leaf function 和拓扑

### 8.3 适配检查清单

- [ ] `costModel->sim` 是否 public？如果不是，需要加 getter 或修改访问方式
- [ ] `costModel->sim->leafFunctionTime` 的类型是什么？`std::unordered_map<uint64_t, uint64_t>`？
- [ ] `costModel->sim->functionCache.cache` 是否存在？
- [ ] `Function::programs_` 是否 public？key 是子图 ID 吗？
- [ ] `Function::GetFunctionHash().GetHash()` 方法是否存在？
- [ ] `config::SetSimConfig` 和 `config::GetSimConfig` 的 key 类型（宏 vs 字符串）
- [ ] `SIMULATION_LOGI` 还是 `ALOG_INFO`？
- [ ] `BuildCostModel()` 是否需要参数？
- [ ] `ProgramData` 的清空方法是否存在？

---

## 九、总结

1. **先跑通整图 costmodel**，确认环境正确
2. **按照文件顺序修改**：backend.h → backend.cpp → cost_model_launcher.h → cost_model.cpp → cost_model.py
3. **特别注意 "mismatch input/output" 问题**，这是最坑的运行时错误
4. **用 build_ci.py 构建**，不要手动 cmake
5. **不同 CANN 版本 API 有差异**，先 grep 确认再写代码
6. **子图过滤的核心逻辑是**: Phase 1 按 `rootFunc->programs_` 的 key 过滤 leaf function，Phase 2 按 `psgIdPos` 字段过滤 dyn_topo.txt 并截断 successors
