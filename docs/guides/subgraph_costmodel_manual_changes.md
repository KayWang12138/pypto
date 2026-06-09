# 子图 CostModel 手动同步操作手册

> 以下是在目标环境（可能不同 CANN 版本、不同源码基线）上手动同步子图 costmodel 功能的完整步骤。
>
> 每一步都标注了：**改哪个文件** → **找到哪个位置** → **插入/替换什么代码**。

---

## 先确认目标环境的文件结构

在目标服务器上执行：

```bash
grep -rn "class CostModelAgent" framework/
grep -rn "CostModelRunOnce" framework/
```

- **有 `CostModelAgent` 且有 `CostModelRunOnce`** → 存在 `CostModelLauncher` 类，走 **方案 A**
- **有 `CostModelAgent` 但没有 `CostModelRunOnce`** → 没有 `CostModelLauncher` 类，走 **方案 B**

---

# ============================================================
# 方案 A：目标环境有 CostModelLauncher（有 cost_model_launcher.h）
# ============================================================

需要改 **6 个源文件** + **新增 3 个测试文件**。

---

## 文件 1: `framework/src/cost_model/simulation/backend.h`

**操作**: 在 `CostModelAgent` 类的 `public:` 区域，`SubmitLeafFunctionsToCostModel()` 声明之后，新增 3 行声明。

**找到**:
```cpp
    void SubmitLeafFunctionsToCostModel();
```

**在它下面插入**:
```cpp
    void SubmitLeafFunctionsBySubgraph(uint64_t pSgId);
    void SubmitSubgraphTopoByPid(std::string& path, uint64_t pSgId);
    std::shared_ptr<CostModel::CostModelInterface> GetCostModel() const { return costModel; }
```

---

## 文件 2: `framework/src/cost_model/simulation/backend.cpp`

**操作**: 新增 `#include` + 新增两个方法实现。

### 2a. 新增头文件

**找到**:
```cpp
#include <cctype>
```

**在它下面插入**:
```cpp
#include <unordered_set>
```

### 2b. 新增 `SubmitLeafFunctionsBySubgraph` 方法

**找到** `GetFunctionFromJson` 方法的结尾（一般在 `ExecuteSimulation` 函数前面），在它后面、`extern "C" int32_t ExecuteSimulation` 前面，插入以下完整方法：

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
    std::vector<npu::tile_fwk::Function*> funcs;
    for (auto& [name, f] : Program::GetInstance().GetFunctionMap()) {
        if (targetHashes.count(f->GetFunctionHash().GetHash())) {
            funcs.push_back(f.get());
        }
    }
    costModel->Submit(funcs, false, "");
}
```

> **适配注意**: 如果目标版本的日志宏不是 `SIMULATION_LOGI` / `SIMULATION_LOGE`，需要换成目标版本对应的宏（如 `ALOG_INFO`）。

### 2c. 新增 `SubmitSubgraphTopoByPid` 方法

紧接着上面的方法后面，继续插入：

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

> **⚠️ 重要**: 这两个方法必须放在 `namespace npu::tile_fwk { ... }` 内部，不要放到 namespace 外面！

---

## 文件 3: `framework/src/cost_model/simulation/cost_model_launcher.h`

**操作**: 新增 1 个 public 静态方法 + 2 个 private 静态方法。

### 3a. 新增 public 方法 `CostModelRunSubgraph`

在 `CostModelLauncher` 类中，找到已有的 `CostModelRunOnce` (两个重载) 之后，`private:` 之前，插入：

```cpp
    static Json CostModelRunSubgraph(Function* function, uint64_t pSgId)
    {
        Json result;
        result["status"] = "success";
        result["error_msg"] = "";

        // Phase 1: LEAF_FUNCTION — 只仿真子图内的 leaf func, 收集 functionTime
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

### 3b. 新增 private 方法 `RunSubgraphCostModel` 和 `RunSubgraphDynCostModel`

在 `CostModelLauncher` 类的 `private:` 区域，找到已有的 `RunDynCostModel()` 方法之后，插入：

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

> **适配注意**:
> - `KEY_SIM_MODE` 在某些版本可能是字符串 `"SIM_MODE"`，需检查目标版本的 `config::SetSimConfig` 用法
> - `costModel->sim` 如果是 private，需查看 `CostModelInterface` 类是否有 getter
> - `sim->functionCache.cache` 如果不可访问，需找替代方式获取 function 名称

---

## 文件 4: `python/src/bindings/cost_model.cpp`

**操作**: 新增 `CostModelRunSubgraphLine` 函数 + 注册绑定 + 应用关键修复。

### 4a. 新增函数

在文件末尾的 `} // namespace pypto` 之前，找到已有的 `CostModelRunOnceDataFromHost` 函数之后，`BindCostModelRuntime` 函数之前，插入：

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

    // ⚠️ 关键修复：跳过 InitInputOutputData
    // ProgramData 已由整图 costmodel 运行时填充，再次调用会追加导致 "mismatch input/output"
    Function* func = Program::GetInstance().GetLastFunction();
    if (func == nullptr) {
        Json error;
        error["status"] = "error";
        error["error_msg"] = "no compiled function found";
        return error.dump();
    }

    Json result = CostModelLauncher::CostModelRunSubgraph(func, pSgId);
    CopyTensorFromModel(inputs, outputs);
    return result.dump();
}
```

> **⚠️ 这是最关键的修复点！** 不要调用 `InitInputOutputData(inputs, outputs)`，否则在整图 costmodel 运行后再调子图会报 "mismatch input/output"。

### 4b. 注册绑定

找到 `BindCostModelRuntime` 函数，**替换整个函数**为：

```cpp
void BindCostModelRuntime(py::module& m)
{
    m.def("CostModelRunOnceDataFromHost", &CostModelRunOnceDataFromHost);
    m.def("CostModelRunSubgraphLine", &CostModelRunSubgraphLine);
}
```

---

## 文件 5: `python/pypto/cost_model.py`

**操作**: 修改 import + 新增导出 + 新增函数。

### 5a. 修改 import

**找到**:
```python
from typing import List, overload
```

**替换为**:
```python
from typing import List

import json
```

> 如果目标版本已经是 `from typing import List` 且已有 `import json`，则跳过此步。

### 5b. 新增导出

**找到**:
```python
__all__ = [
    "_cost_model_run_once_data_from_host",
]
```

**替换为**:
```python
__all__ = [
    "_cost_model_run_once_data_from_host",
    "_cost_model_run_subgraph_line",
]
```

### 5c. 新增函数

在文件末尾（`_cost_model_run_once_data_from_host` 函数之后），追加：

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

---

## 文件 6（可选）: `framework/src/cost_model/simulation/base/ModelTop.h` + `ModelTop.cpp`

> 这是 pipeline_summary.csv 输出功能，与子图 costmodel 核心逻辑无关。如不需要可跳过。

### 6a. ModelTop.h — 新增声明

**找到**:
```cpp
    void PrintCoreStat();
    void PrintStat();
```

**替换为**:
```cpp
    void PrintCoreStat();
    void PrintPipelineSummaryCsv();
    void PrintStat();
```

### 6b. ModelTop.cpp — 新增 include

**找到**:
```cpp
#include <cstdlib>
#include <iostream>
```

**替换为**:
```cpp
#include <cstdlib>
#include <iostream>
#include <fstream>
```

### 6c. ModelTop.cpp — 新增方法

在 `PrintCoreStat()` 方法之后、`PrintStat()` 方法之前，插入：

```cpp
void SimSys::PrintPipelineSummaryCsv()
{
    std::string outPath = outdir + "/pipeline_summary.csv";
    std::ofstream ofs(outPath);
    if (!ofs.is_open()) {
        SIMULATION_LOGW("Failed to open pipeline_summary.csv for write: %s", outPath.c_str());
        return;
    }

    // CSV header
    ofs << "CoreType,CoreIdx";
    for (int pt = 0; pt < static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE); ++pt) {
        ofs << "," << CorePipeName(static_cast<CorePipeType>(pt));
    }
    ofs << "\n";

    // Per-core rows
    std::vector<MachineType> coreTypes = {
        MachineType::AIC, MachineType::AIV, MachineType::MIXAICORE
    };
    for (auto mtype : coreTypes) {
        for (auto& machine : machineGroup[static_cast<int>(mtype)]) {
            auto corePtr = std::dynamic_pointer_cast<CoreMachine>(machine);
            int coreIdx = GetMachineSeq(corePtr->machineId);
            ofs << MachineName(mtype) << ",Core-" << coreIdx;
            for (int pt = 0; pt < static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE); ++pt) {
                uint64_t cycles = 0;
                auto it = corePtr->stats->totalPipeUseCycles.find(pt);
                if (it != corePtr->stats->totalPipeUseCycles.end()) {
                    cycles = it->second;
                }
                ofs << "," << cycles;
            }
            ofs << "\n";
        }
    }
    ofs.close();
    SIMULATION_LOGW("Pipeline Summary CSV: %s", outPath.c_str());
}
```

### 6d. ModelTop.cpp — 在 PrintStat 中调用

在 `PrintStat()` 方法中，找到 `PrintCoreStat();` 调用，在它后面加一行：

```cpp
    PrintCoreStat();
    PrintPipelineSummaryCsv();    // ← 新增这行
```

---
---
# ============================================================
# 方案 B：目标环境没有 CostModelLauncher（没有 cost_model_launcher.h）
# ============================================================

> 如果 `grep -rn "CostModelRunOnce" framework/` 搜不到结果，说明没有 `CostModelLauncher` 类。
> 此时所有逻辑都放在 `CostModelAgent`（backend.h / backend.cpp）里。

需要改 **4 个源文件** + **新增 3 个测试文件**。文件 1、2、5 与方案 A 完全相同，**跳过文件 3**，文件 4 的调用方式不同。

---

## 文件 1: `backend.h` — 同方案 A，但多加一行声明

**找到**:
```cpp
    void SubmitLeafFunctionsToCostModel();
```

**在它下面插入**:
```cpp
    void SubmitLeafFunctionsBySubgraph(uint64_t pSgId);
    void SubmitSubgraphTopoByPid(std::string& path, uint64_t pSgId);
    std::shared_ptr<CostModel::CostModelInterface> GetCostModel() const { return costModel; }
    static Json CostModelRunSubgraph(uint64_t pSgId);
```

> 比方案 A 多了最后一行 `static Json CostModelRunSubgraph(uint64_t pSgId);`，这是子图的总入口。

---

## 文件 2: `backend.cpp` — 同方案 A + 新增 `CostModelRunSubgraph` 方法

### 2a. 新增头文件、2b. `SubmitLeafFunctionsBySubgraph`、2c. `SubmitSubgraphTopoByPid`

与方案 A 完全相同，照搬。

### 2d. 新增 `CostModelRunSubgraph` 方法（方案 B 独有）

在 `SubmitSubgraphTopoByPid` 方法之后、`extern "C" int32_t ExecuteSimulation` 之前，插入：

```cpp
Json CostModelAgent::CostModelRunSubgraph(uint64_t pSgId)
{
    Json result;
    result["status"] = "success";
    result["error_msg"] = "";

    // Phase 1: LEAF_FUNCTION — 子图内 leaf func 的 cycles
    config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::LEAF_FUNCTION);

    CostModelAgent agent1;
    agent1.SubmitLeafFunctionsBySubgraph(pSgId);
    auto costModel = agent1.GetCostModel();

    Json functionsJson = Json::array();
    if (costModel != nullptr) {
        agent1.RunCostModel();
        agent1.TerminateCostModel();

        if (costModel->sim != nullptr) {
            for (auto& [hash, cycles] : costModel->sim->leafFunctionTime) {
                Json funcJson;
                funcJson["hash"] = hash;
                funcJson["cycles"] = cycles;
                auto it = costModel->sim->functionCache.cache.find(hash);
                if (it != costModel->sim->functionCache.cache.end()) {
                    funcJson["name"] = it->second->funcName;
                    funcJson["machine_type"] = static_cast<int>(it->second->machineType);
                }
                functionsJson.push_back(funcJson);
            }
        }
    }
    result["functions"] = functionsJson;

    // Phase 2: NORMAL — 子图拓扑调度 cycles
    config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::NORMAL);

    CostModelAgent agent2;
    std::string path = config::LogTopFolder() + "/dyn_topo.txt";
    agent2.SubmitSubgraphTopoByPid(path, pSgId);
    agent2.SubmitLeafFunctionsBySubgraph(pSgId);
    auto costModel2 = agent2.GetCostModel();

    uint64_t subgraphTotalCycles = UINT64_MAX;
    if (costModel2 != nullptr) {
        agent2.RunCostModel();
        agent2.TerminateCostModel();
        if (costModel2->sim != nullptr) {
            subgraphTotalCycles = costModel2->sim->globalCycles;
        }
    }

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

> **适配注意**:
> - `KEY_SIM_MODE` 在某些版本可能是字符串 `"SIM_MODE"`，需检查 `config::SetSimConfig` 用法
> - `costModel->sim` 如果是 private，需查看 `CostModelInterface` 类是否有 getter
> - `sim->leafFunctionTime`、`sim->functionCache.cache`、`sim->globalCycles` 需确认可访问

---

## ~~文件 3: 跳过~~

没有 `cost_model_launcher.h` 就不需要改这个文件。

---

## 文件 4: `python/src/bindings/cost_model.cpp`

### 4a. 新增函数

在 `BindCostModelRuntime` 函数之前，插入：

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

    // ⚠️ 关键修复：跳过 InitInputOutputData
    // ProgramData 已由整图 costmodel 运行时填充，再次调用会追加导致 "mismatch input/output"
    Function* func = Program::GetInstance().GetLastFunction();
    if (func == nullptr) {
        Json error;
        error["status"] = "error";
        error["error_msg"] = "no compiled function found";
        return error.dump();
    }

    // 方案 B 区别：直接调 CostModelAgent::CostModelRunSubgraph，不走 CostModelLauncher
    Json result = CostModelAgent::CostModelRunSubgraph(pSgId);
    CopyTensorFromModel(inputs, outputs);
    return result.dump();
}
```

> **方案 A vs B 的唯一区别就在这一行**：
> - 方案 A: `CostModelLauncher::CostModelRunSubgraph(func, pSgId)`
> - 方案 B: `CostModelAgent::CostModelRunSubgraph(pSgId)`

### 4b. 注册绑定（同方案 A）

```cpp
void BindCostModelRuntime(py::module& m)
{
    m.def("CostModelRunOnceDataFromHost", &CostModelRunOnceDataFromHost);
    m.def("CostModelRunSubgraphLine", &CostModelRunSubgraphLine);
}
```

---

## 文件 5: `python/pypto/cost_model.py` — 同方案 A

完全相同，照搬。

---
---

# ============================================================
# 公共部分：测试脚本（方案 A / B 通用）
# ============================================================

以下 3 个文件是新增的，直接复制到目标环境即可。

### 测试文件放置位置

```
python/tests/st/test_costmodel_subgraph.py           # 子图基础测试
python/tests/st/test_costmodel_subgraph_verify.py    # 子图完整验证
python/tests/st/test_costmodel_wholegraph.py          # 整图验证
```

### 运行方法

```bash
# 1. 用 build_ci.py 编译（会生成必需的 tmp_simulation.json）
python3 build_ci.py --editable -s1

# 2. 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 3. 运行整图验证
python3 python/tests/st/test_costmodel_wholegraph.py

# 4. 运行子图验证
python3 python/tests/st/test_costmodel_subgraph_verify.py
```

---

## 注意事项

1. **"mismatch input/output" 修复是必须的**（文件 4 中跳过 `InitInputOutputData`），否则子图 costmodel 跑不通
2. **所有 C++ 方法实现必须在 `namespace npu::tile_fwk { }` 内部**
3. **编译用 `build_ci.py --editable -s1`**，不要手动 cmake（会缺 `tmp_simulation.json`）
4. **不同 CANN 版本 API 可能不同**，以下需检查：
   - `KEY_SIM_MODE` 是宏还是字符串 `"SIM_MODE"`
   - 日志宏是 `SIMULATION_LOGI` 还是 `ALOG_INFO`
   - `costModel->sim` 是否 public
   - JIT 装饰器是 `@pypto.frontend.jit` 还是 `@pypto.jit`
5. **子图 costmodel 依赖整图先运行**（生成 `dyn_topo.txt`），不能单独使用
