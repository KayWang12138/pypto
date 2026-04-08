---
name: pypto-perf-regression-bisect-finder
description: PyPTO 性能退化通用二分查找技能。通过自动对比基准版本和劣化版本的泳道图，提取性能退化特征，并利用二分法自动查找导致性能退化的 PR。
---

# PyPTO 性能退化通用二分查找

通过自动对比基准版本（Good）和当前版本（Bad）的泳道图，提取性能退化特征，并利用二分法自动查找导致性能退化的具体 Commit。

---

## 所需输入

### 测试案例路径
提供测试案例的相对或绝对路径：
- 例如：`models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py`
- 例如：`examples/02_intermediate/operators/softmax/softmax.py`

### 测试案例修改（可选）
如果测试案例在不同版本中需要保持运行一致性，请提供需要临时修改的内容（Agent 会在每次编译后自动应用这些修改）：
- 例如：修改某个参数的值，或注释掉某行代码。
- 如果没有需要修改的内容，则跳过此步骤。

### 二分起始版本
提供性能正常的起始 commit hash（Good 版本）。

### 二分结束版本
提供性能退化的结束 commit hash（Bad 版本），默认为 `HEAD`。

---

## 工作流程

### 阶段 1：获取基准性能并自动提取劣化特征

本阶段旨在自动找出 Bad 版本比 Good 版本慢在哪里，并确立二分判定标准。

1. **环境准备与运行 (Good 版本)**：
   - 切换到 `$GOOD_COMMIT`。
   - 编译并安装 PyPTO whl 包。
   - 检查并开启泳道图配置：扫描代码中包含 `jit` 的装饰器，若无配置则自动注入 `debug_options={"runtime_debug_mode": 1}`。
   - 应用用户提供的测试案例修改（确保用例可跑）。
   - 清理旧的 `output` 目录（如果存在）。
   - 运行用例，生成泳道图文件至默认 `output/` 目录。
   - 将 `output` 目录重命名为 `output_good`，便于后续分析。

2. **环境准备与运行 (Bad 版本)**：
   - 切换到 `$BAD_COMMIT`。
   - 严格重复上述编译、配置注入、代码修改和运行步骤。
   - 清理旧的 `output` 目录（如果存在）。
   - 运行用例，生成泳道图文件至默认 `output/` 目录。
   - 将 `output` 目录重命名为 `output_bad`，便于后续分析。

3. **批次间隙提取**：
   - 对 Good 版本调用 `scripts/analyze_batch_gaps.py`：
     ```bash
     python3 scripts/analyze_batch_gaps.py \
       --swimlane output_good/merged_swimlane.json \
       --topo output_good/dyn_topo.txt \
       --output output_good/batch_gaps.json
     ```
   - 对 Bad 版本调用 `scripts/analyze_batch_gaps.py`：
     ```bash
     python3 scripts/analyze_batch_gaps.py \
       --swimlane output_bad/merged_swimlane.json \
       --topo output_bad/dyn_topo.txt \
       --output output_bad/batch_gaps.json
     ```
   - 该脚本会分析 AIC/AIV 任务批次转换点的依赖间隙，记录每一个独立间隙。

4. **自动 Diff 与特征提取**：
   - 调用 `scripts/auto_diff.py --diff` 模式自动对比两份批次间隙数据：
     ```bash
     python3 scripts/auto_diff.py \
       --diff \
       --good output_good/batch_gaps.json \
       --bad output_bad/batch_gaps.json \
       --config bisect_condition.json
     ```
   - **分析逻辑**：
     - 遍历比对每一个对应的批次转换点间隙
     - 找出劣化幅度最大的批次间隙作为二分判定标尺
     - 生成 `bisect_condition.json`，格式如下：
       ```json
       {
         "target_bottleneck": {
           "transition_index": <index>,
           "type": "AIC -> AIV" 或 "AIV -> AIC",
           "from_task_id": <task_id>,
           "good_us": <good_gap_us>,
           "bad_us": <bad_gap_us>,
           "threshold": <threshold_us>
         }
       }
       ```
   - **判定标准**：该阈值是 Good 和 Bad 版本该批次间隙的中点，后续二分时当前版本的该批次间隙 <= threshold 判定为 good，否则判定为 bad。

### 阶段 2：自动化二分查找 (Git Bisect)

使用 `git bisect` 启动自动查找。在每次迭代（每个被 checkout 的 commit）中，严格执行以下 6 步标准循环：

1. **二分代码版本**：Git 自动跳转到二分计算出的中间 commit。
2. **编译并安装 whl 包**：
   - 清理旧构建目录。
   - 忽略外部环境变量，使用默认稳定配置执行编译 `python3 build_ci.py -f=python3`。
   - 强制覆盖安装生成的whl包 `pip install build_out/pypto*whl --force-reinstall --no-deps`。
3. **检查并开启泳道图配置**：解析测试用例 AST 或正则匹配，确保 `jit` 装饰器中 `runtime_debug_mode: 1` 已开启。
4. **统一测试案例**：应用用户在输入中提供的代码修改，确保本轮测试案例逻辑与阶段 1 完全一致。
5. **运行与分析**：
   - 清理旧的 `output` 目录（如果存在）。
   - 运行测试案例，生成泳道图文件至默认 `output/` 目录。
   - 获取当前 commit hash，将 `output` 目录重命名为 `output_<commit_hash>`，便于后续分析。
   - 调用 `scripts/analyze_batch_gaps.py` 分析当前版本的泳道图：
     ```bash
     python3 scripts/analyze_batch_gaps.py \
       --swimlane output_<commit_hash>/merged_swimlane.json \
       --topo output_<commit_hash>/dyn_topo.txt \
       --output output_<commit_hash>/batch_gaps.json
     ```
   - 调用 `scripts/auto_diff.py --evaluate` 模式评估当前版本：
     ```bash
     python3 scripts/auto_diff.py \
       --evaluate \
       --current output_<commit_hash>/batch_gaps.json \
       --config bisect_condition.json
     ```
   - 该脚本会自动读取 `bisect_condition.json` 中的目标瓶颈信息，检查当前版本在该批次转换点的间隙值。
6. **得出单步结论**：
   - `auto_diff.py` 会自动返回退出码：
     - `0`（判定为 good）：当前批次间隙 <= threshold
     - `1`（判定为 bad）：当前批次批次间隙 > threshold
     - `125`（Skip）：批次间隙数据异常或流水线签名发生变化

### 阶段 3：结果分析与根因输出

二分过程结束后，输出最终分析报告：

1. **定位问题提交**：
   - 提取 `git bisect log` 中的第一个 bad commit。
   - 输出该 commit 的 Hash、提交信息、作者和时间。
   - 打印该 commit 的代码变更概要 (`git show --stat`)。
2. **劣化特征复盘**：
   - 指出该 commit 触发了哪项性能劣化指标（基于阶段 1 提取的特征）。
3. **优化建议**：
   - 结合变更代码，提示可能的性能退化根因（如：调度策略变更引入额外 Bubble、新同步机制增加了 AIV 等待时间、指令生成优化失效等）。

---

## 输出说明

### 目录结构管理
二分查找过程中生成的数据将按 commit 进行隔离保存，便于事后复核：

**重要说明**：
- 所有测试用例运行时，泳道图文件默认保存到 `output/` 目录
- 运行完成后，需要将 `output/` 目录重命名并添加相应后缀，便于后续分析

```text
workspace/
├── output_good/                 # Good 版本基准数据（由 output/ 重命名而来）
│   ├── merged_swimlane.json
│   ├── dyn_topo.txt
│   └── batch_gaps.json
├── output_bad/                  # Bad 版本基准数据（由 output/ 重命名而来）
│   ├── merged_swimlane.json
│   ├── dyn_topo.txt
│   └── batch_gaps.json
├── bisect_condition.json        # 自动提取的 Diff 判定阈值（包含目标瓶颈信息）
├── output_<commit_hash>/        # 每次二分迭代的临时输出（由 output/ 重命名而来）
│   ├── merged_swimlane.json
│   ├── dyn_topo.txt
│   └── batch_gaps.json
└── final_bisect_report.txt      # 最终诊断报告
```

### 核心脚本说明

1. **analyze_batch_gaps.py**：批次间隙提取脚本
   - 输功能：分析泳道图和拓扑文件，提取 AIC/AIV 任务批次转换点的依赖间隙
   - 用法：
     ```bash
     python3 scripts/analyze_batch_gaps.py \
       --swimlane <merged_swimlane.json.json> \
       --topo <dyn_topo.txt> \
       --output <batch_gaps.json>
     ```

2. **auto_diff.py**：自动 Diff 和评估脚本
   - **--diff 模式**：对比 Good 和 Bad 版本的批次间隙，找出劣化幅度最大的瓶颈
     ```bash
     python3 scripts/auto_diff.py \
       --diff \
       --good <good_batch_gaps.json> \
       --bad <bad_batch_gaps.json> \
       --config bisect_condition.json
     ```
   - **--evaluate 模式**：评估当前版本的批次间隙，返回判定结果
     ```bash
     python3 scripts/auto_diff.py \
       --evaluate \
       --current <current_batch_gaps.json> \
       --config bisect_condition.json
     ```