---
name: pypto-perf-regression-bisect-finder
description: PyPTO 性能退化通用二分查找技能。通过二分法查找导致性能退化的 PR，支持用户自定义性能现象、分析方法和判断标准。
---

# PyPTO 性能退化通用二分查找

通过二分法查找导致性能退化的 PR，支持用户自定义性能现象、分析方法和判断标准。

---

## 所需输入

### 性能现象描述

描述观察到的性能现象：
- 例如："泳道图中 AIC 和 AIV 之间的间隙变大"
- 例如："测试用例总执行时间变长"
- 例如："核心利用率下降"
- 例如："气泡等待时间占比增加"

### 泳道图分析方法

说明如何从泳道图或日志中提取性能指标：
- 例如："计算最后一个 AIC 完成时间和第一个 AIV 开始时间的差值"
- 例如："计算第一个 AIC 完成时间和第一个 AIV 开始时间的差值"
- 例如："计算测试用例的总执行时间"
- 例如："计算所有核心的平均利用率"

参考 [references/analysis-methods.md](references/analysis-methods.md) 了解常用分析方法。

### 性能判断标准

定义性能判断标准：
- Good 版本的指标范围：[例如：间隙 < 8.0 微秒]
- Bad 版本的指标范围：[例如：间隙 >= 10.0 微秒]
- 或提供判断脚本：[例如：自定义 Python 脚本]

### 测试案例

提供测试案例路径：
- 例如：models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py
- 例如：examples/02_intermediate/operators/softmax/softmax.py

### 版本范围

提供版本范围：
- Good 版本（基准版本）：[commit hash]
- Bad 版本（问题版本）：[commit hash 或 HEAD]

### 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [references/analysis-methods.md](references/analysis-methods.md) | 常用性能分析方法的详细说明和示例代码 | 阶段 3 读取 |
| [scripts/analyze_performance.py](scripts/analyze_performance.py) | 性能分析脚本，根据用户定义的方法提取性能指标 | 阶段 4 执行 |
| [scripts/bisect_test.sh](scripts/bisect_test.sh) | git bisect 测试脚本，编译、运行测试和分析性能 | 阶段 4 执行 |

---

## 工作流程

### 阶段 1：信息收集

与用户交互，收集以下信息：

1. **性能现象描述**：用户描述观察到的性能现象
2. **泳道图分析方法**：用户说明如何从泳道图提取性能指标
3. **性能判断标准**：用户定义 Good/Bad 版本的阈值范围
4. **测试案例**：用户提供测试案例路径
5. **版本范围**：用户提供 Good 和 Bad 版本的 commit hash

### 阶段 2：环境准备

使用 `pypto-environment-setup` skill 完成环境准备。

#### 2.1 环境检查

```bash
# 检查 NPU 环境
npu-smi info

# 检查 PyPTO 安装
python3 -c "import pypto; print(pypto.__version__)"

# 检查 Git 仓库状态
git status
git log --oneline -5
```

### 阶段 3：分析脚本生成

根据用户提供的信息，生成性能分析脚本。

#### 3.1 使用内置分析方法

`scripts/analyze_performance.py` 提供了以下内置分析方法：

| 方法名 | 描述 |
|--------|------|
| `aic_last_to_aiv_first` | 计算最后一个 AIC 完成时间和第一个 AIV 开始时间的差值 |
| `aic_first_to_aiv_first` | 计算第一个 AIC 完成时间和第一个 AIV 开始时间的差值 |
| `total_time` | 计算泳道图中所有事件的总时间跨度 |

#### 3.2 使用自定义分析脚本

如果内置方法不满足需求，用户可以提供自定义分析脚本。自定义脚本需要：

1. 读取泳道图或日志文件
2. 提取性能指标
3. 根据判断标准返回退出码（0=GOOD, 1=BAD, 2=UNKNOWN）

### 阶段 4：二分查找

#### 4.1 使用 git bisect

```bash
# 设置环境变量（根据用户提供的信息填写）
export TEST_CASE="<测试案例路径>"
export ANALYSIS_METHOD="<分析方法>"
export GOOD_THRESHOLD=<Good 版本阈值>
export BAD_THRESHOLD=<Bad 版本阈值>
export GOOD_COMMIT=<Good 版本 commit>
export BAD_COMMIT=<Bad 版本 commit>

# 初始化 bisect
git bisect start $GOOD_COMMIT $BAD_COMMIT

# 运行 bisect
git bisect run ./scripts/bisect_test.sh
```

#### 4.2 手动二分

如果 git bisect 不适用，则手动进行二分：

1. 获取版本列表：`git log --oneline $GOOD_COMMIT..$BAD_COMMIT`
2. 对每个版本：
   - 切换版本：`git checkout <commit>`
   - 编译：`python3 build_ci.py -f=python3`
   - 运行测试：`python3 $TEST_CASE`
   - 保存输出到 `output_<commit>` 目录
   - 分析性能：`python3 scripts/analyze_performance.py`
   - 判断版本类型（GOOD/BAD/UNKNOWN）
3. 根据判断结果调整二分范围

### 阶段 5：结果分析

#### 5.1 定位问题提交

```bash
# 获取 bisect 结果
problem_commit=$(git bisect log | grep "first bad commit" | awk '{print $4}')

echo "问题提交: $problem_commit"
echo "提交信息: $(git log --format="%s" -1 $problem_commit)"
echo "提交作者: $(git log --format="%an <%ae>" -1 $problem_commit)"
echo "提交时间: $(git log --format="%ai" -1 $problem_commit)"
```

#### 5.2 查看代码变更

```bash
# 查看问题提交的代码变更
git show $problem_commit

# 对比 Good 和 Bad 版本的所有差异
git diff $GOOD_COMMIT $BAD_COMMIT
```

#### 5.3 性能对比

```bash
# 测试 Good 版本
git checkout $GOOD_COMMIT
rm -rf build_out
python3 build_ci.py -f=python3
pip install build_out/pypto*.whl --force-reinstall --no-deps
rm -rf output/*
python3 $TEST_CASE
python3 scripts/analyze_performance.py \
    --output-dir output/ \
    --analysis-method "$ANALYSIS_METHOD" \
    --good-threshold $GOOD_THRESHOLD \
    --bad-threshold $BAD_THRESHOLD

# 测试 Bad 版本
git checkout $BAD_COMMIT
rm -rf build_out
python3 build_ci.py -f=python3
pip install build_out/pypto*.whl --force-reinstall --no-deps
rm -rf output/*
python3 $TEST_CASE
python3 scripts/analyze_performance.py \
    --output-dir output/ \
    --analysis-method "$ANALYSIS_METHOD" \
    --good-threshold $GOOD_THRESHOLD \
    --bad-threshold $BAD_THRESHOLD
```

---

## 输出

### 输出目录结构

二分查找过程中，每个测试版本的结果会保存到独立的目录：

```
output_<commit_hash>/
├── output_*/              # 原始输出目录
│   ├── merged_swimlane.json
│   ├── bubble_analysis.log
│   └── ...
└── commit_info.txt        # Commit 信息
```

`commit_info.txt` 包含：
- Commit hash
- Commit message
- Commit date
- Commit author

### 最终报告

向用户输出以下信息：

1. **问题提交信息**：
   - Commit hash
   - Commit message
   - Commit author
   - Commit date

2. **性能对比**：
   - Good 版本性能指标
   - Bad 版本性能指标
   - 性能退化百分比

3. **代码变更摘要**：
   - 问题提交的代码变更
   - Good 和 Bad 版本之间的差异

4. **优化建议**：
   - 基于代码变更的性能优化建议

---

## 错误处理

- **编译失败**：跳过该版本，返回 UNKNOWN，继续二`分
- **测试失败**：跳过该版本，返回 UNKNOWN，继续二分
- **分析失败**：跳过该版本，返回 UNKNOWN，继续二分
- **Git bisect 失败**：尝试手动二分

---

## 注意事项

1. **编译时间**：每个版本需要重新编译，耗时较长
2. **测试稳定性**：某些版本可能编译失败或无法运行，需要跳过
3. **性能波动**：性能指标可能有正常波动，建议多次测试取平均值
4. **环境一致性**：确保测试环境在不同版本间保持一致
5. **分析方法准确性**：确保分析方法能够准确反映性能现象
6. **输出目录管理**：每次测试会创建 `output_<commit>` 目录，包含该版本的完整输出结果和 commit 信息

---

## 常见问题

### Q1: 如何确定 Good 和 Bad 版本？

A: 通过查看 Git 提交历史，找到性能开始退化的提交。可以先手动测试几个版本，确定性能退化的大致范围。

### Q2: 如何编写自定义分析脚本？

A: 分析脚本需要：
1. 读取泳道图或日志文件
2. 提取性能指标
3. 根据判断标准返回退出码（0=GOOD, 1=BAD, 2=UNKNOWN）



### Q3: 如果某个版本编译失败怎么办？

A: 在测试脚本中添加错误处理，编译失败时返回 2（UNKNOWN），让 bisect 跳过该版本。

### Q4: 如何处理性能波动？

A: 可以多次运行测试，取平均值或中位数，减少随机波动的影响。

---

## 性能优化建议

### 常见性能退化原因

1. **调度策略改变**
   - 任务调度顺序调整
   - 优先级修改
   - 负载均衡策略变化

2. **同步机制增加**
   - 额外的同步操作
   - 等待时间增加
   - 锁粒度变化

3. **内存管理优化**
   - 内存分配策略变化
   - 缓存策略调整
   - 内存复用策略变化

4. **代码生成优化**
   - 指令生成策略变化
   - 循环优化策略变化
   - 内联优化策略变化

### 优化建议优先级

1. **高优先级**
   - 调度策略优化
   - 同步机制优化
   - 内存访问优化

2. **中优先级**
   - 任务调度优化
   - 依赖关系优化
   - 代码生成优化

3. **低优先级**
   - 代码重构
   - 文档更新
   - 测试用例优化
