---
name: pypto-perf-regression-bisect-finder
description: PyPTO 性能退化通用二分查找技能。通过二分法查找导致性能退化的 PR，支持用户自定义性能现象、分析方法和判断标准。
---

# PyPTO 性能退化通用二分查找

通过二分法查找导致性能退化的 PR，支持用户自定义性能现象、分析方法和判断标准。

---

## 所需输入

### 测试案例路径

提供测试案例路径：
- 例如：models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py
- 例如：examples/02_intermediate/operators/softmax/softmax.py

### 性能分析方法

说明如何从泳道图或日志中提取性能指标：
- 例如："计算最后一个 AIC 完成时间和第一个 AIV 开始时间的差值"
- 例如："计算第一个 AIC 完成时间和第一个 AIV 开始时间的差值"
- 例如："计算测试用例的总执行时间"

参考 [references/analysis-methods.md](references/analysis-methods.md) 了解常用分析方法。

### 性能判断标准

定义性能判断标准：
- Good 版本的指标范围：[例如：间隙 < 8.0 微秒]
- Bad 版本的指标范围：[例如：间隙 >= 10.0 微秒]

### 二分起始版本

提供性能正常的起始 commit hash（基准版本）。

### 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [references/analysis-methods.md](references/analysis-methods.md) | 常用性能分析方法的详细说明和示例代码 | 阶段 2 读取 |
| [scripts/analyze_performance.py](scripts/analyze_performance.py) | 性能分析脚本，根据用户定义的方法提取性能指标 | 阶段 2 执行 |
| [scripts/bisect_test.sh](scripts/bisect_test.sh) | git bisect 测试脚本，编译、运行测试和分析性能 | 阶段 4 执行 |

---

## 工作流程

### 阶段 1：信息收集

与用户交互，收集以下信息：

1. **测试案例路径**：用户提供测试案例路径
2. **性能分析方法**：用户说明如何从泳道图提取性能指标
3. **性能判断标准**：用户定义 Good/Bad 版本的阈值范围
4. **二分起始版本**：用户提供性能正常的起始 commit hash

### 阶段 2：获取当前版本性能指标

检查测试用例 runtime_debug_mode 是否打开，运行测试用例，得到泳道图。

```bash
# 检查测试用例 runtime_debug_mode 配置
# 运行测试用例生成泳道图
python3 $TEST_CASE
```

运行性能分析脚本，从泳道图或日志中提取当前版本的性能指标。

```bash
python3 scripts/analyze_performance.py \
    --output-dir output/ \
    --analysis-method "$ANALYSIS_METHOD" \
    --good-threshold $GOOD_THRESHOLD \
    --bad-threshold $BAD_THRESHOLD
```

### 阶段 3：验证二分起始版本

回退到二分起始版本，编译安装、运行测试用例，运行性能分析脚本，确认二分起始值的性能指标优于当前性能指标。

```bash
# 切换到二分起始版本
git checkout $GOOD_COMMIT

# 重新编译安装
rm -rf build_out
python3 build_ci.py -f=python3
pip install build_out/pypto*.whl --force-reinstall --no-deps

# 运行测试用例生成泳道图
rm -rf output/*
python3 $TEST_CASE

# 分析性能指标
python3 scripts/analyze_performance.py \
    --output-dir output/ \
    --analysis-method "$ANALYSIS_METHOD" \
    --good-threshold $GOOD_THRESHOLD \
    --bad-threshold $BAD_THRESHOLD
```

### 阶段 4：二分查找

使用 git bisect 自动查找导致性能退化的 commit。

```bash
# 切换回当前版本（HEAD）
git checkout HEAD

# 初始化 bisect
git bisect start $GOOD_COMMIT HEAD

# 运行 bisect
git bisect run ./scripts/bisect_test.sh
```

### 阶段 5：结果分析

输出问题 commit 信息和总结。

```bash
# 获取 bisect 结果
problem_commit=$(git bisect log | grep "first" | awk '{print $4}')

echo "问题提交: $problem_commit"
echo "提交信息: $(git log --format="%s" -1 $problem_commit)"
echo "提交作者: $(git log --format="%an <%ae>" -1 $problem_commit)"
echo "提交时间: $(git log --format="%ai" -1 $problem_commit)"

# 查看问题提交的代码变更
git show $problem_commit
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

## 注意事项

1. **环境一致性**：确保测试环境在不同版本间保持一致
2. **分析方法准确性**：确保分析方法能够准确反映性能现象
3. **编译失败处理**：编译失败或测试失败时，返回 UNKNOWN 让 bisect 跳过该版本

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
