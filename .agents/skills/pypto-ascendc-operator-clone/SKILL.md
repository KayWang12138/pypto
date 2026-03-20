---
name: pypto-ascendc-operator-clone
description: 根据Ascend C算子开发对应的PyPTO算子，完成输出验证和性能对比。包含需求理解、算子开发、输出验证、性能测试的完整流程。
tag: [PyPTO, Ascend C, 算子开发, 输出验证, 性能测试]
---

# PyPTO算子克隆开发流程（基于Ascend C算子）

# 重要：类似**...**的提示内容是重点，要着重理解与记忆！

**宗旨：坚持解决而非逃避问题，争取一站式完成以下任务**

## 工作流程

```
需求理解 → Golden开发与验证 → 算子开发与验证 → 输出验证 → 性能测试 → 清理总结
```

## 核心原则 ⭐⭐⭐

1. **参考AscendC代码**：不确定的地方，直接看AscendC代码怎么实现的
2. **P0+P1必须实现**：第一个版本必须包含P0和P1功能
3. **必须实际运行**：输出验证和性能测试必须在NPU上实际运行，绝不能跳过或编造
4. **交付可用算子**：最终交付的算子必须能正常运行、精度正确、功能完备
5. **禁止逃避问题**：遇到技术困难必须解决，不能退而求其次做简化版本！基本功能和重要的拓展功能必须支持！

## 输入要求

- Ascend C算子README路径
- NPU设备ID（通过环境变量 `TILE_FWK_DEVICE_ID` 设置，如 `export TILE_FWK_DEVICE_ID=0`）

---

## 阶段一：需求理解（核心）

**目标**：提炼AscendC算子的实际功能需求，确定PyPTO实现范围

### 核心步骤

1. **阅读Ascend C算子README** - 提取基本信息
2. **阅读Ascend C算子实现代码** ⭐ - 查看kernel/**/*.cpp，理解实际计算逻辑
3. **提炼实际功能需求** - 创建功能对比表，确定实现优先级（P0-P3）
4. **生成spec.md** - 需求规格文档

### 重点提醒

- **复杂形状分析**：如果README描述不清晰，必须查看AscendC代码
- **P0+P1必须实现**：第一个版本必须实现所有P0和P1功能

### 关键输出

- `custom/{算子名}/spec.md` - 需求规格
- `custom/{算子名}/REQUIREMENT_ANALYSIS.md` - 需求分析报告（可选）

**详细指南**：参见 [guides/requirement_analysis_guide.md](guides/requirement_analysis_guide.md)

---

## 阶段二：Golden开发与验证

**重要：先开发Golden，再开发PyPTO kernel！**

### 核心原则

**Golden的功能范围 = 阶段一评估后确定的PyPTO实现范围**

- 所有P0/P1功能，Golden都必须实现
- 参考AscendC代码确定计算细节

### 关键步骤

1. **确定功能范围** - 回顾阶段一的优先级分析
2. **参考AscendC实现** - 确保计算逻辑与AscendC一致
3. **实现完整功能** - 支持所有P0/P1功能的参数和条件分支
4. **验证AscendC一致性** - 在NPU上实际运行对比
5. **运行golden确保能够运行通过** - 必须验证正确性后才能够进入下一步

### 关键输出

- `custom/{算子名}/{算子名}_golden.py` - Golden参考实现

**详细指南**：参见 [guides/golden_development_guide.md](guides/golden_development_guide.md)

---

## 阶段三：PyPTO算子开发与验证
**注意，运行算子不需要进行PyPTO的重新编译安装，修改了框架代码才需要。所以只要当前环境下已经安装了可用的pypto版本，运行算子前不需要进行再一次的编译安装**

**重要！PyPTO算子开发完毕后，运行它并确保能够运行通过！必须验证正确性后才能够进入下一步，如果有报错，坚持解决问题直到能够跑通，不要逃避问题也不要通过简化代码方式规避**

### 开发规范

- 开发完成后必须运行验证算子能够正常跑通，若碰见问题要去修复，不能逃避或简化代码。
- 合理设置Tile大小
- **P0+P1功能必须全部实现，如果过程中出现问题，尝试解决而不是回退功能！**

### 关键输出

- `custom/{算子名}/{算子名}.py` - PyPTO算子实现

**详细指南**：参见 [guides/pypto_development_guide.md](guides/pypto_development_guide.md)

**API映射参考**：参见 [references/pypto_ascendc_api_mapping.md](references/pypto_ascendc_api_mapping.md)

---

## 阶段四：输出验证 ⭐⭐⭐

**必须实际在NPU上运行算子进行验证！绝不能跳过或编造！**
**在进行到这一阶段前，先确保PyPTO算子已经实现了基础功能以及P0+P1级别的拓展功能，且能够正常运行通过！有问题的话就继续解决问题，直到符合要求，不能逃避或简化代码。**
**该部分将验证分为三个Level，它们都是要完成的验证工作，都不可以省略，都必须完成！**
**如果有报错，坚持解决问题直到能够跑通，不要逃避问题也不要通过简化代码方式规避**

### 验证层次

```
Level 1: PyPTO vs Golden（验证kernel正确性）
Level 2: Golden vs AscendC（验证功能一致性）
Level 3: PyPTO vs AscendC（最终验证）
```

### 验证步骤

1. 检查并搭建环境，确保可以在NPU上正常运行pypto与ascendC算子
3. 在NPU上运行验证脚本
4. 记录实际误差数据

### 注意事项

1. **功能一致性** - 验证的功能必须一致
2. **输出规格差异** - 可只验证核心输出
3. **精度容差** - BF16/FP16: atol=0.01, rtol=0.01; FP32: atol=0.001, rtol=0.001

### 关键输出

- `custom/{算子名}/OUTPUT_VERIFICATION_REPORT.md` - 输出验证报告（包含实际运行数据）

**详细指南**：参见 [guides/output_verification_guide.md](guides/output_verification_guide.md)

---

## 阶段五：性能测试 ⭐⭐⭐

**必须实际在NPU上运行采集数据！绝不能跳过或编造！**
**必须完成PyPTO算子与AscendC算子各自的性能采集！不能缺少！最终的目标就是对比两者性能**
**PyPTO算子与AscendC算子各自的性能采集方式下方都有介绍，不要自己瞎测**
**如果有报错，坚持解决问题直到能够跑通，不要逃避问题也不要通过简化代码方式规避**

### PyPTO性能采集

```python
# 首先一定要在kernel中开启debug模式
kernel.debug_options = {"runtime_debug_mode": 1}
# 开启后运行算子
# 从 bubble_analysis.log 获取 Core Total Work Time
```

### AscendC性能采集（msprof）

```bash
# 执行采集
msprof --output=/tmp/msprof_output --ai-core=on --task-time=l2 python benchmark.py

# 关键文件
op_statistic_*.csv  # Avg Time(us) = 纯kernel时间
```

### 关键输出

- `custom/{算子名}/KERNEL_PERFORMANCE_REPORT.md` - 性能报告（包含实际运行数据）

**详细指南**：参见 [guides/performance_testing_guide.md](guides/performance_testing_guide.md)

---

## 阶段六：清理总结

### 最终交付

```
custom/{算子名}/
├── {算子名}.py                    # PyPTO算子实现
├── {算子名}_golden.py             # Golden参考实现
├── spec.md                        # 需求规格
├── README.md                      # 算子文档
├── OUTPUT_VERIFICATION_REPORT.md  # 输出验证报告
└── KERNEL_PERFORMANCE_REPORT.md   # 性能报告
```

### README.md必须包含

1. 功能描述和数学公式
2. 实现状态表（哪些功能已实现/暂不支持）
3. 已知限制和原因说明
4. 使用示例
5. 验证结果摘要

---

## 检查清单

### 阶段一：需求理解
- [ ] 阅读README和实现代码，创建功能对比表
- [ ] 明确可选参数的实际意义和实现优先级
- [ ] 生成spec.md，标注P0/P1功能

### 阶段二：Golden开发
- [ ] Golden实现完成（覆盖所有P0/P1功能）
- [ ] 参考AscendC代码确保计算逻辑一致

### 阶段三：PyPTO算子开发
- [ ] PyPTO kernel实现完成（包含P0/P1功能）
- [ ] 编译通过，无错误

### 阶段四：输出验证
- [ ] 在NPU上实际运行PyPTO vs Golden验证
- [ ] 在NPU上实际运行PyPTO vs AscendC验证
- [ ] 记录实际误差数据

### 阶段五：性能测试
- [ ] 在NPU上实际采集PyPTO性能数据
- [ ] 在NPU上实际采集AscendC性能数据
- [ ] 生成性能对比报告

### 阶段六：清理总结
- [ ] 文档完整，包含实现状态说明
- [ ] 所有功能正常运行

---

## 参考资料

**API映射参考**：
- [references/pypto_ascendc_api_mapping.md](references/pypto_ascendc_api_mapping.md) - PyPTO与AscendC API映射

**本 Skill 文件**：
- `guides/requirement_analysis_guide.md` - 需求理解详细指南
- `guides/golden_development_guide.md` - Golden开发详细指南
- `guides/pypto_development_guide.md` - PyPTO开发详细指南
- `guides/output_verification_guide.md` - 输出验证详细指南
- `guides/performance_testing_guide.md` - 性能测试详细指南
- `templates/` - 各类模板文件