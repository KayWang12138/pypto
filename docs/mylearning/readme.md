# PyPTO 学习记录与文档归档

本目录包含 PyPTO 算子开发的学习记录、配置指南和问题解决方案。

---

## 📚 文档目录

### 环境配置与维护

- **[environment_sync_and_setup_record.md](./environment_sync_and_setup_record.md)**  
  **环境同步与配置完整记录** (2026-03-03)  
  记录了仓库同步、PyPTO 环境配置、pto-isa 源码编译和环境验证的完整流程。包含详细的步骤说明、问题诊断和解决方案。  
  - ✅ Git 仓库同步与代码合并
  - ✅ PyPTO 0.1.1 编译安装
  - ✅ pto-isa 源码环境配置
  - ✅ 环境验证与测试
  - ✅ 常见问题与解决方案

---

### 算子开发指南

#### IFA (Inference Attention) 算子

- **[ifa_instrucction.md](./ifa_instrucction.md)**  
  IFA 算子开发指导文档

#### PFA (Prefill Attention) 算子

- **[pfa_instrucction.md](./pfa_instrucction.md)**  
  PFA 算子开发详细指导

- **[pfa_graph.md](./pfa_graph.md)**  
  PFA 计算图分析和可视化

- **[pfa_detailed_walkthrough.md](./pfa_detailed_walkthrough.md)** ⭐ **推荐**  
  **PFA 详细迭代过程与寻址计算解读** (2026-03-04)  
  使用具体数值例子详细说明大矩阵拆分成小矩阵的计算过程，包括：
  - 📊 Reshape 操作详解与映射关系
  - 📍 五层循环迭代的寻址计算
  - 🔢 Query/KV 定位与提取的具体步骤
  - 🔄 完整的矩阵运算数据流
  - 🎯 多次迭代的增量更新机制
  - 💡 常见问题解答（FAQ）
  
  **适合**: 已经了解整体流程，但想深入理解寻址和迭代细节的开发者

- **[pfa_config_guide.md](./pfa_config_guide.md)**  
  PFA 配置参数指南

- **[pfa_optimization_guide.md](./pfa_optimization_guide.md)**  
  PFA 性能优化详细指南

- **[pfa_optimization_record.md](./pfa_optimization_record.md)**  
  PFA 优化实践记录

---

### 工作流程

- **[optimization_workflow.md](./optimization_workflow.md)**  
  通用优化工作流程指南

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 安装 PyPTO
python3 -m pip install . --verbose

# 2. 设置环境变量（如果使用 pto-isa 源码）
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/mce/pto-isa
export TILE_FWK_DEVICE_ID=0

# 3. 验证环境
python3 examples/00_hello_world/hello_world.py --run_mode=npu
```

### 参考最新环境配置文档

详细的环境配置步骤请参考：**[environment_sync_and_setup_record.md](./environment_sync_and_setup_record.md)**

---

## 📖 文档维护说明

### 文档命名规范

- **功能文档**: `{功能名称}_guide.md` - 如 `pfa_config_guide.md`
- **记录文档**: `{功能名称}_record.md` - 如 `pfa_optimization_record.md`
- **指令文档**: `{功能名称}_instrucction.md` - 如 `ifa_instrucction.md`
- **分析文档**: `{功能名称}_graph.md` - 如 `pfa_graph.md`

### 更新记录

| 日期 | 文档 | 变更内容 |
|------|------|----------|
| 2026-03-04 | pfa_detailed_walkthrough.md | 新增：PFA 详细迭代过程与寻址计算解读，包含具体数值例子 |
| 2026-03-03 | environment_sync_and_setup_record.md | 新增：完整的环境同步与配置记录 |
| 2026-03-03 | readme.md | 更新：添加文档目录和快速开始指南 |

---

**最后更新**: 2026-03-04  
**维护人**: PyPTO 开发团队
