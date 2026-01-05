# PyPTO 贡献指南

## 概述

欢迎为 PyPTO 项目做出贡献！本文档指导你如何参与项目开发，包括代码贡献、文档改进、问题报告和功能建议。

在开始贡献前，建议先了解社区协作规则并完成必要的协议（如 CLA）。项目侧的社区协作入口以 `cann-community` 为准。

**贡献方式**:
- [🐛 报告问题](#报告问题)
- [💡 功能建议](#功能建议)
- [🛠️ 代码贡献](#代码贡献)
- [📚 文档贡献](#文档贡献)
- [🧪 测试贡献](#测试贡献)
- [🌍 翻译贡献](#翻译贡献)

---

## 报告问题

### Issue 的基本流程（推荐做法）

如果你在项目中发现问题，建议先提交 Issue 再推进代码修复（避免方向不一致导致 PR 无法合入）：

- **Bug 修复**：创建 `Bug-Report|缺陷反馈` 类 Issue，描述问题与复现方式
- **优化/需求**：创建 `Requirement|需求建议` 类 Issue，说明动机与方案
- **文档纠错**：创建 `Documentation|文档反馈` 类 Issue，指出具体文档与待修正点

分配与跟踪（GitCode 工作流常用方式）：
- 在 Issue 评论框输入 `/assign` 或 `/assign @yourself` 把任务分配给自己，便于持续跟踪处理

### 问题报告模板

创建问题报告时，请使用以下模板：

```markdown
## 问题描述

**简要描述**: [用一两句话描述问题]

**预期行为**: [描述期望的结果]

**实际行为**: [描述实际发生的情况]

## 重现步骤

1. [第一步]
2. [第二步]
3. [第三步]

## 环境信息

- **PyPTO 版本**: [例如: 1.0.0]
- **Python 版本**: [例如: 3.9.7]
- **操作系统**: [例如: Ubuntu 20.04]
- **硬件**: [例如: Ascend 910, CUDA 11.8]
- **安装方式**: [例如: pip, 从源码构建]

## 附加信息

- **错误日志**: [粘贴相关错误日志]
- **截图**: [如果适用，添加截图]
- **最小重现代码**: [提供最小重现代码]
```

### 问题分类

#### 🐛 Bug 报告
- 功能不工作
- 出现错误或异常
- 性能问题
- 兼容性问题

#### ❓ 支持问题
- 使用问题
- 配置问题
- 环境问题

#### 💡 功能请求
- 新功能建议
- 改进建议
- API 增强

### 问题优先级

| 优先级 | 描述 | 响应时间 | 示例 |
|--------|------|----------|------|
| **紧急** | 严重影响使用 | < 24 小时 | 崩溃、数据丢失 |
| **高** | 主要功能受影响 | < 72 小时 | 功能错误、性能问题 |
| **中** | 次要功能问题 | < 1 周 | 不便使用、不一致 |
| **低** | 改进建议 | < 2 周 | 优化建议、新功能 |

---

## 功能建议

### 建议评估标准

提出功能建议时，请考虑：

#### 必要性
- **为什么需要这个功能？**
- **解决什么问题？**
- **现有方案的不足？**

#### 可行性
- **技术上可实现吗？**
- **开发成本如何？**
- **维护成本如何？**

#### 影响范围
- **影响的用户范围？**
- **对现有代码的影响？**
- **向后兼容性？**

### 功能建议模板

```markdown
## 功能概述

**功能名称**: [简洁的功能名称]

**功能描述**: [详细描述功能]

## 动机

**为什么需要**: [解释为什么需要这个功能]

**使用场景**: [描述典型的使用场景]

**当前方案**: [当前如何解决这个问题]

## 设计建议

**API 设计**: [建议的 API 接口]

**实现方案**: [技术实现方案]

**配置选项**: [相关的配置选项]

## 示例代码

```python
# 建议的使用方式
@pypto.jit
def example_function(x):
    # 使用新功能的代码
    pass
```

## 影响评估

**向后兼容**: [是否影响现有代码]

**性能影响**: [对性能的影响]

**学习成本**: [用户学习成本]

## 替代方案

**其他方案**: [考虑过的其他实现方案]

**优缺点**: [各方案的优缺点比较]
```

---

## 代码贡献

### 开发环境搭建

#### 1. 克隆代码库

```bash
# 克隆仓库（以实际仓库地址为准，请替换 <repo_url>）
git clone <repo_url>
cd pypto

# （可选）添加上游远程仓库（以实际仓库地址为准，请替换 <upstream_repo_url>）
git remote add upstream <upstream_repo_url>

# 创建功能分支（请替换 <your-feature-name> 为实际功能名，如 fix-runlog-parser）
git checkout -b feature/<your-feature-name>
```

**重要提示：** 所有占位符（`<repo_url>`、`<upstream_repo_url>`、`<your-feature-name>` 等）必须替换为实际值，不要直接使用占位符。

#### 2. 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -e .  # 开发模式安装

# 安装开发工具
pip install -r requirements-dev.txt
```

#### 3. 构建项目

**重要：** 本项目使用 `build_ci.py` 作为主要构建入口，CMake 为底层构建系统。建议优先使用 `build_ci.py`，CMake 仅作为备选。

**推荐方式（使用 build_ci.py）：**
```bash
# 进入项目根目录
cd /path/to/pypto

# 使用 build_ci.py 构建（推荐）
python3 build_ci.py --build_type Debug --editable --clean

# 运行测试
pytest python/tests -v
```

**备选方式（直接使用 CMake，不推荐）：**
```bash
# 创建构建目录
mkdir build && cd build

# 配置构建
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_WITH_CANN=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_COVERAGE=ON

# 编译
make -j$(nproc)

# 运行测试
make test
```

**详细说明：** 参考[Build 系统文档](../02-core/11-build.md)和[从源码构建与跑测试](../00-getting-started/02-build-and-test.md)

### 代码规范

#### Python 代码规范

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 标准：

```python
# 正确示例
def calculate_weighted_sum(values, weights=None):
    """
    Calculate weighted sum of values.

    Args:
        values: Input tensor
        weights: Optional weight tensor

    Returns:
        Weighted sum tensor
    """
    if weights is None:
        weights = torch.ones_like(values)

    return (values * weights).sum()


# 错误示例
def calc_sum(vals,w=None):  # 不好的命名
    if w is None:w=torch.ones_like(vals)  # 缺少空格
    return(vals*w).sum()  # 缺少空格
```

#### C++ 代码规范

遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)：

```cpp
// framework/include/pypto/tensor.h
#ifndef PYTO_TENSOR_H_
#define PYTO_TENSOR_H_

#include <memory>
#include <vector>

namespace pypto {

// 类定义
class Tensor {
 public:
  // 构造函数
  explicit Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);

  // 方法
  const Shape& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }

 private:
  Shape shape_;
  DataType dtype_;
  std::unique_ptr<void, decltype(&free)> data_;
};

}  // namespace pypto

#endif  // PYTO_TENSOR_H_
```

#### 代码格式化

```bash
# Python 格式化
black .                    # 自动格式化
black --check .           # 检查格式
black --diff .            # 显示差异

# C++ 格式化
clang-format -i framework/**/*.cpp framework/**/*.h  # 格式化
clang-format --dry-run -Werror framework/**/*.cpp    # 检查格式
```

### 提交规范

#### 提交消息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### 提交类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **feat** | 新功能 | `feat(compiler): add new optimization pass` |
| **fix** | 修复bug | `fix(tensor): resolve memory leak in reshape` |
| **docs** | 文档更新 | `docs(api): update tensor operation examples` |
| **style** | 代码风格 | `style: format code with black` |
| **refactor** | 重构 | `refactor(backend): simplify codegen logic` |
| **test** | 测试 | `test(tensor): add unit tests for new operations` |
| **chore** | 构建/工具 | `chore: update cmake configuration` |

#### 示例提交

```bash
# 好的提交消息
feat(compiler): add global memory reuse optimization

- Implement GlobalMemoryReuse pass for tensor memory optimization
- Add configuration options for memory threshold
- Update documentation with usage examples

Closes #123

# 不好的提交消息
fix bug
update code
```

### 代码审查流程

#### 1. 自我审查

提交 Pull Request 前，请进行自我审查：

**功能检查**
- [ ] 代码编译通过
- [ ] 所有测试通过
- [ ] 功能按预期工作
- [ ] 边界情况处理正确

**代码质量**
- [ ] 遵循代码规范
- [ ] 有适当的注释
- [ ] 通过静态分析
- [ ] 没有调试代码

**文档更新**
- [ ] 更新相关文档
- [ ] 添加使用示例
- [ ] 更新 API 文档

#### 2. 创建 Pull Request

```markdown
## PR 标题
[类型] 简洁描述功能

## 描述
详细描述所做的更改和原因

## 相关 Issue
Closes #123
Fixes #456

## 变更类型
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## 检查清单
- [ ] 测试通过
- [ ] 文档更新
- [ ] 向后兼容
- [ ] 代码审查完成

## 附加信息
任何其他相关信息
```

#### 3. 审查响应

收到审查意见后：

```markdown
## 审查响应

### 已解决的问题
- [ ] 问题1: 按照建议修改了...
- [ ] 问题2: 添加了错误处理...

### 保留的决定
- [ ] 问题3: 经过考虑保留当前实现，理由是...

### 新增修改
- [ ] 根据审查意见，添加了单元测试
- [ ] 更新了文档示例

### 验证结果
- [ ] 所有测试通过
- [ ] 性能测试正常
- [ ] 兼容性测试通过
```

### 分支管理

#### 分支命名

```
feature/<feature-name>     # 新功能
fix/<issue-number>         # 修复问题
docs/<document-name>       # 文档更新
refactor/<component>       # 重构
```

#### 合并流程

```bash
# 更新主分支
git checkout main
git pull upstream main

# 变基功能分支
git checkout feature/your-feature
git rebase main

# 解决冲突（如果有）
# 编辑冲突文件
git add <resolved-files>
git rebase --continue

# 推送更新
git push origin feature/your-feature --force-with-lease
```

---

## 文档贡献

### 文档类型

#### 1. 用户文档
- 使用指南
- API 文档
- 教程和示例

#### 2. 开发者文档
- 架构文档
- 设计文档
- 贡献指南

#### 3. 技术文档
- 实现细节
- 算法说明
- 性能分析

### 文档规范

#### Markdown 格式

```markdown
# 文档标题

## 概述

本文档说明...

## 目录

- [第一章](#第一章)
- [第二章](#第二章)

## 第一章

### 小节

内容...

```python
# 代码示例
def example_function():
    return "Hello, World!"
```

### 表格

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |

### 链接

[内部链接](../README.md)
[外部链接](https://example.com)
```

#### 文档结构

```
docs/note/
├── 00-getting-started/00-quick-start.md           # 上手指南（统一入口）
├── 02-core/                       # 核心文档
│   ├── 00-overview.md            # 总览
│   ├── 01-framework.md           # 框架介绍
│   └── ...
├── 03-mechanisms/                # 关键机制文档（包含控制流编译机制）
└── 09-contributing/                      # 元文档
    └── 03-contributing.md       # 贡献指南
```

### PR 模板（提交时建议按此补全信息）

项目侧 PR 模板通常包含以下字段，建议在 PR 描述中完整填写：

- **描述**：改动点、背景、目的与方法
- **关联 Issue**：若用于修复/实现某个 Issue，请写清楚关联关系
- **测试**：做了哪些验证（冒烟/单测/回归/算子泛化等）
- **文档更新**：是否更新了文档，更新了哪些文件
- **类型标签**：Bug 修复 / 新特性 / 性能优化 / 文档更新 / 其他

### 文档审查

#### 审查要点

- [ ] **准确性**: 信息是否正确
- [ ] **完整性**: 是否涵盖所有重要内容
- [ ] **清晰性**: 表达是否清晰易懂
- [ ] **一致性**: 与其他文档风格一致
- [ ] **可维护性**: 是否易于更新

#### 文档测试

```bash
# 检查链接
markdown-link-check docs/note/**/*.md

# 检查拼写
aspell check docs/note/**/*.md

# 验证代码示例
python -m doctest docs/note/**/*.md
```

---

## 测试贡献

### 测试类型

#### 1. 单元测试

```python
# tests/unit/test_tensor.py
import pytest
import numpy as np
import pypto


class TestTensor:
    def test_tensor_creation(self):
        """测试张量创建"""
        data = np.random.randn(10, 20).astype(np.float32)
        tensor = Tensor(data.shape, DataType.DT_FP32)  # 简化为张量创建

        assert tensor.shape == (10, 20)
        assert tensor.dtype == pypto.DT_FP32  # (@docs/api/datatype/DataType.md)

    @pytest.mark.parametrize("shape", [(10,), (5, 10), (2, 3, 4)])
    def test_tensor_reshape(self, shape):
        """测试张量重塑"""
        total_elements = np.prod(shape)
        data = np.arange(total_elements).astype(np.float32)

        tensor = Tensor(data.reshape(-1).shape, DataType.DT_FP32)
        reshaped = tensor.reshape(shape)

        assert reshaped.shape == shape
```

#### 2. 集成测试

```python
# tests/integration/test_compilation.py
import pytest
import pypto


class TestCompilation:
    def test_jit_compilation(self):
        """测试 JIT 编译"""

        @pypto.jit
        def add_function(a, b):
            return a + b

        a = pypto.ones((10, 10))
        b = pypto.ones((10, 10))

        result = add_function(a, b)

        assert result.shape == (10, 10)
        # 验证结果为 2
        assert pypto.allclose(result, pypto.full((10, 10), 2.0))

    def test_complex_computation(self):
        """测试复杂计算"""

        @pypto.jit
        def complex_function(x):
            # 多层计算
            y = x * 2
            y = y.relu()
            y = y @ y.T
            return y.sum()

        x = pypto.zeros((32, 64))  # 简化为零张量
        result = complex_function(x)

        assert result.shape == ()  # 标量
        assert result.dtype == pypto.DT_FP32  # (@docs/api/datatype/DataType.md)
```

#### 3. 性能测试

```python
# tests/performance/test_benchmarks.py
import pytest
import time
import pypto


class TestBenchmarks:
    @pytest.fixture
    def benchmark_data(self):
        return {
            'small': pypto.zeros((128, 128)),
            'medium': pypto.zeros((512, 512)),
            'large': pypto.zeros((1024, 1024))
        }

    def test_matrix_multiplication_performance(self, benchmark_data, benchmark):
        """矩阵乘法性能基准测试"""

        @pypto.jit
        def matmul_optimized(a, b):
            return a @ b

        for size_name, data in benchmark_data.items():
            result = benchmark(matmul_optimized, data, data.T)
            assert result.shape == (data.shape[0], data.shape[0])

    def test_memory_usage(self, benchmark_data):
        """内存使用测试"""
        initial_memory = get_memory_usage()

        @pypto.jit
        def memory_test(data):
            temp = data * 2
            result = temp @ data.T
            return result

        for data in benchmark_data.values():
            result = memory_test(data)

            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory

            # 确保内存使用合理
            assert memory_increase < 1024 * 1024 * 1024  # 1GB 限制
```

### 测试规范

#### 测试命名

```python
# 好的测试命名
def test_tensor_addition_with_broadcasting():
def test_compilation_fails_with_invalid_syntax():
def test_performance_improves_with_optimization():

# 不好的测试命名
def test_func():
def test1():
def test_tensor():
```

#### 测试组织

```python
class TestTensorOperations:
    """张量操作测试"""

    @pytest.fixture
    def setup_tensors(self):
        """设置测试张量"""
        self.a = pypto.zeros((10, 20))
        self.b = pypto.zeros((10, 20))

    def test_elementwise_addition(self, setup_tensors):
        """测试逐元素加法"""
        result = self.a + self.b
        expected = Tensor(
            # 注意：PyPTO Tensor 不提供 to_numpy() 方法
            # 测试时请使用其他验证方式
        )
        assert pypto.allclose(result, expected)

    def test_broadcasting_addition(self):
        """测试广播加法"""
        a = pypto.zeros((3, 1, 5))
        b = pypto.zeros((1, 4, 5))

        result = a + b
        assert result.shape == (3, 4, 5)
```

#### 测试运行

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/test_tensor.py::TestTensor::test_addition

# 运行带覆盖率的测试
pytest --cov=pypto --cov-report=html tests/

# 运行性能测试
pytest tests/performance/ -v --tb=short

# 运行失败重试
pytest --reruns 3 --reruns-delay 1 tests/
```

---

## 翻译贡献

### 支持的语言

- 🇨🇳 简体中文 (zh-CN)
- 🇹🇼 繁体中文 (zh-TW)
- 🇯🇵 日语 (ja)
- 🇰🇷 韩语 (ko)

### 翻译流程

#### 1. 选择文档

优先翻译以下文档：
1. 用户指南和教程
2. API 文档
3. 常见问题 FAQ
4. 错误消息

#### 2. 翻译规范

```markdown
# 英文原文
# PyPTO User Guide

## Overview

This guide helps you get started with PyPTO.

# 中文翻译
# PyPTO 用户指南

## 概述

本指南帮助您开始使用 PyPTO。
```

#### 3. 术语表

| 英文 | 中文 | 说明 |
|------|------|------|
| Tensor | 张量 | 基本数据结构 |
| JIT | JIT | Just-In-Time 编译 |
| Pass | Pass | 优化阶段 |
| Shape | 形状 | 张量维度 |

#### 4. 提交翻译

```bash
# 创建翻译分支
git checkout -b translation/zh-cn-user-guide

# 添加翻译文件
cp docs/note/02-core/00-overview.md docs/note/02-core/00-overview.zh-CN.md

# 提交翻译
git add docs/note/02-core/00-overview.zh-CN.md
git commit -m "docs(translation): add Chinese translation for overview"
```

---

## 行为准则

### 我们的承诺

PyPTO 项目致力于为所有人提供一个无骚扰的环境。我们承诺，无论年龄、体型、残疾、民族、性别认同和表达、经验水平、国籍、外貌、种族、宗教或性认同和取向如何，都不会受到歧视。

### 标准

**鼓励的行为：**
- 使用友好的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有益的事
- 对其他社区成员表示同理心

**不允许的行为：**
- 使用性化语言或图像
- 进行人身攻击
- 发表垃圾信息或恶意评论
- 公开或私下骚扰
- 未经明确许可发布他人私人信息
- 其他可能被合理认为不适当的行为

### 责任

项目维护者有责任澄清可接受行为的标准，并对任何不可接受行为采取适当和公平的纠正措施。

### 适用范围

此行为准则适用于所有项目空间，也适用于个人在代表项目时的公共空间。

### 执行

如遇到违规行为，请通过以下方式报告：
- 通过项目托管平台的 Issue/PR @ 项目维护者，并在描述中说明涉及的行为与证据

所有投诉都将被审查和调查，并将做出适当的回应。

---

## 获得帮助

### 联系方式与协作渠道（以项目托管平台为准）

- **问题反馈**：通过项目托管平台的 Issues 提交
- **功能建议/讨论**：通过项目托管平台的讨论区参与交流
- **技术支持**：优先查阅 `docs/note/` 与 `docs/` 文档，并在必要时提交 Issue

### 资源

- [开发指南](../08-best-practices/00-development-guide.md) - 详细的开发指南
- [API 文档](../02-core/02-api-reference.md) - 完整的 API 参考
- [完整调试指南](../05-debugging/00-complete-guide.md) - 系统化调试方法

---

## 致谢

感谢所有为 PyPTO 项目做出贡献的开发者！

特别感谢我们的金牌贡献者：
- 🏆 **核心贡献者**: 为项目架构和实现做出重大贡献
- 🥈 **活跃贡献者**: 持续为项目改进做出贡献
- 🥉 **新星贡献者**: 新加入社区并做出优质贡献

---

*本贡献指南将持续更新。如有问题或建议，请提交 Issue 或 Pull Request。*

