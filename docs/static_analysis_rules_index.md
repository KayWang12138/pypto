# 静态分析规则文档索引

本目录包含 GitHub 和 GitCode 两个平台的静态分析规则文档。

## 📚 文档列表

### 1. [GitHub vs GitCode 静态告警规则深度对比分析](./static_analysis_comparison.md) ⭐ 推荐

**文件**: `static_analysis_comparison.md` (20 KB, 649 行) **[已更新]**

**内容概要**:
- **执行摘要**: 关键发现和规则数量对比
- **C++ 安全规则详细对比**: 
  - 内存安全规则 (4 条)
  - 除零和算术安全 (1 条)
  - 危险函数和命令注入 (3 条)
  - 缓冲区和字符串安全 (2 条)
- **Python 安全规则对比**:
  - 数据安全处理 (DSP)
  - 文件安全 (FIO)
  - 错误处理 (ERR)
- **综合分析**: 规则覆盖率、关键差距矩阵、风险评估
- **实施建议**: 短期、中期、长期建议，包含具体配置示例
- **规则集成优先级路线图**: Phase 1-3 详细计划

**重点发现**:
- 🔴 **GitHub cpplint 缺失 70% 的安全规则**
- 🔴 **内存安全是最大差距** (完全缺失)
- 🔴 **除零检查缺失** (常见运行时错误)
- 🔴 **危险函数检查缺失** (alloca, system 等)
- 🔴 **命令注入风险** (外部数据未校验)

**适用场景**: 
- 了解两个平台的安全规则差距
- 制定安全规则集成策略
- 确定规则集成优先级
- 获取具体的工具配置示例

---

### 2. [GitHub 静态检查工具规则详细列表](./github_rules_detailed.md)

**文件**: `github_rules_detailed.md` (36 KB, 1539 行)

**内容概要**:
- **Ruff 规则**: 295 条 Python 代码检查规则详细列表
  - PL (Pylint): 114 条
  - I (isort): 13 条
  - E (pycodestyle errors): 69 条
  - W (pycodestyle warnings): 7 条
  - F (pyflakes): 92 条
- **cpplint 规则**: 66 条 C++ 代码风格检查规则
  - build: 18 条
  - legal: 1 条
  - readability: 15 条
  - runtime: 16 条
  - whitespace: 19 条
- **clang-format 配置**: Google 风格的 C++ 格式化配置
- **Pyright 配置**: Python 类型检查配置
- **自定义检查规则**: 版权头检查、英文注释检查等

**适用场景**:
- 查询具体的规则代码和说明
- 了解每个工具的详细配置
- 开发自定义规则时参考

---

### 3. [GitCode CI 静态告警规则](./rule_ch.xlsx)

**文件**: `rule_ch.xlsx` (Excel 文件)

**内容概要**:
- **C++ 规则**: 34 条 (致命 1, 严重 23, 一般 9, 提示 1)
- **C 规则**: 21 条 (严重 13, 一般 7, 提示 1)
- **Python 规则**: 111 条 (致命 2, 严重 40, 一般 69)

**规则字段** (15 列):
1. 规则名称
2. 问题级别
3. 支持的工具版本
4. 语言
5. 标签
6. 适用范围
7-8. 延迟时间
9. 选项
10-11. 正确/错误示例
12. 修改建议
13. 样例
14. 链接
15. CWE 信息

**适用场景**:
- 查看 GitCode 完整规则列表
- 了解规则的详细示例和修改建议
- 追溯规则到 CWE 安全分类

---

## 📊 规则统计对比

### 规则数量

| 平台 | 工具 | 规则数 | 类型 |
|------|------|--------|------|
| **GitHub** | Ruff | 295 条 | Python 代码质量 |
| **GitHub** | cpplint | 66 条 | C++ 代码风格 |
| **GitHub** | clang-format | Google 风格 | C++ 格式化 |
| **GitHub** | Pyright | basic 模式 | Python 类型检查 |
| **GitHub** | 自定义检查 | 6 条 | 版权、英文、通用 |
| **GitCode** | CI 规则 | 166 条 | C/C++/Python 安全和规范 |

### 安全规则覆盖率

| 安全检查项 | GitHub | GitCode | 差距 |
|-----------|--------|---------|------|
| **C++ 内存安全** | ❌ 0/4 | ✓ 4/4 | 🔴 100% |
| **C++ 除零检查** | ❌ 0/1 | ✓ 1/1 | 🔴 100% |
| **C++ 命令注入** | ❌ 0/3 | ✓ 3/3 | 🔴 100% |
| **C++ 缓冲区安全** | ⚠️ 1/2 | ✓ 2/2 | 🟡 50% |
| **Python 安全随机** | ❌ 0/1 | ✓ 1/1 | 🔴 100% |
| **Python SSL/TLS** | ❌ 0/1 | ✓ 1/1 | 🔴 100% |
| **Python 文件安全** | ❌ 0/1 | ✓ 1/1 | 🔴 100% |
| **Python 异常处理** | ⚠️ 1/2 | ✓ 2/2 | 🟡 50% |

---

## 🎯 快速导航

### 按需求查找

**我想了解...**

- ❓ **两个平台的安全差距？** → [深度对比分析 - 执行摘要](./static_analysis_comparison.md#-执行摘要)
- ❓ **C++ 安全规则缺失了什么？** → [C++ 安全规则详细对比](./static_analysis_comparison.md#-c-安全规则详细对比)
- ❓ **Python 安全规则缺失了什么？** → [Python 安全规则对比](./static_analysis_comparison.md#-python-安全规则对比)
- ❓ **如何快速集成安全检查？** → [实施建议](./static_analysis_comparison.md#-实施建议)
- ❓ **GitHub 具体用了哪些规则？** → [GitHub 规则详细列表](./github_rules_detailed.md)
- ❓ **GitCode 有哪些规则？** → [rule_ch.xlsx](./rule_ch.xlsx)

### 按工具查找

**Python 工具**:
- [Ruff 规则详情](./github_rules_detailed.md#ruff-规则-python) (295 条)
- [Pyright 配置](./github_rules_detailed.md#pyright-配置-python)
- [Bandit 配置示例](./static_analysis_comparison.md#2-集成-python-安全检查工具) (推荐集成)

**C/C++ 工具**:
- [cpplint 规则详情](./github_rules_detailed.md#cpplint-规则-cc) (66 条)
- [clang-format 配置](./github_rules_detailed.md#clang-format-配置-cc)
- [cppcheck 配置示例](./static_analysis_comparison.md#1-集成-c-静态分析工具) (推荐集成)
- [clang-tidy 配置](./static_analysis_comparison.md#1-启用-clang-tidy-目前被注释) (推荐启用)

**自定义检查**:
- [版权头检查](./github_rules_detailed.md#1-check-headers-版权头检查)
- [英文注释检查](./github_rules_detailed.md#2-check-english-only-英文注释检查)

### 按优先级查找

**🔴 高优先级 (P0 - 立即修复)**:
1. [C++ 内存安全](./static_analysis_comparison.md#1-内存安全规则) - 4 条规则缺失
2. [C++ 除零检查](./static_analysis_comparison.md#2-除零和算术安全) - 1 条规则缺失
3. [C++ 命令注入](./static_analysis_comparison.md#3-危险函数和命令注入) - 3 条规则缺失
4. [Python 安全随机数](./static_analysis_comparison.md#1-数据安全处理-dsp) - 1 条规则缺失
5. [Python SSL/TLS](./static_analysis_comparison.md#1-数据安全处理-dsp) - 1 条规则缺失
6. [Python 文件解压](./static_analysis_comparison.md#2-文件安全-fio) - 1 条规则缺失

**🟡 中优先级 (P1 - 近期修复)**:
- [C++ 缓冲区安全](./static_analysis_comparison.md#4-缓冲区和字符串安全) - 部分覆盖
- [Python 异常处理](./static_analysis_comparison.md#3-错误处理-err) - 部分覆盖

---

## 🚀 快速开始

### 立即集成安全检查 (5 分钟)

#### 1. 集成 cppcheck (C++ 安全检查)

```bash
# 安装 cppcheck
pip install cppcheck

# 添加到 .pre-commit-config.yaml
cat >> .pre-commit-config.yaml << 'YAML'
- repo: https://github.com/pocc/pre-commit-hooks
  rev: v1.3.5
  hooks:
    - id: cppcheck
      args:
        - --enable=warning,style,performance,portability
        - --error-exitcode=1
YAML
```

#### 2. 集成 bandit (Python 安全检查)

```bash
# 安装 bandit
pip install bandit

# 添加到 .pre-commit-config.yaml
cat >> .pre-commit-config.yaml << 'YAML'
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
    - id: bandit
      args: ['-c', 'pyproject.toml']
YAML

# 添加配置到 pyproject.toml
cat >> pyproject.toml << 'TOML'
[tool.bandit]
exclude_dirs = ['tests', 'build', 'dist']
tests = ['B311', 'B501', 'B502', 'B503', 'B504', 'B601', 'B602']
TOML
```

#### 3. 运行检查

```bash
pre-commit run --all-files
```

---

## 📝 相关文件

- **GitHub 配置文件**:
  - `github/pypto/.pre-commit-config.yaml` - Pre-commit 配置
  - `github/pypto/pyproject.toml` - Python 项目配置
  - `github/pypto/.clang-format` - C++ 格式化配置

- **GitCode 规则文件**:
  - `pypto_open/docs/rule_ch.xlsx` - CI 静态告警规则 (Excel)

---

## 🔗 外部参考

**工具文档**:
- [Ruff 官方文档](https://docs.astral.sh/ruff/)
- [cpplint GitHub](https://github.com/cpplint/cpplint)
- [cppcheck 官方文档](http://cppcheck.net/)
- [clang-tidy 文档](https://clang.llvm.org/extra/clang-tidy/)
- [bandit 文档](https://bandit.readthedocs.io/)
- [Pyright 文档](https://microsoft.github.io/pyright/)

**安全标准**:
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CERT C++ Coding Standard](https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=88046682)
- [SEI CERT C Coding Standard](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard)

**代码风格指南**:
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [PEP 8 - Python 代码风格指南](https://peps.python.org/pep-0008/)

---

**生成时间**: 2026-02-04  
**文档版本**: 2.0 (深度分析版)  
**维护者**: PyPTO Team
