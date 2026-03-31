# 常见问题类型

本文档列出 references 文件中常见的问题类型及其识别方法。

---

## 一、P0 必须修复

### 1.1 代码错误

**定义**：语法错误、拼写错误、函数名错误

**示例**：
```cpp
// 错误：EXPECT_EQ_EQ 重复了 EQ
EXPECT_EQ_EQ(pass.PostCheck(function), SUCCESS);

// 正确
EXPECT_EQ(pass.PostCheck(function), SUCCESS);
```

**识别方法**：
- 代码审查
- 如可执行，实际运行验证

---

### 1.2 路径错误

**定义**：文件路径不存在、相对路径指向错误位置

**示例**：
```markdown
# 错误：同级目录的文件用 ../ 会指向上一级
[common_errors.md](../common_errors.md)

# 正确
[common_errors.md](./common_errors.md)
```

**识别方法**：
- `ls <path>` 验证路径存在
- 分析相对路径的基准目录

---

### 1.3 API 不存在

**定义**：引用了 docs 中不存在的 API

**示例**：
```markdown
# 错误：docs 中没有 pypto.compile API
- `pypto.compile` — 编译 API

# 正确：PyPTO 编译入口是 @pypto.frontend.jit
- `pypto.frontend.jit` — JIT 编译装饰器
```

**识别方法**：
- `grep "pypto\.<api>" docs/` 搜索 API 定义
- 检查 docs/api/ 下是否有对应文档

---

### 1.4 与 docs 矛盾

**定义**：参数说明、选项值等与 docs 直接矛盾

**示例**：
```markdown
# references 说
"禁止 --type=all"

# docs 说
--type | 可选：deps, cann, third_party, all
```

**识别方法**：
- 对比 docs 中相同主题的描述
- 检查 docs 中的参数说明、约束说明

---

## 二、P1 建议修复

### 2.1 概念歧义

**定义**：术语使用可能造成误解

**示例**：
```markdown
# 歧义：示例说 pypto.reshape 不存在，但实际存在
**示例**：尝试调用 `pypto.reshape` 但该 API 不存在

# 建议：使用虚构的 API 名称
**示例**：尝试调用 `pypto.nonexistent_op` 但该 API 不存在
```

---

### 2.2 术语不一致

**定义**：与 docs 术语表不一致

**示例**：
```markdown
# references 用词
"尾轴 32B 对齐"

# docs 用词
"外轴切分大小满足 32B 对齐" / "尾轴 32B 对齐"

# 建议：确认语境后统一表述
```

---

### 2.3 正则/格式错误

**定义**：正则表达式、格式规范有语法问题

**示例**：
```markdown
# 错误：正则缺少结束符
^(feat|fix|...)(.*): [A-Z].{10,200}

# 正确
^(feat|fix|...)(.*): [A-Z].{10,200}$
```

---

## 三、P2 可选修复

### 3.1 描述模糊

**定义**：简化写法可能遗漏重要信息

**示例**：
```markdown
# 模糊
- dtype: INT8-64

# 更清晰（或指向 docs）
- dtype: INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64
- dtype: 详见 docs/api/others/pypto-from_torch.md
```

---

### 3.2 引用缺失

**定义**：引用了不存在的资源文件

**示例**：
```markdown
# 问题：引用了不存在的 Excel 文件
完整规则定义以官方 Excel 为准（`rule_ch.xlsx`）

# 处理：添加文件，或提供获取方式，或删除引用
```

---

## 四、检查原则：先确认执行上下文

### 4.1 核心原则

> **references 中的路径、命令、环境变量，其"基准"不一定是项目根目录**

在标记为"路径错误"或"命令错误"之前，必须先确认**执行上下文**：

| 上下文要素 | 说明 | 影响范围 |
|------------|------|----------|
| 工作目录 | 命令执行时的 pwd | 相对路径解析 |
| 环境变量 | 预设的变量如 `$SKILL_DIR` | 路径、配置引用 |
| 前置条件 | 如"已 cd 到某目录" | 后续命令的基准 |

### 4.2 判断方法

**步骤 1：检查 SKILL.md 是否说明了执行上下文**

```markdown
# 示例：明确说明工作目录
## 约定
执行以下命令前，请先 cd 到 <skill_dir>

# 示例：定义环境变量
- `$SKILL_DIR`：指向当前 skill 根目录
```

**步骤 2：检查同类 skill 的惯例**

```bash
# 检查其他 skill 是否有相同写法
grep -r "scripts/" .agents/skills/*/SKILL.md | wc -l
# 如果多个 skill 都这样写，说明是约定，不是错误
```

**步骤 3：验证资源是否存在（在正确的上下文中）**

```bash
# 错误做法：从项目根目录验证
ls scripts/xxx.py  # 不存在 → 误判为错误

# 正确做法：从 skill 目录验证
ls .agents/skills/<skill>/scripts/xxx.py  # 存在 → 不是错误
```

## 五、排除项示例

以下情况**不需要标记为问题**：

| 情况 | 示例 | 原因 |
|------|------|------|
| 合理简化 | dtype: FP16/BF16/FP32/INT8-64/BOOL | 速查表 + 指向 docs |
| 更严格规范 | commit message 10-200 字符限制 | skill 可定义更严格要求 |
| 内部知识 | 常见错误排查经验 | docs 不一定包含 |
| 上下文相关路径 | scripts/xxx.py | 已确认执行上下文 |
| 内部模板/规则 | 评审规则、报告模板 | 无 docs 对应