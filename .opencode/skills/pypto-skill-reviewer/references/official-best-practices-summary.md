# 官方最佳实践摘要

审查 skill 时作为权威参考使用的核心指导原则。

**来源：**
1. [Skill 编写最佳实践](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices)
2. [改进 Skill Creator：测试、衡量与迭代](https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills)
3. [用 Skill 扩展 Claude](https://docs.anthropic.com/en/docs/claude-code/skills)
4. [构建 Claude Skill 完整指南](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf)

---

## 1. 核心原则：简洁

> "编写 skill 时最重要的事：保持简洁。"
> — 官方最佳实践

- 每个 token 都在争夺上下文窗口空间
- Claude 本身已具备强大能力——只添加它不知道的内容
- 对每段内容提问："这是否帮助 agent 更好地完成任务？"
- 如果删除某节不会降低输出质量，就删掉它

## 2. Frontmatter 规则

### name
- 仅限小写字母、数字、连字符
- 最长 64 字符
- 推荐动名词形式（如 `processing-pdfs` 而非 `pdf-processor`）
- 必须描述 skill 的功能

### description
- 最长 1024 字符
- 第三人称（"审查 skill..." 而非 "我来审查 skill..."）
- **必须**包含：
  - **做什么**（WHAT）——skill 的功能
  - **何时用**（WHEN）——具体触发场景
- **应该**包含触发关键词以确保可靠激活
- 这是"何时使用"信息的**唯一**存放位置——永远不要放在 body 中

### 标准字段
仅 `name` 和 `description` 是标准 frontmatter 字段。`license`、`version`、`author` 等为非标准字段，应避免使用。

## 3. 渐进披露（三级结构）

```
第 1 级：Frontmatter（元数据）— 始终加载，门控激活
第 2 级：SKILL.md（指令）— 触发时加载，指导执行
第 3 级：references/（资源）— 按需加载，提供深度
```

### 规则
- SKILL.md 应 < 500 行
- 引用文件仅一级深度（无 `references/sub/deep/`）
- Agent 仅在第 2 级明确指示时才读取第 3 级
- 使用显式读取指令：`→ 读取: $SKILL_DIR/references/foo.md`

## 4. 自由度匹配

根据任务脆性匹配指令精确度：

| 脆性 | 精确度 | 格式 | 示例 |
|------|--------|------|------|
| 高（需要精确输出） | 低自由度 | 精确命令、脚本 | `git commit -m "type: message"` |
| 中（结构化但灵活） | 中自由度 | 伪代码、编号步骤 | "1. 读取配置 2. 验证 3. 应用" |
| 低（需要判断力） | 高自由度 | 文字指引 | "使用清晰的描述性命名" |

**常见错误：**
- 对灵活任务过度规定（创意写作用精确模板）
- 对刚性任务欠规定（"推送你的更改"但不给精确参数）

## 5. 工作流与反馈环

### 何时使用工作流
- 多步骤流程
- 有失败模式的操作
- 需要验证的任务

### 模式
```
执行动作 → 验证结果 → 若失败 → 修复 → 重试
```

### 检查清单模式
复杂任务提供编号清单供 agent 追踪：
```markdown
### 第 1 步：收集输入
### 第 2 步：处理
### 第 3 步：验证
### 第 4 步：输出
```

## 6. 内容规范

| ✅ 应该 | ❌ 不应该 |
|---------|----------|
| 使用祈使句/动词不定式 | 使用"你应该"或被动语态 |
| 全文语言一致 | 结构元素中混用语言 |
| 结构化数据用表格 | 用段落列举选项 |
| 首次使用时定义术语 | 使用模糊术语 |
| 编写无时限的指引 | 包含"截至 2024 年..."、"目前..." |

### Body 内容规则
- **禁止** body 中出现"何时使用"节（属于 description）
- **禁止**将 description 重述为"概述"
- **禁止**解释 agent 已知的概念
- **禁止**已发布 skill 中出现占位符/TODO 内容

## 7. 常用 Skill 模式

### 模板模式
输出需要一致结构时（报告、commit、配置）：
```markdown
## 输出模板
<带占位符的模板>
```

### 示例模式
展示如何处理请求时：
```markdown
## 示例
**用户说**："审查 auth skill"
**操作**：读取 skill → 应用清单 → 生成报告
```

### 条件工作流模式
需要分支逻辑时：
```markdown
## 决策树
- 若条件 A → 步骤序列 1
- 若条件 B → 步骤序列 2
- 默认 → 步骤序列 3
```

## 8. 反模式（必须避免）

| 反模式 | 为何有害 | 修复方式 |
|--------|---------|---------|
| 硬编码绝对路径 | 破坏可移植性 | 使用 `$SKILL_DIR` 或相对路径 |
| Windows 风格路径 | 平台特定 | 始终使用正斜杠 |
| 多选项无默认值 | 决策瘫痪 | 高亮推荐默认值 |
| 深度嵌套引用 | 认知负担 | 距 SKILL.md 最多 1 跳 |
| body 中放"何时使用" | 与 description 冗余 | 移到 frontmatter description |
| 冗长解释 | 浪费 token | 简洁指令 |
| 占位符内容 | skill 不完整 | 发布前删除或补完 |

## 9. 两种 Skill 类型

来自《测试、衡量与迭代》：

### 能力提升型（Capability Uplift）
- 教 agent 做它原本做不到的事
- 示例：使用特定 API、操作特定工具
- 测试方式：比较 agent 没有此 skill 时能否完成任务？

### 偏好编码型（Encoded Preference）
- 编码一种 agent 已能做但你希望按特定方式做的偏好
- 示例：特定 commit 消息格式、审查检查清单
- 测试方式：比较 agent 没有此 skill 时是否按你的方式做？

### 评估方法
- 在编写大量文档**之前**先构建评估
- 用真实任务测试，不用合成场景
- 迭代：skill v1 → 评估 → 改进 → skill v2 → 评估 → 比较

## 10. 文件组织

```
skill-name/
├── SKILL.md              # 主指令文件（<500 行）
├── references/           # 详细参考文档
│   ├── api-guide.md      # 按需加载
│   └── examples.md       # 按需加载
├── scripts/              # 自动化脚本
│   └── validate.sh       # 可执行，含错误处理
└── assets/               # 非上下文文件（模板等）
    └── template.txt      # 用于输出，不加载到上下文
```

### 命名约定
- 全部小写+连字符
- 描述性名称（`best-practices-checklist.md` 而非 `ref1.md`）
- SKILL.md 必须记录所有可用文件（资源表）

## 11. 文件引用规范

### ✅ 推荐做法

| 类型 | 格式 | 示例 |
|------|------|------|
| 参考文档 | Markdown 链接 | `[editing.md](editing.md)` |
| 子目录文档 | 相对路径链接 | `[MCP Best Practices](./reference/mcp.md)` |
| 脚本/命令 | 反引号包裹 | `python scripts/helper.py` |
| 多文件 | 表格组织 | 见下方模板 |

### ❌ 不推荐

| 反模式 | 修复 |
|--------|------|
| `@file.md` 引用同目录文件 | 使用 `[file.md](file.md)` |
| `` `file.md` - Description `` | 使用表格或链接列表 |

### 模板示例

```markdown
## Scripts

Run with `--help` first:
```bash
python scripts/helper.py --help
python scripts/processor.py input.txt output.txt
```

## Reference Files

| File | Contents |
|------|----------|
| [references/pattern-a.md](references/pattern-a.md) | Pattern A |
| [references/pattern-b.md](references/pattern-b.md) | Pattern B |
```