# 问题记录模板

```markdown
### 问题 N：<简短标题>

- **文件**：`<skill>/references/<file>:<line>`
- **类型**：P0 / P1 / P2
- **问题**：
  <具体问题描述>

- **docs 对比**：
  <docs 中的正确内容，引用文件和行号>

- **修复建议**：
  <具体修复方案>

- **联动修改**：
  - [ ] `<其他文件路径>` — <修改内容>
```

## 使用示例

```markdown
### 问题 1：代码笔误 EXPECT_EQ_EQ

- **文件**：`pypto-pass-ut-generate/references/common_errors.md:640`
- **类型**：P0
- **问题**：
  代码示例中使用 `EXPECT_EQ_EQ`，重复了 `EQ`

- **docs 对比**：
  Google Test 标准断言宏为 `EXPECT_EQ`

- **修复建议**：
  将 `EXPECT_EQ_EQ` 改为 `EXPECT_EQ`

- **联动修改**：
  无
```

```markdown
### 问题 2：虚构不存在的 API

- **文件**：`pypto-fracture-point-detector/references/entity-patterns.md:18`
- **类型**：P0
- **问题**：
  列出 `pypto.compile` 作为 API 实体，但 docs 中不存在此 API

- **docs 对比**：
  - `docs/api/pypto-frontend-jit.md` 定义编译入口为 `@pypto.frontend.jit`
  - `grep "pypto.compile" docs/` 无匹配结果

- **修复建议**：
  删除此行，或改为 `pypto.frontend.jit — JIT 编译装饰器`

- **联动修改**：
  无
```

```markdown
### 问题 3：与 docs 矛盾的约束描述

- **文件**：`pypto-environment-setup/references/prepare_environment.md:39`
- **类型**：P0
- **问题**：
  声称"禁止 --type=all"，但 docs 中 `all` 是合法选项

- **docs 对比**：
  `docs/install/prepare_environment.md:132` 明确列出：
  ```
  --type | 可选：deps, cann, third_party, all
  ```

- **修复建议**：
  将"禁止 --type=all"改为"推荐分步安装"或直接删除该限制

- **联动修改**：
  无
```