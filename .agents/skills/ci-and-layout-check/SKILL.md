---
name: pypto-kernel-ci-and-layout-check
description: 针对 custom/<operator>/ 的自动化布局校验——CI、pre-commit 以及 Agent 可运行的检查。包含 extract_pypto_calls.py 用于逐算子调试。
---

# PyPTO 复杂 Kernel — CI 与布局检查

本技能打包了 `custom/<operator>/` 目录的自动化验证工具和 CI 集成。

## 内容

| 文件 | 用途 |
|------|---------|
| **`CI.md`** | 完整的 CI 文档——命令、pre-commit 钩子、GitHub Actions 配置 |
| **`run_validate_layout.sh`** | Agent 入口：从仓库根目录运行以检查布局 |
| **`scripts/validate_custom_kernel_layout.py`** | 实现：计划文件、测试运行器、暂存命名、kernel 代码中的 `for...in range`、模板层级头 |
| **`scripts/extract_pypto_calls.py`**** | 按行号列出每个 `pypto.*` 调用点——用于逐算子调试 |

---

## 快速参考

### 布局检查（每次 `custom/` 变更后运行）

```bash
bash .agents/skills/ci-and-layout-check/run_validate_layout.sh
```

等效命令：
```bash
python3 .agents/skills/ci-and-layout-check/scripts/validate_custom_kernel_layout.py \
  --repo-root "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
```

- 尚无 `custom/` 工作 → **exit 0**（无需验证）。
- 发现违规 → **exit 1** 并附带详情。

### 布局检查的内容

| 检查项 | 失败意味着 |
|-------|---------------|
| `custom/plan/<op>.md` 存在 | 缺少计划文件 |
| `custom/<op>/test_<op>.py` 存在且导入了 `detailed_tensor_compare` | 缺少或不完整的 E2E 运行器 |
| 暂存文件使用累积后缀（`1`、`12`、`123`、…） | 命名规范错误 |
| `pypto_function` / kernel 函数中无 `for ... in range(...)` | 应使用 `pypto.loop`（`skills/lead-orchestrator/references/rules.md` 规则 18，禁止项 B） |
| 模板 `# Layer A`–`L` / `# Appendix` 头保持 `LOCKED` 契约 | 不要删除模板头 |

### 提取 PyPTO 调用点（用于调试）

```bash
python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py \
  custom/<operator_name>/<kernel_file>.py
```

添加 `--json` 获取机器可读输出。参见 `skills/debugging/SKILL.md` → 逐算子检查协议。

---

## 何时运行

- **每次有意义的编辑后**在 `custom/` 下（`skills/lead-orchestrator/references/rules.md` 规则 15）
- **在声称布局完成之前**
- 同一验证器在 CI（`.github/workflows/validate-custom-kernel-layout.yml`）和 pre-commit（`.pre-commit-config.yaml`）中运行
