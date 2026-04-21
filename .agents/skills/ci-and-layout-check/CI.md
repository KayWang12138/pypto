# 自动化布局检查（Agent 可运行）

本技能提供一个**无需 NPU** 的校验器，用于检查 `custom/<operator_name>/` 的目录规范（plan 文件、`test_<op>.py`、暂存 `*_module*.py` 命名，以及 `custom/<op>/**/*.py` 中 `pypto_function` 内**不得包含 `for ... in range(...)`**）。

## Agent：从仓库根目录运行

```bash
bash .agents/skills/ci-and-layout-check/run_validate_layout.sh
```

等价命令：

```bash
python3 .agents/skills/ci-and-layout-check/scripts/validate_custom_kernel_layout.py --repo-root "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
```

- 如果尚无 `custom/` 相关工作，脚本**以退出码 0 退出**（无需校验）。
- 如果 `custom/<op>/` 下存在 `test_<op>.py` 或 `<op>_module*.py`，则执行检查——**违规时以退出码 1 退出**。

**运行时机：** 在 `custom/` 下新增或修改文件之后，以及**声明布局完成之前**（参见 **`skills/lead-orchestrator/references/rules.md`**）。

## 仓库集成（人工 / CI）

| 机制 | 位置 |
|-----------|----------|
| **pre-commit** | 仓库根目录 **`.pre-commit-config.yaml`** — hook `validate-custom-kernel-layout` |
| **GitHub Actions** | **`.github/workflows/validate-custom-kernel-layout.yml`** |

在其他克隆中添加该 hook，请合并根目录 **`.pre-commit-config.yaml`** 中的 `repo: local` 块，或在拉取后运行 `pre-commit install`。
