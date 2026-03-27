# Autodev 特殊指令处理

当用户请求以下操作时，直接执行对应命令，不启动算子开发流程。

## 查询进度

用户说"当前开发进度如何"、"查看进度"等：

```bash
python autodev/scripts/get_progress.py \
  --csv autodev/scan_results.csv --format markdown
```

## 重试失败算子

用户说"重试 {op_name}"：

```bash
python autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --reset
```

重置后重新执行 Step 2b 开始选择。该算子因 `fail_count` 保留会降分，但不会因为工作目录已存在而被直接排除。
断点续跑：reset 后 select_next_op 会自动检测已有产物，从断点继续而非从头开始。

## 手动指定算子

用户说"帮我开发 {op_name} 算子"：
1. 调用 `add_op.py` 添加（若已存在则跳过）
2. 从 Step 2b 开始（该算子刚创建，分数高，会被选中）

## 验证已完成算子

用户说"验证 {op_name}"或"验证所有算子"：

```bash
# 验证单个
python autodev/scripts/verify_op.py \
  --csv autodev/scan_results.csv --op {op_name} --run-test

# 验证全部
python autodev/scripts/verify_op.py \
  --csv autodev/scan_results.csv --verify-all --run-test
```

## 产物一致性检查

用户说"检查产物一致性"：

```bash
python autodev/scripts/check_artifacts.py \
  --operators-dir autodev/custom --check-all
```

## 依赖管理

算子间可声明依赖关系，通过 CSV 的 `depends_on` 字段记录（多个依赖用 `|` 分隔）。

- `select_next_op.py` 自动检查依赖：前置算子未 `completed` 的算子不会被选中
- 添加算子时可指定依赖：在 `requirements.md` 中注明，autodev 在 Step 2a 中设置

示例：`depends_on` 字段值为 `exp|sum_reduction` 表示该算子依赖 exp 和 sum_reduction。
