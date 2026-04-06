# 特殊指令

以下操作直接执行，不启动算子开发流程。命令中 `{scripts}` 和 `{csv_path}` 等变量含义见 SKILL.md 的"运行时变量"部分。

## 查询进度

```bash
python {scripts}/get_progress.py --csv {csv_path} --format markdown
```

## 重试失败算子

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} --reset
```

重置后从 Step 2b 开始选择。`fail_count` 保留（降分但不排除）。reset 后 `select_next_op` 自动检测已有产物，支持断点续跑。

## 手动指定算子

```bash
python {scripts}/add_op.py --csv {csv_path} --op {op_name} --source manual \
  --complexity {推断} --category {推断} \
  --description "一句话需求描述" \
  [--requirement sources/{op_name}.md] \
  [--reference "torch.nn.functional.softmax"]
```

`--description` 为必填项。`--requirement` 和 `--reference` 可选。添加后从 Step 2b 开始（新算子分数高，会被选中）。

## 验证算子

```bash
# 单个
python {scripts}/verify_op.py --csv {csv_path} --op {op_name} --run-test
# 全部
python {scripts}/verify_op.py --csv {csv_path} --verify-all --run-test
```

## 产物一致性检查

```bash
python {scripts}/check_artifacts.py --operators-dir {work_dir} --check-all
```

## 依赖管理

CSV `depends_on` 字段记录依赖（多个用 `|` 分隔，如 `exp|sum_reduction`）。`select_next_op.py` 自动检查：前置算子未 completed 的不会被选中。添加算子时可在 `requirements.md` 中注明依赖。
