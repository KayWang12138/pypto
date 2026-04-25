# Anti-cheat fixtures

`scripts/local/test-unit.sh` 和 `scripts/local/test-skill-cheat.sh` 用的反作弊
样例代码; 与脚本逻辑解耦, 方便单独人工 cheat-check.

## 5 类机械层 fixture (`<case>_impl.py` 单文件)

| 目录            | 形态                                | cheat_detector 期望 verdict |
|-----------------|-------------------------------------|------------------------------|
| `clean/`        | 1 个 `@pypto.jit`, body 真用 pypto  | `pass`                       |
| `multi_jit/`    | 2 个 `@pypto.jit` 装饰函数          | `cheat` (jit_def_count)      |
| `no_pypto/`     | 不 `import pypto`, 纯 `torch.relu`  | `cheat` (import_pypto+has_jit)|
| `no_jit/`       | `import pypto` 但 0 个 `@pypto.jit` | `cheat` (has_jit)            |
| `suspicious/`   | 1 个 jit + "workaround" 注释        | `suspicious` (forbidden_text)|

## 端到端 cheat 算子 `pypto_op_cheat/`

完整的 PyPTO 算子产物形态 (`<op>_impl.py` + `<op>_pypto_impl.py` + `task_desc.py`),
但故意拆成 2 个 jit kernel + 第 2 个 kernel 是 mock no-op `+ 0.0`. 用于 skill
端到端测试: 验证 `pypto-kernel-validator` agent 能把它判 `FAIL_CHEAT`,
LLM 必须自抓 S5/S6/S9.
