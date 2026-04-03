#!/usr/bin/env python3
"""全面验证 pypto-op-lint — 覆盖所有规则、hook 模式和边界场景"""
import subprocess
import json
import sys
import os
import tempfile
import shutil

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pypto_op_lint.py"))
FIXTURES = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))
passed = 0
failed = 0


def run(args, label, expect_exit=0, expect_in_stdout=None, expect_not_in_stdout=None):
    global passed, failed
    result = subprocess.run(
        ["python3", SCRIPT] + args,
        capture_output=True, text=True,
        input=getattr(run, "_stdin", None),
    )
    ok = True
    reasons = []
    if result.returncode != expect_exit:
        ok = False
        reasons.append(f"exit={result.returncode}, expected {expect_exit}")
    if expect_in_stdout and expect_in_stdout not in result.stdout:
        ok = False
        reasons.append(f"stdout 缺少: {expect_in_stdout}")
    if expect_not_in_stdout and expect_not_in_stdout in result.stdout:
        ok = False
        reasons.append(f"stdout 不应包含: {expect_not_in_stdout}")
    if not ok:
        failed += 1
        print(f"  FAIL: {label}")
        for r in reasons:
            print(f"        {r}")
        if result.stderr.strip():
            print(f"        stderr: {result.stderr[:200]}")
        if result.stdout.strip():
            print(f"        stdout: {result.stdout[:200]}")
    else:
        passed += 1
        print(f"  PASS: {label}")
    return result


def run_hook(hook_mode, stdin_json, label, expect_exit=0, expect_in_stdout=None):
    global passed, failed
    result = subprocess.run(
        ["python3", SCRIPT, "--hook", hook_mode],
        input=json.dumps(stdin_json),
        capture_output=True, text=True,
    )
    ok = True
    if result.returncode != expect_exit:
        ok = False
    if expect_in_stdout and expect_in_stdout not in result.stdout:
        ok = False
    if not ok:
        failed += 1
        print(f"  FAIL: {label}")
        print(f"        exit={result.returncode}, stdout={result.stdout[:200]}")
    else:
        passed += 1
        print(f"  PASS: {label}")


# ─────────────────────────────────────────────
print("\n=== 1. D1 框架约束 — good_op (全部 PASS) ===")
# ─────────────────────────────────────────────
run(["--lint-impl", "--op-dir", f"{FIXTURES}/good_op", "--stage", "5"],
    "OL01-OL08 + OL16 good_op 通过，OL23 给出 loop 提醒",
    expect_exit=0,
    expect_in_stdout='"warn": 1')

print("\n=== 2. D1 框架约束 — bad_op (无 jit，部分 SKIP) ===")
run(["--lint-impl", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "5"],
    "OL01 FAIL + OL07 FAIL + OL08 FAIL", expect_exit=2, expect_in_stdout='"has_s0_fail": true')

print("\n=== 3. D1 框架约束 — partial_bad_op (有 jit 但内部违规) ===")
run(["--lint-impl", "--op-dir", f"{FIXTURES}/partial_bad_op", "--stage", "5"],
    "OL01 PASS + OL02 FAIL + OL03 FAIL + OL06 FAIL",
    expect_exit=2,
    expect_in_stdout='"has_s0_fail": true')

# 验证 partial_bad_op 具体违规项
result = subprocess.run(
    ["python3", SCRIPT, "--lint-impl", "--op-dir", f"{FIXTURES}/partial_bad_op", "--stage", "5"],
    capture_output=True, text=True,
)
data = json.loads(result.stdout)
fail_rules = {f["rule_id"] for f in data["findings"] if f["status"] == "FAIL"}
print(f"  PASS: partial_bad_op 失败规则: {sorted(fail_rules)}" if {"OL02", "OL03", "OL06"} == fail_rules else f"  FAIL: partial_bad_op 失败规则 {sorted(fail_rules)} != {{'OL02','OL03','OL06'}}")
if {"OL02", "OL03", "OL06"} == fail_rules:
    passed += 1
else:
    failed += 1

# OL01/OL04/OL05/OL07/OL08 应 PASS
pass_rules = {f["rule_id"] for f in data["findings"] if f["status"] == "PASS"}
expected_pass = {"OL01", "OL04", "OL05", "OL07", "OL08", "OL16"}
print(f"  PASS: partial_bad_op 通过规则: {sorted(pass_rules)}" if expected_pass == pass_rules else f"  FAIL: partial_bad_op 通过规则 {sorted(pass_rules)} != {sorted(expected_pass)}")
if expected_pass == pass_rules:
    passed += 1
else:
    failed += 1

print("\n=== 4. D2 阶段门禁 — good_op (工件完整) ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/good_op", "--stage", "5"],
    "OL12 PASS + OL13 PASS", expect_exit=0, expect_in_stdout='"passed": true')

print("\n=== 5. D2 阶段门禁 — bad_op (缺少工件) ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "5"],
    "OL12 FAIL + OL13 FAIL", expect_exit=2, expect_in_stdout='"has_s0_fail": true')

print("\n=== 6. D2 阶段门禁 — missing_files_op (缺三件套) ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/missing_files_op", "--stage", "5"],
    "OL12 FAIL + OL13 FAIL (缺 design.md/README/test)", expect_exit=2)

print("\n=== 7. D2 OL09 — spec.md 检查 ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/good_op", "--stage", "2"],
    "OL09 PASS (spec.md 存在)", expect_exit=0)
run(["--check-gate", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "2"],
    "OL09 FAIL (spec.md 不存在)", expect_exit=2)

print("\n=== 8. D2 OL10 — api_report.md 检查 ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/good_op", "--stage", "3"],
    "OL10 PASS (api_report.md 存在)", expect_exit=0)
run(["--check-gate", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "3"],
    "OL10 FAIL (api_report.md 不存在)", expect_exit=2)

print("\n=== 8B. D2 文档结构门禁 — 必需章节 ===")
with tempfile.TemporaryDirectory() as tmp_dir:
    weak_spec_dir = os.path.join(tmp_dir, "good_op")
    shutil.copytree(f"{FIXTURES}/good_op", weak_spec_dir)
    with open(os.path.join(weak_spec_dir, "spec.md"), "w", encoding="utf-8") as f:
        f.write("# good_op 算子需求规格\n\n## 算子名\ngood_op\n")
    run(["--check-gate", "--op-dir", weak_spec_dir, "--stage", "2"],
        "OL09 FAIL (spec 缺公式/输入输出/精度)", expect_exit=2, expect_in_stdout="缺少必需内容")

with tempfile.TemporaryDirectory() as tmp_dir:
    weak_api_dir = os.path.join(tmp_dir, "good_op")
    shutil.copytree(f"{FIXTURES}/good_op", weak_api_dir)
    with open(os.path.join(weak_api_dir, "api_report.md"), "w", encoding="utf-8") as f:
        f.write("# good_op API 探索报告\n\n## API 映射\npypto.sin -> torch.sin\n")
    run(["--check-gate", "--op-dir", weak_api_dir, "--stage", "3"],
        "OL10 FAIL (api_report 缺约束/Tiling)", expect_exit=2, expect_in_stdout="缺少必需内容")

with tempfile.TemporaryDirectory() as tmp_dir:
    weak_design_dir = os.path.join(tmp_dir, "good_op")
    shutil.copytree(f"{FIXTURES}/good_op", weak_design_dir)
    with open(os.path.join(weak_design_dir, "design.md"), "w", encoding="utf-8") as f:
        f.write(
            "# good_op 设计文档\n\n"
            "## API 映射\npypto.sin -> torch.sin\n\n"
            "## 数据切分策略\nset_vec_tile_shapes(8, 8)\n"
        )
    run(["--check-gate", "--op-dir", weak_design_dir, "--stage", "5"],
        "OL12 FAIL (design 缺验证方案)", expect_exit=2, expect_in_stdout="缺少必需内容")

print("\n=== 9. D2 OL24 — 状态文件结构校验 ===")
run(["--check-gate", "--op-dir", f"{FIXTURES}/good_op", "--stage", "5"],
    "OL24 PASS (状态文件结构合法)", expect_exit=0, expect_in_stdout='"status": "PASS"')

print("\n=== 10. D3 三文件分离 — golden ===")
run(["--lint-golden", "--op-dir", f"{FIXTURES}/good_op", "--stage", "3"],
    "OL15 PASS (golden 无 import pypto)", expect_exit=0)
run(["--lint-golden", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "3"],
    "OL15 FAIL (golden 导入了 pypto)", expect_exit=2)

print("\n=== 11. D3 三文件分离 — partial_bad_op ===")
run(["--lint-golden", "--op-dir", f"{FIXTURES}/partial_bad_op", "--stage", "3"],
    "partial_bad_op golden PASS", expect_exit=0)

print("\n=== 12. D4 测试规范 ===")
run(["--lint-test", "--op-dir", f"{FIXTURES}/good_op", "--stage", "5"],
    "OL17-OL22 good_op 全部 PASS", expect_exit=0, expect_in_stdout='"passed": true')
run(["--lint-test", "--op-dir", f"{FIXTURES}/bad_op", "--stage", "5"],
    "OL17/OL18/OL19/OL20/OL21/OL22 FAIL", expect_exit=2)

# ─────────────────────────────────────────────
print("\n=== 13. Hook: post-edit 合规 impl ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/good_op/good_op_impl.py"}},
         "post-edit good impl 返回 loop 提醒", expect_exit=0, expect_in_stdout="OL23")

print("\n=== 14. Hook: post-edit 违规 impl ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/bad_op/bad_op_impl.py"}},
         "post-edit bad impl 返回 OL01 违规", expect_in_stdout="OL01")

print("\n=== 15. Hook: post-edit partial_bad impl ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/partial_bad_op/partial_bad_op_impl.py"}},
         "post-edit partial_bad 返回 OL02 违规", expect_in_stdout="OL02")

print("\n=== 16. Hook: post-edit 合规 golden ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/good_op/good_op_golden.py"}},
         "post-edit good golden 无输出", expect_exit=0)

print("\n=== 17. Hook: post-edit 违规 golden (stage 3) ===")
# OL15 只在 stage 3 适用，需要修改 bad_op 的状态文件为 stage 3
# 用 partial_bad_op 的 golden（无 pypto 导入）验证 stage 3 放行
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/partial_bad_op/partial_bad_op_golden.py"}},
         "post-edit partial_bad golden stage5 无输出", expect_exit=0)

# 验证 bad_op golden 在 stage 3 触发（临时修改状态文件）
bad_state = f"{FIXTURES}/bad_op/.orchestrator_state.json"
with open(bad_state) as f:
    orig_state = f.read()
import json as _json
state_data = _json.loads(orig_state)
state_data["current_stage"] = 3
with open(bad_state, "w") as f:
    _json.dump(state_data, f)
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/bad_op/bad_op_golden.py"}},
         "post-edit bad golden stage3 返回 OL15",
         expect_in_stdout="OL15")
# 恢复
with open(bad_state, "w") as f:
    f.write(orig_state)

print("\n=== 18. Hook: post-edit 合规 test ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/good_op/test_good_op.py"}},
         "post-edit good test 无输出", expect_exit=0)

print("\n=== 19. Hook: post-edit 违规 test ===")
run_hook("post-edit", {"tool_input": {"file_path": f"{FIXTURES}/bad_op/test_bad_op.py"}},
         "post-edit bad test 返回 OL17 违规", expect_in_stdout="OL17")

print("\n=== 20. Hook: post-edit 非 op 文件 ===")
run_hook("post-edit", {"tool_input": {"file_path": "/tmp/readme.md"}},
         "post-edit 非 op 文件无输出", expect_exit=0)

print("\n=== 21. Hook: post-edit 无状态文件 ===")
run_hook("post-edit", {"tool_input": {"file_path": "/tmp/some_impl.py"}},
         "post-edit 无 .orchestrator_state.json 无输出", expect_exit=0)

print("\n=== 22. Hook: post-bash precision_pass ===")
run_hook("post-bash",
         {"tool_input": {"command": "python3 test_add.py"},
          "tool_result": {"stdout": "[PRECISION_PASS]", "stderr": "", "exit_code": 0}},
         "post-bash precision_pass", expect_in_stdout="precision_pass")

print("\n=== 23. Hook: post-bash precision_fail ===")
run_hook("post-bash",
         {"tool_input": {"command": "python3 test_add.py"},
          "tool_result": {"stdout": "[PRECISION_FAIL]", "stderr": "", "exit_code": 1}},
         "post-bash precision_fail", expect_in_stdout="precision_fail")

print("\n=== 24. Hook: post-bash 运行失败 (无标记) ===")
run_hook("post-bash",
         {"tool_input": {"command": "python3 test_add.py"},
          "tool_result": {"stdout": "Traceback...", "stderr": "RuntimeError", "exit_code": 1}},
         "post-bash other (运行失败)", expect_in_stdout="other")

print("\n=== 25. Hook: post-bash 非 test 命令 ===")
run_hook("post-bash",
         {"tool_input": {"command": "ls -la"},
          "tool_result": {"stdout": "", "stderr": "", "exit_code": 0}},
         "post-bash 非 test 命令无输出", expect_exit=0)

print("\n=== 26. Hook: pre-edit 非 impl 文件 ===")
run_hook("pre-edit",
         {"tool_input": {"file_path": f"{FIXTURES}/good_op/good_op_golden.py"}},
         "pre-edit 非 impl 无输出", expect_exit=0)

print("\n=== 27. Hook: pre-edit 非 Stage 6 ===")
run_hook("pre-edit",
         {"tool_input": {"file_path": f"{FIXTURES}/good_op/good_op_impl.py"}},
         "pre-edit Stage 5 无输出", expect_exit=0)

print("\n=== 28. Hook: pre-edit 无状态文件 ===")
run_hook("pre-edit",
         {"tool_input": {"file_path": "/tmp/some_impl.py"}},
         "pre-edit 无状态文件无输出", expect_exit=0)

print("\n=== 29. 阶段过滤 — Stage 2 不触发 OL01 ===")
result = subprocess.run(
    ["python3", SCRIPT, "--lint-impl", "--op-dir", f"{FIXTURES}/good_op", "--stage", "2"],
    capture_output=True, text=True,
)
data = json.loads(result.stdout)
all_skip = all(f["status"] == "SKIP" for f in data["findings"])
print(f"  {'PASS' if all_skip else 'FAIL'}: Stage 2 时 OL01-OL08 全部 SKIP (阶段不适用)")

print("\n=== 30. 阶段过滤 — Stage 7 触发全部 D1 规则 ===")
result = subprocess.run(
    ["python3", SCRIPT, "--lint-impl", "--op-dir", f"{FIXTURES}/good_op", "--stage", "7"],
    capture_output=True, text=True,
)
data = json.loads(result.stdout)
non_skip = [f for f in data["findings"] if f["status"] != "SKIP"]
all_ok = all(f["status"] in ("PASS", "WARN") for f in non_skip)
print(f"  {'PASS' if all_ok else 'FAIL'}: Stage 7 时 D1 规则通过，OL23 保留提醒")
if all_ok:
    passed += 1
else:
    failed += 1

print("\n=== 31. OL14 — Stage 7 需精度通过 ===")
# good_op stage 5 是 in_progress，不是 completed。OL14 是 S1，非 S0，exit 0
run(["--check-gate", "--op-dir", f"{FIXTURES}/good_op", "--stage", "7"],
    "OL14 FAIL (stage 5 未 completed), S1 非 S0 所以 exit 0", expect_exit=0, expect_in_stdout="OL14")

print("\n=== 32. 边界 — 畸形状态文件 ===")
# malformed_state_op 的状态文件不是合法 JSON
result = subprocess.run(
    ["python3", SCRIPT, "--lint-impl", "--op-dir", f"{FIXTURES}/malformed_state_op", "--stage", "5"],
    capture_output=True, text=True,
)
# 应该 fallback 到目录名作为 op_name，但由于没有对应 impl 文件应该 SKIP
print(f"  {'PASS' if result.returncode in (0, 2) else 'FAIL'}: 畸形状态文件不崩溃 (exit={result.returncode})")

print("\n=== 33. 实际算子 — add ===")
PROJECT_ROOT = os.path.abspath(os.path.join(FIXTURES, "..", "..", "..", "..", ".."))
ADD_DIR = os.path.join(PROJECT_ROOT, "custom", "add")
if os.path.isdir(ADD_DIR):
    run(["--lint-impl", "--op-dir", ADD_DIR, "--stage", "5"],
        "add lint-impl（含 loop 提醒）", expect_exit=0, expect_in_stdout='"warn": 1')
    run(["--lint-golden", "--op-dir", ADD_DIR, "--stage", "3"],
        "add lint-golden", expect_exit=0, expect_in_stdout='"passed": true')
    run(["--lint-test", "--op-dir", ADD_DIR, "--stage", "5"],
        "add lint-test", expect_exit=0, expect_in_stdout='"passed": true')
    run(["--check-gate", "--op-dir", ADD_DIR, "--stage", "5"],
        "add check-gate stage5", expect_exit=0, expect_in_stdout='"passed": true')
    run(["--check-gate", "--op-dir", ADD_DIR, "--stage", "2"],
        "add check-gate stage2", expect_exit=0, expect_in_stdout='"passed": true')
    run(["--check-gate", "--op-dir", ADD_DIR, "--stage", "3"],
        "add check-gate stage3", expect_exit=0, expect_in_stdout='"passed": true')
else:
    print("  SKIP: custom/add 目录不存在")

print("\n=== 34. flash_attention 回归夹具 ===")
FLASH_DIR = os.path.join(FIXTURES, "flash_attention")
run(["--lint-impl", "--op-dir", FLASH_DIR, "--stage", "5"],
    "flash_attention lint-impl（存在 loop，无提醒）", expect_exit=0, expect_in_stdout='"warn": 0')
run(["--lint-golden", "--op-dir", FLASH_DIR, "--stage", "3"],
    "flash_attention lint-golden", expect_exit=0, expect_in_stdout='"passed": true')
run(["--lint-test", "--op-dir", FLASH_DIR, "--stage", "5"],
    "flash_attention lint-test", expect_exit=0, expect_in_stdout='"passed": true')
run(["--check-gate", "--op-dir", FLASH_DIR, "--stage", "2"],
    "flash_attention check-gate stage2", expect_exit=0, expect_in_stdout='"passed": true')
run(["--check-gate", "--op-dir", FLASH_DIR, "--stage", "3"],
    "flash_attention check-gate stage3", expect_exit=0, expect_in_stdout='"passed": true')
run(["--check-gate", "--op-dir", FLASH_DIR, "--stage", "5"],
    "flash_attention check-gate stage5", expect_exit=0, expect_in_stdout='"passed": true')
run_hook("post-edit", {"tool_input": {"file_path": f"{FLASH_DIR}/flash_attention_impl.py"}},
         "post-edit flash_attention impl 无输出", expect_exit=0)
run_hook("post-bash",
         {"tool_input": {"command": "python3 test_flash_attention.py"},
          "tool_result": {"stdout": "[PRECISION_PASS]", "stderr": "", "exit_code": 0}},
         "post-bash flash_attention precision_pass", expect_in_stdout="precision_pass")

print("\n=== 35. Hook: stop — good_op (无 S0 违规) ===")
# 模拟 Stop hook — good_op 在 stage 5，impl 合规，不应 block
# Stop hook 会搜索 cwd 下的 .orchestrator_state.json
run_hook("stop",
         {"cwd": f"{FIXTURES}/good_op"},
         "stop good_op 无 block", expect_exit=0)

print("\n=== 36. Hook: stop — bad_op (S0 违规) ===")
run_hook("stop",
         {"cwd": f"{FIXTURES}/bad_op"},
         "stop bad_op 有 block (OL01 S0)", expect_exit=2, expect_in_stdout="block")

print("\n" + "=" * 50)
total = passed + failed
print(f"总计: {total} 项, 通过: {passed}, 失败: {failed}")
sys.exit(1 if failed > 0 else 0)
