#!/usr/bin/env python3
"""Hook 模式集成测试 — 模拟 Claude Code hook JSON 输入输出"""
import unittest
import subprocess
import json
import os
import tempfile
from typing import Any

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "pypto_op_lint.py")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestPostEditHook(unittest.TestCase):

    def _run_hook(self, hook_mode: str, stdin_data: dict[str, Any]) -> dict[str, Any]:
        result = subprocess.run(
            ["python3", SCRIPT, "--hook", hook_mode],
            input=json.dumps(stdin_data),
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
        return {}

    def test_bad_impl_returns_violations(self):
        bad_impl_path = os.path.join(FIXTURES, "bad_op", "bad_op_impl.py")
        stdin = {"tool_input": {"file_path": bad_impl_path}}
        output = self._run_hook("post-edit", stdin)
        self.assertIn("hookSpecificOutput", output)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("pypto-op-lint", ctx)
        self.assertIn("OL01", ctx)

    def test_good_impl_no_output(self):
        good_impl_path = os.path.join(FIXTURES, "good_op", "good_op_impl.py")
        stdin = {"tool_input": {"file_path": good_impl_path}}
        output = self._run_hook("post-edit", stdin)
        self.assertIn("hookSpecificOutput", output)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("OL23", ctx)
        self.assertIn("无需 loop", ctx)

    def test_flash_attention_impl_no_output_when_loop_exists(self):
        flash_impl_path = os.path.join(FIXTURES, "flash_attention", "flash_attention_impl.py")
        stdin = {"tool_input": {"file_path": flash_impl_path}}
        output = self._run_hook("post-edit", stdin)
        self.assertEqual(output, {})

    def test_helper_loop_outside_jit_returns_loop_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "helper_loop_op")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "helper_loop_op_impl.py")
            state_path = os.path.join(op_dir, ".orchestrator_state.json")
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "def helper(values):\n"
                    "    for value in values:\n"
                    "        if value:\n"
                    "            return value\n"
                    "    return None\n\n"
                    "@pypto.frontend.jit\n"
                    "def helper_loop_op_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(x)\n"
                )
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "operator_name": "helper_loop_op",
                    "current_stage": 5,
                    "stage_status": {"5": "in_progress"},
                }, f)
            stdin = {"tool_input": {"file_path": impl_path}}
            output = self._run_hook("post-edit", stdin)
            self.assertIn("hookSpecificOutput", output)
            ctx = output["hookSpecificOutput"]["additionalContext"]
            self.assertIn("OL23", ctx)
            self.assertIn("未检测到 loop 相关结构", ctx)

    def test_irrelevant_file_no_output(self):
        stdin = {"tool_input": {"file_path": "/some/path/readme.md"}}
        output = self._run_hook("post-edit", stdin)
        self.assertEqual(output, {})

    def test_no_state_file_no_output(self):
        """没有 .orchestrator_state.json 的目录不触发"""
        stdin = {"tool_input": {"file_path": "/tmp/some_impl.py"}}
        output = self._run_hook("post-edit", stdin)
        self.assertEqual(output, {})

    def test_stateless_workflow_op_dir_still_triggers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "gelu")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "gelu_impl.py")
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n"
                    "import torch\n\n"
                    "@pypto.frontend.jit\n"
                    "def gelu_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(x)\n\n"
                    "def gelu_wrapper(x: torch.Tensor) -> torch.Tensor:\n"
                    "    y = torch.empty_like(x)\n"
                    "    gelu_kernel(x, y)\n"
                    "    return y\n"
                )
            with open(os.path.join(op_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("# gelu\n")
            stdin = {"tool_input": {"file_path": impl_path}}
            output = self._run_hook("post-edit", stdin)
            self.assertIn("hookSpecificOutput", output)
            ctx = output["hookSpecificOutput"]["additionalContext"]
            self.assertIn("OL23", ctx)

    def test_invalid_json_input_soft_exits(self):
        result = subprocess.run(
            ["python3", SCRIPT, "--hook", "post-edit"],
            input="not-json",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")


class TestPostBashHook(unittest.TestCase):

    def _run_hook(self, stdin_data: dict[str, Any]) -> dict[str, Any]:
        result = subprocess.run(
            ["python3", SCRIPT, "--hook", "post-bash"],
            input=json.dumps(stdin_data),
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
        return {}

    def test_precision_pass(self):
        stdin = {
            "tool_input": {"command": "python3 test_mock_op.py"},
            "tool_result": {"stdout": "...[PRECISION_PASS]...", "stderr": "", "exit_code": 0},
        }
        output = self._run_hook(stdin)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("precision_pass", ctx)

    def test_precision_fail(self):
        stdin = {
            "tool_input": {"command": "python3 test_mock_op.py"},
            "tool_result": {"stdout": "...[PRECISION_FAIL]...", "stderr": "", "exit_code": 1},
        }
        output = self._run_hook(stdin)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("precision_fail", ctx)

    def test_non_test_command_no_output(self):
        stdin = {
            "tool_input": {"command": "ls -la"},
            "tool_result": {"stdout": "", "stderr": "", "exit_code": 0},
        }
        output = self._run_hook(stdin)
        self.assertEqual(output, {})

    def test_precision_fail_marker_in_stderr(self):
        stdin = {
            "tool_input": {"command": "python3 test_mock_op.py"},
            "tool_result": {"stdout": "", "stderr": "...[PRECISION_FAIL]...", "exit_code": 1},
        }
        output = self._run_hook(stdin)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("precision_fail", ctx)


class TestStopHook(unittest.TestCase):

    def _run_hook(self, stdin_data: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", SCRIPT, "--hook", "stop"],
            input=json.dumps(stdin_data),
            capture_output=True, text=True,
        )

    def test_stateless_workflow_op_dir_can_block(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "demo")
            os.makedirs(op_dir)

            with open(os.path.join(op_dir, "demo_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "def demo_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(x)\n\n"
                    "def demo_wrapper(x):\n"
                    "    return x\n"
                )
            with open(os.path.join(op_dir, "demo_golden.py"), "w", encoding="utf-8") as f:
                f.write("def demo_golden(x):\n    return x\n")
            with open(os.path.join(op_dir, "test_demo.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import os\n"
                    "import torch\n"
                    "from numpy.testing import assert_allclose\n"
                    "from demo_impl import demo_wrapper\n"
                    "from demo_golden import demo_golden\n\n"
                    "def test_demo_level0():\n"
                    "    assert_allclose([1], [1])\n\n"
                    "def test_demo_level1():\n"
                    "    torch.manual_seed(0)\n"
                    "    _ = os.environ.get('TILE_FWK_DEVICE_ID')\n"
                    "    assert_allclose([1], [1])\n"
                )
            with open(os.path.join(op_dir, "design.md"), "w", encoding="utf-8") as f:
                f.write("## API 映射\n## 数据切分 / Tiling\n## 验证方案\n")
            with open(os.path.join(op_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("# demo\n")

            result = self._run_hook({"cwd": op_dir})
            self.assertEqual(result.returncode, 2)
            self.assertIn("block", result.stdout)


if __name__ == "__main__":
    unittest.main()
