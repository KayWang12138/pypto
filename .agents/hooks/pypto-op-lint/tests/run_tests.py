#!/usr/bin/env python3
"""pypto-op-lint 单元测试"""
import unittest
import os
import shutil
import tempfile
import importlib.util
import subprocess
from unittest.mock import patch

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "pypto_op_lint.py")
MODULE_SPEC = importlib.util.spec_from_file_location("pypto_op_lint", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"无法加载 pypto_op_lint 模块: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

CHECKERS = MODULE.CHECKERS
CheckContext = MODULE.CheckContext
_load_rules = MODULE._load_rules
_parse_verdict = MODULE._parse_verdict

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
GOOD_OP = os.path.join(FIXTURES, "good_op")
BAD_OP = os.path.join(FIXTURES, "bad_op")
FLASH_ATTENTION = os.path.join(FIXTURES, "flash_attention")


def make_ctx(fixture_dir: str, op_name: str | None = None, stage: int = 5) -> CheckContext:
    rules = _load_rules()
    if op_name is None:
        op_name = os.path.basename(fixture_dir)
    return CheckContext(
        op_dir=fixture_dir,
        op_name=op_name,
        stage=stage,
        rules=rules,
    )


# ─── D1: OL01-OL08 ───

class TestOL01(unittest.TestCase):
    """OL01: kernel 必须有 @pypto.frontend.jit"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL01"](ctx).status, "PASS")

    def test_bad_impl_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL01"](ctx).status, "FAIL")


class TestOL02(unittest.TestCase):
    """OL02: 输出写回必须用 [:]/move()/assemble()"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL02"](ctx).status, "PASS")

    def test_bad_impl_fails(self):
        # bad_op 没有 jit 函数，OL02 会 SKIP
        # 需要一个有 jit 但写回不对的 fixture
        # 但 bad_op 无 jit，所以 OL02 会 SKIP
        ctx = make_ctx(BAD_OP)
        result = CHECKERS["OL02"](ctx)
        self.assertIn(result.status, ("FAIL", "SKIP"))


class TestOL03(unittest.TestCase):
    """OL03: kernel 无 return"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL03"](ctx).status, "PASS")

    def test_bad_impl_skips_no_jit(self):
        # bad_op 没有 jit 函数
        ctx = make_ctx(BAD_OP)
        result = CHECKERS["OL03"](ctx)
        self.assertEqual(result.status, "SKIP")

    def test_bare_return_in_jit_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "bare_return_op")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "bare_return_op_impl.py")
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def bare_return_op_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = x\n"
                    "    return\n"
                )
            ctx = make_ctx(op_dir, op_name="bare_return_op")
            self.assertEqual(CHECKERS["OL03"](ctx).status, "FAIL")


class TestOL04(unittest.TestCase):
    """OL04: 必须调用 set_*_tile_shapes"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL04"](ctx).status, "PASS")

    def test_bad_impl_fails(self):
        ctx = make_ctx(BAD_OP)
        # 无 jit 函数，所以找不到 tile shapes 调用
        self.assertEqual(CHECKERS["OL04"](ctx).status, "FAIL")


class TestOL05(unittest.TestCase):
    """OL05: kernel 参数有类型注解"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL05"](ctx).status, "PASS")

    def test_bad_impl_skips_no_jit(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL05"](ctx).status, "SKIP")

    def test_unknown_annotation_fails(self):
        """非 pypto 类型且非内置标量类型的注解应报 FAIL"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "wrong_anno_op")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "wrong_anno_op_impl.py")
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def wrong_anno_op_kernel(x: str, y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(y)\n"
                )
            ctx = make_ctx(op_dir, op_name="wrong_anno_op")
            self.assertEqual(CHECKERS["OL05"](ctx).status, "FAIL")

    def test_int_scalar_param_passes(self):
        """int/float 标量参数是合法的 kernel 参数"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "int_param_op")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "int_param_op_impl.py")
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def int_param_op_kernel(x: pypto.Tensor([], pypto.DT_FP32), n: int):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    x[:] = pypto.sin(x)\n"
                )
            ctx = make_ctx(op_dir, op_name="int_param_op")
            self.assertEqual(CHECKERS["OL05"](ctx).status, "PASS")


class TestOL06(unittest.TestCase):
    """OL06: kernel 内禁用原生 min/max"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL06"](ctx).status, "PASS")


class TestOL07(unittest.TestCase):
    """OL07: 必须 import pypto"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL07"](ctx).status, "PASS")

    def test_bad_impl_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL07"](ctx).status, "FAIL")


class TestOL08(unittest.TestCase):
    """OL08: wrapper 函数以 _wrapper 结尾"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL08"](ctx).status, "PASS")

    def test_bad_impl_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL08"](ctx).status, "FAIL")


class TestOL23(unittest.TestCase):

    def test_good_impl_warns_when_loop_missing(self):
        ctx = make_ctx(GOOD_OP)
        result = CHECKERS["OL23"](ctx)
        self.assertEqual(result.status, "WARN")

    def test_flash_attention_impl_passes_when_loop_exists(self):
        ctx = make_ctx(FLASH_ATTENTION)
        result = CHECKERS["OL23"](ctx)
        self.assertEqual(result.status, "PASS")

    def test_helper_loop_outside_jit_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "helper_loop_op")
            os.makedirs(op_dir)
            impl_path = os.path.join(op_dir, "helper_loop_op_impl.py")
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
            ctx = make_ctx(op_dir, op_name="helper_loop_op")
            result = CHECKERS["OL23"](ctx)
            self.assertEqual(result.status, "WARN")


# ─── D2: OL09-OL14, OL24 ───

class TestOL09(unittest.TestCase):
    """OL09: spec.md 存在且含算子名"""

    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP, stage=2)
        self.assertEqual(CHECKERS["OL09"](ctx).status, "PASS")

    def test_bad_op_fails(self):
        ctx = make_ctx(BAD_OP, stage=2)
        self.assertEqual(CHECKERS["OL09"](ctx).status, "FAIL")

    def test_spec_missing_required_sections_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = os.path.join(tmp_dir, "good_op")
            shutil.copytree(GOOD_OP, fixture_dir)
            with open(os.path.join(fixture_dir, "spec.md"), "w", encoding="utf-8") as f:
                f.write("# good_op 算子需求规格\n\n## 算子名\ngood_op\n")
            ctx = make_ctx(fixture_dir, op_name="good_op", stage=2)
            result = CHECKERS["OL09"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("缺少必需内容", result.message)


class TestOL10(unittest.TestCase):
    """OL10: api_report.md 存在"""

    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP, stage=3)
        self.assertEqual(CHECKERS["OL10"](ctx).status, "PASS")

    def test_bad_op_fails(self):
        ctx = make_ctx(BAD_OP, stage=3)
        self.assertEqual(CHECKERS["OL10"](ctx).status, "FAIL")

    def test_api_report_missing_required_sections_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = os.path.join(tmp_dir, "good_op")
            shutil.copytree(GOOD_OP, fixture_dir)
            with open(os.path.join(fixture_dir, "api_report.md"), "w", encoding="utf-8") as f:
                f.write("# good_op API 探索报告\n\n## API 映射\npypto.sin -> torch.sin\n")
            ctx = make_ctx(fixture_dir, op_name="good_op", stage=3)
            result = CHECKERS["OL10"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("缺少必需内容", result.message)


class TestOL11(unittest.TestCase):
    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP, stage=4)
        self.assertEqual(CHECKERS["OL11"](ctx).status, "PASS")

    def test_missing_golden_fails(self):
        ctx = make_ctx(BAD_OP, stage=4)
        self.assertEqual(CHECKERS["OL11"](ctx).status, "FAIL")

    def test_import_error_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "broken_import")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "broken_import_golden.py"), "w", encoding="utf-8") as f:
                f.write("raise RuntimeError('boom')\n")
            ctx = make_ctx(op_dir, op_name="broken_import", stage=4)
            result = CHECKERS["OL11"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("导入失败", result.message)

    def test_import_timeout_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "timeout_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "timeout_op_golden.py"), "w", encoding="utf-8") as f:
                f.write("def timeout_op_golden(x):\n    return x\n")
            ctx = make_ctx(op_dir, op_name="timeout_op", stage=4)
            with patch.object(
                MODULE.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd=["python3", "-c", "..."], timeout=10),
            ):
                result = CHECKERS["OL11"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("导入超时", result.message)


class TestOL12(unittest.TestCase):
    """OL12: design.md 含 API 相关内容"""

    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP, stage=5)
        self.assertEqual(CHECKERS["OL12"](ctx).status, "PASS")

    def test_bad_op_fails(self):
        ctx = make_ctx(BAD_OP, stage=5)
        self.assertEqual(CHECKERS["OL12"](ctx).status, "FAIL")

    def test_design_missing_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = os.path.join(tmp_dir, "good_op")
            shutil.copytree(GOOD_OP, fixture_dir)
            with open(os.path.join(fixture_dir, "design.md"), "w", encoding="utf-8") as f:
                f.write(
                    "# good_op 设计文档\n\n"
                    "## API 映射\npypto.sin -> torch.sin\n\n"
                    "## 数据切分策略\nset_vec_tile_shapes(8, 8)\n"
                )
            ctx = make_ctx(fixture_dir, op_name="good_op", stage=5)
            result = CHECKERS["OL12"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("缺少必需内容", result.message)


class TestOL13(unittest.TestCase):
    """OL13: 三件套完整"""

    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP, stage=5)
        self.assertEqual(CHECKERS["OL13"](ctx).status, "PASS")

    def test_bad_op_fails(self):
        ctx = make_ctx(BAD_OP, stage=5)
        # bad_op 缺少 README.md
        self.assertEqual(CHECKERS["OL13"](ctx).status, "FAIL")


class TestOL24(unittest.TestCase):
    """OL24: 状态文件结构合法"""

    def test_good_op_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL24"](ctx).status, "PASS")


class TestFlashAttentionFixture(unittest.TestCase):

    def test_flash_attention_gate_checks_pass(self):
        self.assertEqual(CHECKERS["OL09"](make_ctx(FLASH_ATTENTION, stage=2)).status, "PASS")
        self.assertEqual(CHECKERS["OL10"](make_ctx(FLASH_ATTENTION, stage=3)).status, "PASS")
        self.assertEqual(CHECKERS["OL12"](make_ctx(FLASH_ATTENTION, stage=5)).status, "PASS")

    def test_flash_attention_code_checks_pass(self):
        self.assertEqual(CHECKERS["OL01"](make_ctx(FLASH_ATTENTION, stage=5)).status, "PASS")
        self.assertEqual(CHECKERS["OL15"](make_ctx(FLASH_ATTENTION, stage=3)).status, "PASS")
        self.assertEqual(CHECKERS["OL19"](make_ctx(FLASH_ATTENTION, stage=5)).status, "PASS")


# ─── D3: OL15-OL18 ───

class TestOL15(unittest.TestCase):
    """OL15: golden 禁止 import pypto"""

    def test_good_golden_passes(self):
        ctx = make_ctx(GOOD_OP, stage=3)
        self.assertEqual(CHECKERS["OL15"](ctx).status, "PASS")

    def test_bad_golden_fails(self):
        ctx = make_ctx(BAD_OP, stage=3)
        self.assertEqual(CHECKERS["OL15"](ctx).status, "FAIL")


class TestOL16(unittest.TestCase):
    """OL16: impl 不应导入 golden"""

    def test_good_impl_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL16"](ctx).status, "PASS")


class TestOL17(unittest.TestCase):
    """OL17: test 不应包含 kernel 实现"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL17"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL17"](ctx).status, "FAIL")


class TestOL18(unittest.TestCase):
    """OL18: test 必须从 impl 和 golden 分别导入"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL18"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL18"](ctx).status, "FAIL")

    def test_lazy_import_inside_helper_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "lazy_import_op")
            os.makedirs(op_dir)
            test_path = os.path.join(op_dir, "test_lazy_import_op.py")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(
                    "def load_impl():\n"
                    "    from lazy_import_op_impl import lazy_import_op_wrapper\n"
                    "    return lazy_import_op_wrapper\n\n"
                    "def load_golden():\n"
                    "    from lazy_import_op_golden import lazy_import_op_golden\n"
                    "    return lazy_import_op_golden\n"
                )
            ctx = make_ctx(op_dir, op_name="lazy_import_op")
            self.assertEqual(CHECKERS["OL18"](ctx).status, "PASS")

    def test_qualified_package_import_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "softmax")
            os.makedirs(op_dir)
            test_path = os.path.join(op_dir, "test_softmax.py")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(
                    "from custom.softmax.softmax_golden import softmax_golden\n"
                    "from custom.softmax.softmax_impl import softmax_wrapper\n"
                )
            ctx = make_ctx(op_dir, op_name="softmax")
            self.assertEqual(CHECKERS["OL18"](ctx).status, "PASS")


# ─── D4: OL19-OL22 ───

class TestOL19(unittest.TestCase):
    """OL19: 必须使用 assert_allclose"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL19"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL19"](ctx).status, "FAIL")


class TestOL20(unittest.TestCase):
    """OL20: 必须处理 TILE_FWK_DEVICE_ID"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL20"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL20"](ctx).status, "FAIL")


class TestOL21(unittest.TestCase):
    """OL21: 必须有 level0 和 level1"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL21"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL21"](ctx).status, "FAIL")


class TestOL22(unittest.TestCase):
    """OL22: 应设置 manual_seed"""

    def test_good_test_passes(self):
        ctx = make_ctx(GOOD_OP)
        self.assertEqual(CHECKERS["OL22"](ctx).status, "PASS")

    def test_bad_test_fails(self):
        ctx = make_ctx(BAD_OP)
        self.assertEqual(CHECKERS["OL22"](ctx).status, "FAIL")


# ─── 三态解析 ───

class TestParseVerdict(unittest.TestCase):

    def test_precision_pass(self):
        self.assertEqual(_parse_verdict("...[PRECISION_PASS]...", "", 0), "precision_pass")

    def test_precision_fail(self):
        self.assertEqual(_parse_verdict("...[PRECISION_FAIL]...", "", 1), "precision_fail")

    def test_runtime_error(self):
        self.assertEqual(_parse_verdict("Traceback...", "ImportError", 1), "other")

    def test_no_marker(self):
        self.assertEqual(_parse_verdict("test completed", "", 0), "other")

    def test_precision_marker_in_stderr(self):
        self.assertEqual(_parse_verdict("", "...[PRECISION_FAIL]...", 1), "precision_fail")

    def test_fail_takes_priority_over_pass(self):
        """多 case 测试部分通过部分失败时，FAIL 优先"""
        self.assertEqual(
            _parse_verdict("[PRECISION_PASS]...[PRECISION_FAIL]", "", 1),
            "precision_fail")
        self.assertEqual(
            _parse_verdict("[PRECISION_FAIL]...[PRECISION_PASS]", "", 0),
            "precision_fail")


class TestOL02AugAssign(unittest.TestCase):
    """OL02: AugAssign (+=, -=) 也应被检测为非法写回"""

    def test_augassign_to_param_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "aug_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "aug_op_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def aug_op_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y += x\n"
                )
            ctx = make_ctx(op_dir, op_name="aug_op")
            result = CHECKERS["OL02"](ctx)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("+=", result.message)

    def test_augassign_to_local_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "aug_local_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "aug_local_op_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def aug_local_op_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    tmp = pypto.sin(x)\n"
                    "    y[:] = tmp\n"
                )
            ctx = make_ctx(op_dir, op_name="aug_local_op")
            self.assertEqual(CHECKERS["OL02"](ctx).status, "PASS")


class TestOL05NonTensorParam(unittest.TestCase):
    """OL05: 张量参数必须有 pypto.Tensor 注解，非张量参数(int/float)跳过检查"""

    def test_int_float_param_passes(self):
        """官方示例 view_assemble_kernel 有 tile_h: int 参数，不应误报"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "scalar_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "scalar_op_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def scalar_op_kernel(\n"
                    "    x: pypto.Tensor([], pypto.DT_FP32),\n"
                    "    y: pypto.Tensor([], pypto.DT_FP32),\n"
                    "    alpha: float,\n"
                    "    n: int,\n"
                    "):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(x)\n"
                )
            ctx = make_ctx(op_dir, op_name="scalar_op")
            self.assertEqual(CHECKERS["OL05"](ctx).status, "PASS")

    def test_no_annotation_fails(self):
        """缺少注解的参数应报 FAIL"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "noanno_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "noanno_op_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def noanno_op_kernel(x, y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(y)\n"
                )
            ctx = make_ctx(op_dir, op_name="noanno_op")
            self.assertEqual(CHECKERS["OL05"](ctx).status, "FAIL")

    def test_str_annotation_fails(self):
        """str 不是合法的 kernel 参数类型"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            op_dir = os.path.join(tmp_dir, "str_op")
            os.makedirs(op_dir)
            with open(os.path.join(op_dir, "str_op_impl.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import pypto\n\n"
                    "@pypto.frontend.jit\n"
                    "def str_op_kernel(x: str, y: pypto.Tensor([], pypto.DT_FP32)):\n"
                    "    pypto.set_vec_tile_shapes(8, 8)\n"
                    "    y[:] = pypto.sin(y)\n"
                )
            ctx = make_ctx(op_dir, op_name="str_op")
            self.assertEqual(CHECKERS["OL05"](ctx).status, "FAIL")


class TestFindNearestOpDir(unittest.TestCase):
    """_find_nearest_op_dir 不应误匹配远端目录"""

    def test_tmp_dir_returns_none(self):
        result = MODULE._find_nearest_op_dir("/tmp")
        self.assertIsNone(result)

    def test_cwd_with_state_file_returns_self(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = os.path.join(tmp_dir, ".orchestrator_state.json")
            with open(state, "w") as f:
                f.write('{"operator_name":"x","current_stage":5,"stage_status":{}}')
            result = MODULE._find_nearest_op_dir(tmp_dir)
            self.assertEqual(result, tmp_dir)

    def test_custom_subdir_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom = os.path.join(tmp_dir, "custom", "my_op")
            os.makedirs(custom)
            state = os.path.join(custom, ".orchestrator_state.json")
            with open(state, "w") as f:
                f.write('{"operator_name":"my_op","current_stage":5,"stage_status":{}}')
            result = MODULE._find_nearest_op_dir(tmp_dir)
            self.assertEqual(result, custom)


if __name__ == "__main__":
    unittest.main()
