#!/usr/bin/env python3
# coding: utf-8

"""PyPTO {op} operator test (complex-kernel template).

模板说明:
  - 本文件是复杂算子（attention / recurrent / fused 类）的 test_{op}.py 起点模板。
  - 简单算子请使用 ../test-template.py（204 行的精简版）。
  - 所有 {op} 占位符需替换为实际算子名称。
  - test_{op}.py 只做 import + 调用 + 精度对比，不包含 golden 或 kernel 实现代码。
  - golden 实现来自 {op}_golden.py（由 pypto-golden-generate 生成）。
  - kernel 实现来自 {op}_impl.py（由 ./impl-template.py 派生）。
  - 复杂算子推荐使用 detailed_tensor_compare 而非简单的 assert_allclose。

文件职责（Layer L）:
  - 测试驱动: torch_golden_reference vs pypto_function（每个输出 tensor 都要对比）
  - 配置（device, dtype, shape, seed, run_mode）
  - 主入口（main 或 pytest）
"""

from __future__ import annotations

import os
import sys
import argparse

import torch

# 复杂算子推荐使用 bundled 的 detailed_tensor_compare（每 tensor 容差统计 + outlier 排序）。
# 简单算子可改用 numpy.testing.assert_allclose（见 ../test-template.py）。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../.agents/skills/pypto-op-validate/scripts",
))
from detailed_tensor_compare import detailed_tensor_compare  # noqa: E402

from {op}_golden import {op}_golden  # noqa: E402
from {op}_impl import pypto_function  # noqa: E402


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────


def get_device_id() -> int:
    """从环境变量读取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=<chip_id>")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────
#
# 复杂算子的测试流程:
#   1) 配置 seed / device / dtype / shape / run_mode
#   2) 构造输入 tensor
#   3) 调用 golden（{op}_golden）→ ref 输出（可能是 tuple）
#   4) 调用 impl（pypto_function）→ pto 输出（同结构）
#   5) **每个输出 tensor** 单独 detailed_tensor_compare（不要只比第一个）
#   6) 三态标记: [PRECISION_PASS] / [PRECISION_FAIL] / 无标记 + exit≠0
#
# ─────────────────────────────────────────────


def test_{op}_level0(device_id: int = 0, run_mode: str = "npu") -> None:
    """Level 0: 小规模基础功能验证。"""
    print("=" * 60)
    print("Test: {op} Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{{device_id}}" if run_mode == "npu" else "cpu"

    torch.manual_seed(0)

    # TODO: 替换为算子真实输入
    # 示例: x = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    # x = ...

    # TODO: 调用 golden 和 impl
    # ref_out = {op}_golden(x, ...)
    # pto_out = pypto_function(x, ...)

    # TODO: detailed_tensor_compare 每个输出 tensor
    # 单输出:
    #   result = detailed_tensor_compare(pto_out, ref_out, "out", rtol=1e-3, atol=1e-3)
    # 多输出（tuple）:
    #   for name, p, r in zip(["out", "aux"], pto_out, ref_out):
    #       detailed_tensor_compare(p, r, name)

    # TODO: 根据 detailed_tensor_compare 返回的 all_close 输出三态标记
    # if result["all_close"]:
    #     print("[PRECISION_PASS]")
    # else:
    #     print(f"[PRECISION_FAIL] max_diff={{result['max_diff']:.6e}}", file=sys.stderr)
    #     sys.exit(1)

    raise NotImplementedError(f"[{{OP_NAME}}] Agent: implement test_{op}_level0.")


def test_{op}_level1(device_id: int = 0, run_mode: str = "npu") -> None:
    """Level 1: 典型规模验证。"""
    print("=" * 60)
    print("Test: {op} Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{{device_id}}" if run_mode == "npu" else "cpu"

    torch.manual_seed(42)

    # TODO: 替换为典型 shape（attention 类常用 [B, H, S, D]）
    raise NotImplementedError(f"[{{OP_NAME}}] Agent: implement test_{op}_level1.")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "{op}::test_{op}_level0": {
        "name": "{op} Level 0",
        "description": "小数据量基础功能验证",
        "function": test_{op}_level0,
    },
    "{op}::test_{op}_level1": {
        "name": "{op} Level 1",
        "description": "典型场景验证",
        "function": test_{op}_level1,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyPTO {op} operator test (complex kernel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {{key}}  — {{info['description']}}")
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{{args.example_id}}'")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    device_id = 0
    if args.run_mode == "npu":
        device_id = get_device_id()
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for key, info in to_run:
            print(f"\n▸ Running {{key}}: {{info['name']}}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {{e}}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
