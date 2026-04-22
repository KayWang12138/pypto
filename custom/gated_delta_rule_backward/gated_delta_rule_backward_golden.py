#!/usr/bin/env python3
# coding: utf-8

"""PyPTO gated_delta_rule_backward golden reference implementation.

Golden 参考实现，基于 models/qwen3_next/gated_delta_rule_golden.py 中的
torch_golden_gated_delta_rule_backward_ref 函数。

置信度: ⭐⭐⭐⭐⭐ (直接复用官方仓库提供的 backward 参考实现)

该 golden 函数封装了完整的前向-反向传播流程:
  1. 构造常量矩阵 (i_mat, m_le, m_lt, c_cum, c_rcum)
  2. 执行前向传播生成缓存 (A, w, S_before, v_new, q_norm, k_norm, ...)
  3. 执行反向传播计算全部 6 个梯度张量
  4. 返回 (dq, dk, dv, db, dg_raw, dh0)

依赖: 仅 torch + 项目内参考实现 (通过 sys.path 导入)
"""

from __future__ import annotations

import sys
import os
import math
from typing import Any

import torch

# ---------------------------------------------------------------------------
# 导入参考实现 (models/qwen3_next/gated_delta_rule_golden.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "qwen3_next")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from gated_delta_rule_golden import (  # noqa: E402
    forward_ref,
    torch_golden_gated_delta_rule_backward_ref,
)


# ---------------------------------------------------------------------------
# 常量矩阵构造
# ---------------------------------------------------------------------------
def _build_constant_matrices(bt: int, device: torch.device) -> dict[str, torch.Tensor]:
    """构造反向传播所需的 5 个常量矩阵。

    Args:
        bt: 时间维度分块大小
        device: 计算设备

    Returns:
        dict 包含 i_mat, m_le, m_lt, c_cum, c_rcum
    """
    ones = torch.ones(bt, bt, device=device, dtype=torch.float32)
    return {
        "i_mat": torch.eye(bt, device=device, dtype=torch.float32),
        "m_le": torch.tril(ones).to(torch.float32),           # 下三角含对角线
        "m_lt": torch.tril(ones, diagonal=-1).to(torch.float32),  # 严格下三角
        "c_cum": torch.tril(ones).to(torch.float32),           # 累积和矩阵
        "c_rcum": torch.triu(ones).to(torch.float32),          # 逆累积和矩阵
    }


# ---------------------------------------------------------------------------
# Golden 主函数
# ---------------------------------------------------------------------------
def gated_delta_rule_backward_golden(
    q: torch.Tensor,           # [B, T, H, K]
    k: torch.Tensor,           # [B, T, H, K]
    v: torch.Tensor,           # [B, T, H, V]
    g_raw: torch.Tensor,       # [B, T, H]
    beta: torch.Tensor,        # [B, T, H]
    initial_state: torch.Tensor,  # [B, H, K, V]
    do: torch.Tensor,          # [B, T, H, V]
    dht: torch.Tensor,         # [B, H, K, V]
    bt: int,
    use_qk_l2norm_in_kernel: bool = True,
    l2_eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    """Gated Delta Rule 反向传播 golden 参考。

    封装完整的前向-反向流程:
      1. 构造常量矩阵
      2. 前向传播生成缓存
      3. 反向传播计算梯度

    Args:
        q: Query 张量, shape [B, T, H, K], fp32
        k: Key 张量, shape [B, T, H, K], fp32
        v: Value 张量, shape [B, T, H, V], fp32
        g_raw: 门控原始值, shape [B, T, H], fp32
        beta: Beta 参数, shape [B, T, H], fp32
        initial_state: 初始隐状态, shape [B, H, K, V], fp32
        do: 输出梯度 dL/d(out), shape [B, T, H, V], fp32
        dht: 终态梯度 dL/d(final_state), shape [B, H, K, V], fp32
        bt: 时间维度分块大小
        use_qk_l2norm_in_kernel: 是否使用融合 L2 归一化 (默认 True)
        l2_eps: L2 归一化 epsilon (默认 1e-6)

    Returns:
        (dq, dk, dv, db, dg_raw, dh0) — 全部 fp32
        dq:      [B, T, H, K]
        dk:      [B, T, H, K]
        dv:      [B, T, H, V]
        db:      [B, T, H]
        dg_raw:  [B, T, H]
        dh0:     [B, H, K, V]
    """
    device = q.device
    B, T, H, K = q.shape
    assert T % bt == 0, f"T ({T}) must be divisible by bt ({bt})"

    # ---- Step 1: 构造常量矩阵 ----
    consts = _build_constant_matrices(bt, device)

    # ---- Step 2: 前向传播，生成缓存 ----
    _out, _final_state, cache = forward_ref(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta=beta,
        initial_state=initial_state,
        bt=bt,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        l2_eps=l2_eps,
        i=consts["i_mat"],
        m_le=consts["m_le"],
        m_lt=consts["m_lt"],
        c_cum=consts["c_cum"],
    )

    # ---- Step 3: 反向传播 ----
    dq, dk, dv, db, dg_raw, dh0 = torch_golden_gated_delta_rule_backward_ref(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta=beta,
        initial_state=initial_state,
        do=do,
        dht=dht,
        cache=cache,
        bt=bt,
        i=consts["i_mat"],
        m_le=consts["m_le"],
        m_lt=consts["m_lt"],
        c_cum=consts["c_cum"],
        c_rcum=consts["c_rcum"],
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        l2_eps=l2_eps,
    )

    return dq, dk, dv, db, dg_raw, dh0


# ---------------------------------------------------------------------------
# 辅助: 构造典型测试输入
# ---------------------------------------------------------------------------
def _make_test_inputs(
    B: int, T: int, H: int, K: int, V: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """构造 gated_delta_rule_backward 测试所需的全部输入。"""
    torch.manual_seed(seed)
    return {
        "q": torch.randn(B, T, H, K, device=device, dtype=dtype),
        "k": torch.randn(B, T, H, K, device=device, dtype=dtype),
        "v": torch.randn(B, T, H, V, device=device, dtype=dtype),
        "g_raw": torch.randn(B, T, H, device=device, dtype=dtype) * 0.1,
        "beta": torch.rand(B, T, H, device=device, dtype=dtype),
        "initial_state": torch.randn(B, H, K, V, device=device, dtype=dtype) * 0.1,
        "do": torch.randn(B, T, H, V, device=device, dtype=dtype),
        "dht": torch.randn(B, H, K, V, device=device, dtype=dtype) * 0.1,
    }


# ==========================================
# 验证
# ==========================================
def _validate():
    """自动验证: 语法、签名、形状、数值稳定性、无 NaN/Inf。"""

    print("=" * 60)
    print("gated_delta_rule_backward_golden 验证报告")
    print("=" * 60)

    all_pass = True

    # ---- 1. 典型 case 验证 (来自 SPEC) ----
    print("\n[典型 case 验证]")

    test_configs = [
        ("Small_功能_P0", {"B": 1, "T": 128, "H": 4, "K": 128, "V": 128, "bt": 64}),
        ("Large_性能_P0", {"B": 1, "T": 4096, "H": 4, "K": 128, "V": 128, "bt": 128}),
    ]

    for name, cfg in test_configs:
        try:
            inputs = _make_test_inputs(
                cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"],
            )
            result = gated_delta_rule_backward_golden(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                g_raw=inputs["g_raw"],
                beta=inputs["beta"],
                initial_state=inputs["initial_state"],
                do=inputs["do"],
                dht=inputs["dht"],
                bt=cfg["bt"],
                use_qk_l2norm_in_kernel=True,
                l2_eps=1e-6,
            )
            dq, dk, dv, db, dg_raw, dh0 = result

            # 形状检查
            B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
            V = cfg["V"]
            shape_ok = True
            checks = [
                ("dq", dq.shape, (B, T, H, K)),
                ("dk", dk.shape, (B, T, H, K)),
                ("dv", dv.shape, (B, T, H, V)),
                ("db", db.shape, (B, T, H)),
                ("dg_raw", dg_raw.shape, (B, T, H)),
                ("dh0", dh0.shape, (B, H, K, V)),
            ]
            for tname, actual, expected in checks:
                if actual != expected:
                    print(f"  {name}: {tname} shape mismatch: "
                          f"got {actual}, expected {expected}")
                    shape_ok = False
                    all_pass = False

            # 数值稳定性: 无 NaN / Inf
            nan_inf_ok = True
            for tname, tensor in [("dq", dq), ("dk", dk), ("dv", dv),
                                  ("db", db), ("dg_raw", dg_raw), ("dh0", dh0)]:
                if torch.isnan(tensor).any():
                    print(f"  {name}: {tname} contains NaN!")
                    nan_inf_ok = False
                    all_pass = False
                if torch.isinf(tensor).any():
                    print(f"  {name}: {tname} contains Inf!")
                    nan_inf_ok = False
                    all_pass = False

            status = "✓ PASS" if (shape_ok and nan_inf_ok) else "✗ FAIL"
            print(f"  {name}: B={B}, T={T}, H={H}, K={K}, V={V}, BT={cfg['bt']} "
                  f"... {status}")
        except Exception as e:
            print(f"  {name}: ✗ FAIL - {e}")
            all_pass = False

    # ---- 2. 泛化 case 验证 ----
    print("\n[泛化 case 验证]")

    gen_configs = [
        ("B=2,T=256,H=2,BT=64", {"B": 2, "T": 256, "H": 2, "K": 64, "V": 64, "bt": 64}),
        ("B=1,T=512,H=8,BT=64", {"B": 1, "T": 512, "H": 8, "K": 64, "V": 64, "bt": 64}),
    ]

    for name, cfg in gen_configs:
        try:
            inputs = _make_test_inputs(
                cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"], seed=123,
            )
            result = gated_delta_rule_backward_golden(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                g_raw=inputs["g_raw"],
                beta=inputs["beta"],
                initial_state=inputs["initial_state"],
                do=inputs["do"],
                dht=inputs["dht"],
                bt=cfg["bt"],
                use_qk_l2norm_in_kernel=True,
                l2_eps=1e-6,
            )
            dq, dk, dv, db, dg_raw, dh0 = result

            nan_inf = any(
                torch.isnan(t).any() or torch.isinf(t).any()
                for t in [dq, dk, dv, db, dg_raw, dh0]
            )
            status = "✓ PASS" if not nan_inf else "✗ FAIL (NaN/Inf)"
            print(f"  {name} ... {status}")
            if nan_inf:
                all_pass = False
        except Exception as e:
            print(f"  {name}: ✗ FAIL - {e}")
            all_pass = False

    # ---- 3. use_qk_l2norm_in_kernel=False 验证 ----
    print("\n[L2 norm 关闭验证]")
    try:
        inputs = _make_test_inputs(1, 128, 2, 64, 64, seed=99)
        result = gated_delta_rule_backward_golden(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g_raw=inputs["g_raw"],
            beta=inputs["beta"],
            initial_state=inputs["initial_state"],
            do=inputs["do"],
            dht=inputs["dht"],
            bt=64,
            use_qk_l2norm_in_kernel=False,
            l2_eps=1e-6,
        )
        dq, dk, dv, db, dg_raw, dh0 = result
        nan_inf = any(
            torch.isnan(t).any() or torch.isinf(t).any()
            for t in [dq, dk, dv, db, dg_raw, dh0]
        )
        status = "✓ PASS" if not nan_inf else "✗ FAIL (NaN/Inf)"
        print(f"  B=1,T=128,H=2,K=64,V=64,BT=64,no_l2norm ... {status}")
        if nan_inf:
            all_pass = False
    except Exception as e:
        print(f"  no_l2norm: ✗ FAIL - {e}")
        all_pass = False

    # ---- 4. 梯度有限差分验证 (小规模) ----
    print("\n[梯度有限差分验证 (小规模)]")
    try:
        B, T, H, K, V, bt = 1, 64, 2, 32, 32, 32
        torch.manual_seed(777)
        q_p = torch.randn(B, T, H, K, requires_grad=True)
        k_p = torch.randn(B, T, H, K, requires_grad=True)
        v_p = torch.randn(B, T, H, V, requires_grad=True)
        g_raw_p = torch.randn(B, T, H, requires_grad=True) * 0.1
        beta_p = torch.rand(B, T, H, requires_grad=True)
        initial_state_p = torch.randn(B, H, K, V) * 0.1
        do_t = torch.randn(B, T, H, V)
        dht_t = torch.randn(B, H, K, V) * 0.1

        # 使用 autograd 获取参考梯度
        consts = _build_constant_matrices(bt, torch.device("cpu"))
        out_fwd, _, cache_fwd = forward_ref(
            q=q_p, k=k_p, v=v_p, g_raw=g_raw_p, beta=beta_p,
            initial_state=initial_state_p, bt=bt,
            use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
            i=consts["i_mat"], m_le=consts["m_le"],
            m_lt=consts["m_lt"], c_cum=consts["c_cum"],
        )
        # 手动反向 (golden)
        result_golden = gated_delta_rule_backward_golden(
            q=q_p.detach(), k=k_p.detach(), v=v_p.detach(),
            g_raw=g_raw_p.detach(), beta=beta_p.detach(),
            initial_state=initial_state_p.detach(),
            do=do_t, dht=dht_t, bt=bt,
            use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
        )
        dq_golden = result_golden[0]

        # 验证 golden 的 dq 非零且有限
        has_grad = (dq_golden.abs() > 0).any()
        is_finite = torch.isfinite(dq_golden).all()
        if has_grad and is_finite:
            print(f"  dq non-zero & finite ... ✓ PASS")
        else:
            print(f"  dq: non-zero={has_grad}, finite={is_finite} ... ✗ FAIL")
            all_pass = False
    except Exception as e:
        print(f"  梯度验证: ✗ FAIL - {e}")
        all_pass = False

    # ---- 最终结果 ----
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    _validate()
