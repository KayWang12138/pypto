import pypto
import torch
from torch.fx.passes.infra.pass_base import PassResult
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
import time
import argparse
import sys


@pypto.jit
def pypto_lse_full(logits: pypto.Tensor, lse_out: pypto.Tensor, tilling: int=128):

    M = logits.shape[0]
    N = logits.shape[1]
    pypto.set_vec_tile_shapes(1, tilling)  # match the [1, N] row shape
    for i in pypto.loop(0, M, 1, name="row_loop"):
        # print(f"Processing row {i}")
        logit_row = pypto.view(logits, shape=[1, N], offsets=[i, 0])
        logit_fp32 = pypto.cast(logit_row, pypto.DT_FP32)
        
        m = pypto.amax(logit_fp32, dim=-1, keepdim=True)      # [1, 1]
        shifted = pypto.sub(logit_fp32, m)       # [1, N]
        exp_shifted = pypto.exp(shifted)         # [1, N]
        sum_exp = pypto.sum(exp_shifted, dim=-1, keepdim=True) # [1, 1]
        lse = pypto.add(m, pypto.log(sum_exp))     # [1, 1]

        lse_out[i:i+1] = pypto.reshape(lse, [1])

def test_pypto_lse_full():
    device = "npu:0"
    torch.npu.set_device(device)

    M, N = 4, 512
    # Use controlled logits to avoid extreme values
    logits = torch.randn(M, N, dtype=torch.float16, device=device)
    
    # Golden: use float32 for higher precision reference
    golden_lse = torch.logsumexp(logits.float(), dim=-1)  # [M], fp32

    # Output buffer
    lse_out = torch.empty(M, dtype=torch.float32, device=device)

    pto_logits = pypto.from_torch(logits, "logits")
    pto_lse_out = pypto.from_torch(lse_out, "lse_out")
    # Execute
    pypto_lse_full(pto_logits, pto_lse_out)

    # Compare
    print("pypto LSE   :", lse_out.cpu().numpy())
    print("Golden LSE:", golden_lse.cpu().numpy())

    assert_allclose(
        lse_out.cpu().numpy(),
        golden_lse.cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
        err_msg="LSE mismatch!"
    )
    print("✅ test_pypto_lse_full passed!")



@pypto.jit
def pypto_lse_tiled(logits: pypto.Tensor, lse_out: pypto.Tensor, tiling: int = 256):

    M = logits.shape[0]
    N = logits.shape[1]
    T = tiling
    num_tiles = (N + T - 1) // T
    assert N % T == 0, "N must be divisible by T"
    pypto.set_vec_tile_shapes(1, T)  # tile width = T
    for i in pypto.loop(0, M, 1, name="row_loop"):
        # 初始状态
        neg_inf = pypto.full([1, 1], torch.finfo(torch.float32).min, dtype=pypto.DT_FP32)
        zero = pypto.full([1, 1], 0.0, dtype=pypto.DT_FP32)

        m_state = neg_inf
        lse_state = zero
        def tile_body(k, m_old, lse_sum_old):
            start_n = k * T
            actual_chunk_size_int = min(N - start_n, T)
            logit_tile = pypto.view(logits, shape=[1, actual_chunk_size_int], offsets=[i, start_n])
            x_fp32 = pypto.cast(logit_tile, pypto.DT_FP32)
            m_tile = pypto.amax(x_fp32, dim=-1, keepdim=True)      # [1,1]
            m_new = pypto.maximum(m_old, m_tile)     # [1,1]

            exp_corr = pypto.exp(pypto.sub(m_old, m_new))  # [1,1]
            term1 = pypto.mul(lse_sum_old, exp_corr)     # [1,1]

            shifted_x = pypto.sub(x_fp32, m_new)         # [1,T]
            exp_shifted = pypto.exp(shifted_x)           # [1,T]
            sum_exp = pypto.sum(exp_shifted, dim=-1, keepdim=True)     # [1,1]
            lse_sum_new = pypto.add(term1, sum_exp)      # [1,1]
            m_state[:] = m_new
            lse_state[:] = lse_sum_new
        
        for k in range(num_tiles):
            tile_body(k, m_state, lse_state)

        final_lse = pypto.add(m_state, pypto.log(lse_state))
        lse_out[i:i+1] = pypto.reshape(final_lse, [1])


@pypto.jit
def pypto_lse_tiled_v1(logits: pypto.Tensor, lse_out: pypto.Tensor, tiling: int = 256):
    M = logits.shape[0]
    N = logits.shape[1]
    T = tiling
    num_tiles = (N + T - 1) // T
    assert N % T == 0, "N must be divisible by T"
    pypto.set_vec_tile_shapes(1, T)  # tile width = T

    for i in pypto.loop(0, M, 1, name="row_loop"):
        # 初始状态
        neg_inf = pypto.full([1, 1], torch.finfo(torch.float32).min, dtype=pypto.DT_FP32)
        zero = pypto.full([1, 1], 0.0, dtype=pypto.DT_FP32)

        m_state = neg_inf
        lse_state = zero
        def tile_body(k, m_old, lse_sum_old):
            start_n = k * T

            logit_tile = pypto.view(logits, shape=[1, T], offsets=[i, start_n])
            x_fp32 = pypto.cast(logit_tile, pypto.DT_FP32)
            m_tile = pypto.amax(x_fp32, dim=-1, keepdim=True)      # [1,1]
            m_new = pypto.maximum(m_old, m_tile)     # [1,1]

            exp_corr = pypto.exp(pypto.sub(m_old, m_new))  # [1,1]
            term1 = pypto.mul(lse_sum_old, exp_corr)     # [1,1]

            shifted_x = pypto.sub(x_fp32, m_new)         # [1,T]
            exp_shifted = pypto.exp(shifted_x)           # [1,T]
            sum_exp = pypto.sum(exp_shifted, dim=-1, keepdim=True)     # [1,1]
            lse_sum_new = pypto.add(term1, sum_exp)      # [1,1]
            m_state[:] = m_new
            lse_state[:] = lse_sum_new
        
        for k in range(num_tiles): 
            tile_body(k, m_state, lse_state)

        final_lse = pypto.add(m_state, pypto.log(lse_state))
        lse_out[i:i+1] = pypto.reshape(final_lse, [1])

                
def test_pypto_lse_tiled():
    device = "npu:0"
    torch.npu.set_device(device)

    M, N =64, 4096

    tiling = 128  # must divide N or handle remainder (our code does)

    logits = torch.randn(M, N, dtype=torch.float16, device=device)

    golden_lse = torch.logsumexp(logits.float(), dim=-1)  # [M], fp32
    # Tiled LSE
    lse_tiled = torch.empty(M, dtype=torch.float32, device=device)

    pto_logits = pypto.from_torch(logits, "logits")
    pto_lse_tiled = pypto.from_torch(lse_tiled, "lse_tiled")
    # Execute
    pypto_lse_tiled_v1(pto_logits, pto_lse_tiled, tiling=tiling)

    print("Full LSE  :", golden_lse.cpu().numpy())
    print("Tiled LSE :", lse_tiled.cpu().numpy())
    print("golden_lse shape:", golden_lse.shape)
    print("lse_out shape:", lse_tiled.shape)
    abs_lse_diff = torch.abs(lse_tiled - golden_lse)
    print(f"Max abs LSE diff: {abs_lse_diff.max().item():.6f}")
    print(f"Mean abs LSE diff: {abs_lse_diff.mean().item():.6f}")

    np.testing.assert_allclose(
        lse_tiled.cpu().numpy(),
        golden_lse.cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
        err_msg="Tiled LSE != Full LSE!"
    )
    print("✅ test_pypto_lse_tiled passed!")

@pypto.jit
def pypto_fused_ce_forward_v0_original(
    logits: pypto.Tensor, 
    logits_at_label: pypto.Tensor, 
    lse_out: pypto.Tensor, 
    loss_out: pypto.Tensor, 
    tiling: int = 128):

    M = logits.shape[0]
    N = logits.shape[1]
    T = tiling
    num_tiles = (N + T - 1) // T
    assert N % T == 0, "N must be divisible by T"
    pypto.set_vec_tile_shapes(1, T)
    for i in pypto.loop(0, M, 1, name="row_loop"):

        # 初始状态
        neg_inf = pypto.full([1, 1], torch.finfo(torch.float32).min, dtype=pypto.DT_FP32)
        zero = pypto.full([1, 1], 0.0, dtype=pypto.DT_FP32)

        m_state = neg_inf
        lse_state = zero

        def tile_body(k, m_old, lse_sum_old):
                start_n = k * T

                logit_tile = pypto.view(logits, shape=[1, T], offsets=[i, start_n])
                x_fp32 = pypto.cast(logit_tile, pypto.DT_FP32)
                m_tile = pypto.amax(x_fp32, dim=-1, keepdim=True)      # [1,1]
                m_new = pypto.maximum(m_old, m_tile)     # [1,1]

                exp_corr = pypto.exp(pypto.sub(m_old, m_new))  # [1,1]
                term1 = pypto.mul(lse_sum_old, exp_corr)     # [1,1]

                shifted_x = pypto.sub(x_fp32, m_new)         # [1,T]
                exp_shifted = pypto.exp(shifted_x)           # [1,T]
                sum_exp = pypto.sum(exp_shifted, dim=-1, keepdim=True)     # [1,1]
                lse_sum_new = pypto.add(term1, sum_exp)      # [1,1]
                m_state[:] = m_new
                lse_state[:] = lse_sum_new

        for k in range(num_tiles):
            tile_body(k, m_state, lse_state)

        final_lse = pypto.add(m_state, pypto.log(lse_state))
        lse_out[i:i+1] = pypto.reshape(final_lse, [1])


        target_logit = pypto.cast(logits_at_label[i:i+1, 0:1], pypto.DT_FP32) # [T_M, 1]
        loss_i = pypto.sub(final_lse, target_logit)  # [1, 1]
        loss_out[i:i+1] = pypto.reshape(loss_i, [1])


@pypto.jit(codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None,
        verify_options=None,
        debug_options=None)
def pypto_fused_ce_backward_v0_original(
    logits_backward: pypto.Tensor, 
    lse_backward: pypto.Tensor, 
    labels_backward: pypto.Tensor, 
    labels_idexes_backward: pypto.Tensor, 
    dlogits_backward: pypto.Tensor, 
    tiling: int = 128
    ) -> None:
    M_backward = logits_backward.shape[0]
    N_backward = logits_backward.shape[1]
    T_backward = tiling
    num_tiles_backward = (N_backward + T_backward - 1) // T_backward
    pypto.set_vec_tile_shapes(1, T_backward)
    for i_backward in pypto.loop(0, M_backward, 1, name="backward_token_loop"):
        
        lse_i_backward = pypto.reshape(lse_backward[i_backward:i_backward+1], [1, 1])      # [1,1], fp32

        label_id_backward = pypto.reshape(labels_backward[i_backward: i_backward+1], [1, 1])

        ones_backward = pypto.full([1, T_backward], 1.0, dtype=pypto.DT_FP32)
        
        def tile_body(k_backward):
            pypto.set_vec_tile_shapes(1, T_backward)
            start_n_backward = k_backward * T_backward

            logit_tile_backward = pypto.view(
                logits_backward, 
                shape=[1, T_backward], 
                offsets=[i_backward, start_n_backward]
                )
            mask_base_backward = pypto.view(labels_idexes_backward, shape=[1, T_backward], offsets=[0, 0])

            x_fp32_backward = pypto.cast(logit_tile_backward, pypto.DT_FP32)           # [1, T_backward]
            shifted_backward = pypto.sub(x_fp32_backward, lse_i_backward)                     # [1, T_backward]
            probs_backward = pypto.exp(shifted_backward)   
                                        # [1, T_backward]
            mask_abs_fp32_backward = pypto.cast(pypto.sub(pypto.add(mask_base_backward, start_n_backward), label_id_backward), pypto.DT_FP32)    # [1, T_backward]

            mask_abs_backward = pypto.abs(mask_abs_fp32_backward)
            
            max_value_backward = pypto.view(ones_backward, shape=[1, T_backward], offsets=[0, 0])
            
            mask_abs_clip_backward = pypto.minimum(mask_abs_backward, max_value_backward)
            
            adjusted_probs_backward = pypto.add(
                pypto.mul(pypto.add(pypto.neg(mask_abs_clip_backward), 1.0), pypto.sub(probs_backward, 1.0)),
                pypto.mul(probs_backward, mask_abs_clip_backward)
            ) 
            
            grad_tile_fp16_backward = pypto.cast(adjusted_probs_backward, pypto.DT_FP16)   # [1, N]

            dlogits_backward[i_backward:i_backward+1, start_n_backward:start_n_backward+T_backward] = grad_tile_fp16_backward

        for k_backward in range(num_tiles_backward):
            tile_body(k_backward)


def test_pypto_ce():
    device = "npu:2"
    torch.npu.set_device(device)

    tiling = 128
    M, N = 64, 4096
    
    # tiling_n = 1024

    for i in range(5):
        print("================================ iter {} start ================================".format(i))

        torch.npu.synchronize()


        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)

        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        logits_at_label = logits[torch.arange(M, device=device), labels].unsqueeze(-1)
        golden_loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='none')  # [M]
        golden_lse = torch.logsumexp(logits.float(), dim=-1)  # [M], fp32
        # pypto
        lse_tiled = torch.zeros(M, dtype=torch.float32, device=device)
        loss_out = torch.zeros(M, dtype=torch.float32, device=device)
        

        pto_logits = pypto.from_torch(logits, "logits")
        pto_logits_at_label = pypto.from_torch(logits_at_label, "logits_at_label")

        pto_lse_tiled = pypto.from_torch(lse_tiled, "lse_tiled")
        pto_loss_out = pypto.from_torch(loss_out, "loss_out")
        # Execute
        pypto_fused_ce_forward_v0_original(pto_logits, pto_logits_at_label, pto_lse_tiled, pto_loss_out, tiling=tiling)
        
        print("golden_lse shape:", golden_lse.shape)
        print("lse_tiled shape:", lse_tiled.shape)
        abs_lse_diff = torch.abs(lse_tiled - golden_lse)
        print(f"Max abs LSE diff: {abs_lse_diff.max().item():.6f}")
        print(f"Mean abs LSE diff: {abs_lse_diff.mean().item():.6f}")

        print("Golden CE :", golden_loss.detach().cpu().numpy())
        # print("pto CE  :", loss_tiled.detach().cpu().numpy())
        print("pto CE  :", loss_out.detach().cpu().numpy())

        # abs_diff = torch.abs(loss_tiled - golden_loss)
        abs_diff = torch.abs(loss_out - golden_loss)
        print(f"Max abs diff: {abs_diff.max().item():.6f}")
        print(f"Mean abs diff: {abs_diff.mean().item():.6f}")

        np.testing.assert_allclose(
            loss_out.detach().cpu().numpy(),
            golden_loss.detach().cpu().numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused CE mismatch!"
        )
        print(f"✅ iter: {i} test_pto_fused_ce forward passed!")

        # backward

        grad_logits = torch.zeros_like(logits, device=logits.device) # fp32
        labels_idexes = torch.arange(tiling, dtype=torch.int32, device=logits.device).reshape(1, tiling)
         
        pto_logits = pypto.from_torch(logits, "logits")
        pto_lse_tiled = pypto.from_torch(lse_tiled, "lse_tiled")
        pto_labels = pypto.from_torch(labels, "labels")
        pto_labels_idexes = pypto.from_torch(labels_idexes, "labels_idexes")
        pto_grad_logits = pypto.from_torch(grad_logits, "grad_logits")
        # Execute
        pypto_fused_ce_backward_v0_original(pto_logits, pto_lse_tiled, pto_labels, pto_labels_idexes, pto_grad_logits, tiling=tiling)

        grad_logits_golden = torch.autograd.grad(golden_loss.sum(), logits)[0]

        print("Golden grad :", grad_logits_golden.cpu().numpy())
        print("pypto grad    :", grad_logits.cpu().numpy())
        print("\n✅ Backward gradient comparison:")
        abs_diff = torch.abs(grad_logits - grad_logits_golden)
        print(f"Max abs diff: {abs_diff.max().item():.6f}")
        print(f"Mean abs diff: {abs_diff.mean().item():.6f}")

        np.testing.assert_allclose(
            grad_logits.cpu().numpy(),
            grad_logits_golden.cpu().numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Backward gradient mismatch!"
        )

        print(f"🎉 iter: {i} test_pto_fused_ce backward PASSED!")


@pypto.jit(
    # codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_fused_ce_forward_v1_original(
    logits: pypto.Tensor,
    logits_at_label: pypto.Tensor,
    lse_out: pypto.Tensor,
    loss_out: pypto.Tensor,
    tiling_m: int = 16,
    tiling_n: int = 1024,
    task_id: int = 0
) -> None:
    M = logits.shape[0]
    N = logits.shape[1]
    T_M = tiling_m
    T_N = tiling_n
    num_tiles = (N + T_N - 1) // T_N
    assert M % T_M == 0, "M must be divisible by T_M"
    assert N % T_N == 0, "N must be divisible by T_N"
    pypto.set_vec_tile_shapes(T_M, T_N)

    for i in pypto.loop(0, M, T_M, name="row_loop_{}".format(task_id),
        unroll_list=[48]
        ):
        # 初始状态
        def token_body(i):
            neg_inf = pypto.full([T_M, 1], torch.finfo(torch.float32).min, dtype=pypto.DT_FP32)
            zero = pypto.full([T_M, 1], 0.0, dtype=pypto.DT_FP32)

            m_state = neg_inf
            lse_state = zero

            def tile_body(k, m_old, lse_sum_old):
                start_n = k * T_N
                
                logit_tile = pypto.cast(pypto.view(logits, shape=[T_M, T_N], offsets=[i, start_n]), pypto.DT_FP32)
                m_new = pypto.maximum(m_old, pypto.amax(logit_tile, dim=-1, keepdim=True))     # [1]
                lse_sum_new = pypto.add(
                    pypto.mul(lse_sum_old, pypto.exp(pypto.sub(m_old, m_new))),
                    pypto.sum(pypto.exp(pypto.sub(logit_tile, m_new)), dim=-1, keepdim=True)
                )      # [1]
                m_state[:] = m_new
                lse_state[:] = lse_sum_new
            
            for k in range(num_tiles):  
                tile_body(k, m_state, lse_state)

            final_lse = pypto.add(m_state, pypto.log(lse_state))     # [T_M, 1]

            lse_out[i:i+T_M] = pypto.reshape(final_lse, [T_M])


            target_logit = pypto.cast(logits_at_label[i:i+T_M, 0:1], pypto.DT_FP32) # [T_M, 1]
            
            loss_i = pypto.sub(final_lse, target_logit)  # [T_M, 1]

            loss_out[i:i+T_M] = pypto.reshape(loss_i, [T_M])

        token_body(i)

@pypto.jit(
    # codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_fused_ce_backward_v1_original(
    logits_backward: pypto.Tensor, 
    lse_backward: pypto.Tensor, 
    labels_backward: pypto.Tensor, 
    labels_idexes_backward: pypto.Tensor, 
    ones: pypto.Tensor,
    dlogits_backward: pypto.Tensor, 
    tiling_m: int = 16,
    tiling_n: int = 1024
) -> None:

    M_backward = logits_backward.shape[0]
    N_backward = logits_backward.shape[1]
    T_backward_m = tiling_m
    T_backward_n = tiling_n
    num_tiles_backward_n = (N_backward + T_backward_n - 1) // T_backward_n
    pypto.set_vec_tile_shapes(T_backward_m, T_backward_n)

    for i_backward in pypto.loop(0, M_backward, T_backward_m, name="backward_token_loop", unroll_list=[48]):

        def tile_body(k_backward):
            start_n_backward = k_backward * T_backward_n
            logit_tile_backward = pypto.view(
                logits_backward, 
                shape=[T_backward_m, T_backward_n], 
                offsets=[i_backward, start_n_backward],
                )
            mask_base_backward = pypto.view(labels_idexes_backward, shape=[1, T_backward_n], offsets=[0, 0])
            lse_i_backward = pypto.view(lse_backward, shape=[T_backward_m, 1], offsets=[i_backward, 0])     # [1,1], fp32
            label_id_backward = pypto.view(labels_backward, shape=[T_backward_m, 1], offsets=[i_backward, 0])

            x_fp32_backward = pypto.cast(logit_tile_backward, pypto.DT_FP32)           # [1, valid_len_backward]
            shifted_backward = pypto.sub(x_fp32_backward, lse_i_backward)                     # [1, valid_len_backward]
            probs_backward = pypto.exp(shifted_backward)   
                                        # [1, valid_len_backward]
            mask_abs_fp32_backward = pypto.cast(pypto.sub(pypto.add(mask_base_backward, start_n_backward), label_id_backward), pypto.DT_FP32)    # [1, valid_len_backward]
            

            mask_abs_backward = pypto.abs(mask_abs_fp32_backward)
            

            mask_abs_clip_backward = pypto.minimum(ones, mask_abs_backward)
            
            adjusted_probs_backward = pypto.add(
                pypto.mul(pypto.add(pypto.neg(mask_abs_clip_backward), 1.0), pypto.sub(probs_backward, 1.0)),
                pypto.mul(probs_backward, mask_abs_clip_backward)
            ) 
            
            grad_tile_fp16_backward = pypto.cast(adjusted_probs_backward, pypto.DT_FP16)   # [1, N]

            dlogits_backward[i_backward:i_backward+T_backward_m, start_n_backward:start_n_backward+T_backward_n] = grad_tile_fp16_backward

        for k_backward in range(num_tiles_backward_n):
            tile_body(k_backward)


def test_pypto_ce_mn():
    device = "npu:2"
    torch.npu.set_device(device)

    tiling_m = 4
    tiling_n = 1024
    M, N = 128, 4096
    
    # tiling_n = 1024

    for i in range(5):
        print("================================ iter {} start ================================".format(i))

        torch.npu.synchronize()

        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)
        print("logits shape:", logits.shape)
        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        logits_at_label = logits[torch.arange(M, device=device), labels].unsqueeze(-1)
        golden_loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='none')  # [M]
        golden_lse = torch.logsumexp(logits.float(), dim=-1)  # [M], fp32
        # pypto
        lse_tiled = torch.zeros(M, dtype=torch.float32, device=device)
        loss_out = torch.zeros(M, dtype=torch.float32, device=device)
        
        pto_logits = pypto.from_torch(logits, "logits")
        pto_logits_at_label = pypto.from_torch(logits_at_label, "logits_at_label")

        pto_lse_tiled = pypto.from_torch(lse_tiled, "lse_tiled")
        pto_loss_out = pypto.from_torch(loss_out, "loss_out")
        # Execute
        pypto_fused_ce_forward_v1_original(pto_logits, pto_logits_at_label, pto_lse_tiled, pto_loss_out, tiling_m=tiling_m, tiling_n=tiling_n)
        
        print("golden_lse shape:", golden_lse.shape)
        print("lse_tiled shape:", lse_tiled.shape)
        abs_lse_diff = torch.abs(lse_tiled - golden_lse)
        print(f"Max abs LSE diff: {abs_lse_diff.max().item():.6f}")
        print(f"Mean abs LSE diff: {abs_lse_diff.mean().item():.6f}")

        print("Golden CE :", golden_loss.detach().cpu().numpy())
        print("pto CE  :", loss_out.detach().cpu().numpy())

        abs_diff = torch.abs(loss_out - golden_loss)
        print(f"Max abs diff: {abs_diff.max().item():.6f}")
        print(f"Mean abs diff: {abs_diff.mean().item():.6f}")

        np.testing.assert_allclose(
            loss_out.detach().cpu().numpy(),
            golden_loss.detach().cpu().numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused CE mismatch!"
        )
        print(f"✅ iter: {i} test_pto_fused_ce forward passed!")

        # backward

        grad_logits = torch.zeros_like(logits, device=logits.device) # fp32
        labels_idexes = torch.arange(tiling_n, dtype=torch.int32, device=logits.device).reshape(1, tiling_n)
        ones = torch.ones((tiling_m, tiling_n), dtype=torch.float32, device=logits.device)

        pto_logits = pypto.from_torch(logits, "logits")
        pto_lse_tiled = pypto.from_torch(lse_tiled.view(-1, 1), "lse_tiled")
        pto_labels = pypto.from_torch(labels.view(-1, 1), "labels")
        pto_labels_idexes = pypto.from_torch(labels_idexes, "labels_idexes")
        pto_ones = pypto.from_torch(ones, "ones")
        pto_grad_logits = pypto.from_torch(grad_logits, "grad_logits")
        # Execute
        pypto_fused_ce_backward_v1_original(pto_logits, pto_lse_tiled, pto_labels, pto_labels_idexes, pto_ones, pto_grad_logits, tiling_m=tiling_m, tiling_n=tiling_n)

        grad_logits_golden = torch.autograd.grad(golden_loss.sum(), logits)[0]

        print("Golden grad :", grad_logits_golden.cpu().numpy())
        print("pypto grad    :", grad_logits.cpu().numpy())
        print("\n✅ Backward gradient comparison:")
        abs_diff = torch.abs(grad_logits - grad_logits_golden)
        print(f"Max abs diff: {abs_diff.max().item():.6f}")
        print(f"Mean abs diff: {abs_diff.mean().item():.6f}")

        np.testing.assert_allclose(
            grad_logits.cpu().numpy(),
            grad_logits_golden.cpu().numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Backward gradient mismatch!"
        )

        print(f"🎉 iter: {i} test_pto_fused_ce backward PASSED!")


class FusedCrossEntropyPypto_original_v0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, tiling_n=256):
        """
        logits: [M, N], fp16
        labels: [M], int32
        """
        M, _ = logits.shape
        device = logits.device

        logits_at_label = logits[torch.arange(M, device=device), labels].unsqueeze(-1)
        lse_tiled = torch.zeros(M, dtype=torch.float32, device=device)
        loss_out = torch.zeros(M, dtype=torch.float32, device=device)
        
        # inputs = [logits, labels]
        pto_logits = pypto.from_torch(logits, "logits")
        pto_logits_at_label = pypto.from_torch(logits_at_label, "logits_at_label")
        # outputs = [loss_tiled, lse_out]
        pto_lse_tiled = pypto.from_torch(lse_tiled, "lse_tiled")
        pto_loss_out = pypto.from_torch(loss_out, "loss_out")

        pypto_fused_ce_forward_v0_original(pto_logits, pto_logits_at_label, pto_lse_tiled, pto_loss_out, tiling=tiling_n)

        ctx.save_for_backward(logits, lse_tiled, labels)
        ctx.tiling_n = tiling_n

        return loss_out

    @staticmethod
    def backward(ctx, grad_output):
        logits, lse, labels = ctx.saved_tensors
        tiling_n = ctx.tiling_n
        grad_logits = torch.zeros_like(logits, device=logits.device) # fp32
        labels_idexes = torch.arange(tiling_n, dtype=torch.int32, device=logits.device).reshape(1, tiling_n)
        # 在 backward 前
        pto_logits = pypto.from_torch(logits, "logits")
        pto_lse = pypto.from_torch(lse, "lse")
        pto_labels = pypto.from_torch(labels, "labels")
        pto_labels_idexes = pypto.from_torch(labels_idexes, "labels_idexes")
        pto_grad_logits = pypto.from_torch(grad_logits, "grad_logits")
        
        # Execute
        pypto_fused_ce_backward_v0_original(pto_logits, pto_lse, pto_labels, pto_labels_idexes, pto_grad_logits, tiling=tiling_n)
        return grad_logits, None, None, None
    

def test_pypto_ce_v0_whole():
        device = "npu:2"
        torch.npu.set_device(device)

        # 配置参数（可调整）
        M, N = 128, 4096
        tiling_n = 1024

        torch.manual_seed(42)

        for i in range(5):
    
            print(f"================================ iter {i} start ================================")
            torch.npu.synchronize()
            
            # === Step 1: 创建两组相同的输入 ===
            logits_np = torch.randn(M, N, dtype=torch.float16).numpy()
            labels_np = torch.randint(0, N, (M,), dtype=torch.int32).numpy()
            # labels_np = torch.randint(0, N, (M,), dtype=torch.int64).numpy()
            # --- PyTorch 路径 ---
            logits_torch = torch.from_numpy(logits_np).to(device).requires_grad_(True)
            labels_torch = torch.from_numpy(labels_np).to(device).long()  # PyTorch CE expects long

            # Forward
            loss_torch = torch.nn.functional.cross_entropy(logits_torch.float(), labels_torch, reduction='none')  # [M]
            # Backward
            grad_logits_torch = torch.autograd.grad(loss_torch.sum(), logits_torch)[0]  # [M, N], fp32
            torch.npu.synchronize()
            
            # --- pypto 路径 ---
            logits_pypto = torch.from_numpy(logits_np).to(device).requires_grad_(True)

            labels_pypto = torch.from_numpy(labels_np).to(device)
            
            loss_pypto = FusedCrossEntropyPypto_original_v0.apply(logits_pypto, labels_pypto, tiling_n)
            
            # Backward
            grad_logits_pypto = torch.autograd.grad(loss_pypto.sum(), logits_pypto)[0]
            # === Step 2: 同步设备 ===
            
            torch.npu.synchronize()

            # === Step 3: 数值对比 ===
            print("\n🔍 Loss comparison:")
            loss_diff = torch.abs(loss_pypto - loss_torch)
            print(f"  Max abs loss diff: {loss_diff.max().item():.6f}")
            print(f"  Mean abs loss diff: {loss_diff.mean().item():.6f}")

            np.testing.assert_allclose(
                loss_pypto.detach().cpu().numpy(),
                loss_torch.detach().cpu().numpy(),
                atol=1e-2,
                rtol=1e-2,
                err_msg=f"Iter {i}: Loss mismatch!"
            )
            print("✅ Loss matches!")

            print("\n🔍 Gradient comparison:")
            grad_diff = torch.abs(grad_logits_pypto - grad_logits_torch)
            print(f"  Max abs grad diff: {grad_diff.max().item():.6f}")
            print(f"  Mean abs grad diff: {grad_diff.mean().item():.6f}")

            np.testing.assert_allclose(
                grad_logits_pypto.detach().cpu().numpy(),
                grad_logits_torch.detach().cpu().numpy(),
                atol=1e-2,
                rtol=1e-2,
                err_msg=f"Iter {i}: Gradient mismatch!"
            )
            print("✅ Gradient matches!")

            print(f"🎉 Iter {i} PASSED!\n")


def test_pypto_ce_v1_whole():
        device = "npu:2"
        torch.npu.set_device(device)

        # 配置参数（可调整）
        M, N = 8192, 152576
        tiling_m = 4
        tiling_n = 1024

        torch.manual_seed(42)

        for i in range(10):
    
            print(f"================================ iter {i} start ================================")
            torch.npu.synchronize()
            
            # === Step 1: 创建两组相同的输入 ===
            logits_np = torch.randn(M, N, dtype=torch.float16).numpy()
            labels_np = torch.randint(0, N, (M,), dtype=torch.int32).numpy()
            # labels_np = torch.randint(0, N, (M,), dtype=torch.int64).numpy()
            # --- PyTorch 路径 ---
            logits_torch = torch.from_numpy(logits_np).to(device).requires_grad_(True)
            labels_torch = torch.from_numpy(labels_np).to(device).long()  # PyTorch CE expects long

            # Forward 计时
            torch.npu.synchronize()
            torch_forward_start = time.perf_counter()
            loss_torch = torch.nn.functional.cross_entropy(logits_torch.float(), labels_torch, reduction='none')  # [M]
            torch.npu.synchronize()
            torch_forward_end = time.perf_counter()
            torch_forward_time = torch_forward_end - torch_forward_start
            
            # Backward 计时
            torch.npu.synchronize()
            torch_backward_start = time.perf_counter()
            grad_logits_torch = torch.autograd.grad(loss_torch.sum(), logits_torch)[0]  # [M, N], fp32
            torch.npu.synchronize()
            torch_backward_end = time.perf_counter()
            torch_backward_time = torch_backward_end - torch_backward_start
            
            torch_total_time = torch_forward_time + torch_backward_time
            
            # --- pypto 路径 ---
            logits_pypto = torch.from_numpy(logits_np).to(device).requires_grad_(True)
            labels_pypto = torch.from_numpy(labels_np).to(device)
            # Forward 计时
            torch.npu.synchronize()
            pypto_forward_start = time.perf_counter()
            loss_pypto = FusedCrossEntropyPypto_new_v1.apply(logits_pypto, labels_pypto, tiling_m, tiling_n)
            torch.npu.synchronize()
            pypto_forward_end = time.perf_counter()
            pypto_forward_time = pypto_forward_end - pypto_forward_start

            # Backward 计时
            torch.npu.synchronize()
            pypto_backward_start = time.perf_counter()
            grad_logits_pypto = torch.autograd.grad(loss_pypto.sum(), logits_pypto)[0]
            torch.npu.synchronize()
            pypto_backward_end = time.perf_counter()
            pypto_backward_time = pypto_backward_end - pypto_backward_start
            
            pypto_total_time = pypto_forward_time + pypto_backward_time
            # === Step 2: 同步设备 ===

            # === Step 3: 数值对比 ===
            print("\n🔍 Loss comparison:")
            loss_diff = torch.abs(loss_pypto - loss_torch)
            print(f"  Max abs loss diff: {loss_diff.max().item():.6f}")
            print(f"  Mean abs loss diff: {loss_diff.mean().item():.6f}")

            print("✅ Loss matches!")

            print("\n🔍 Gradient comparison:")
            grad_diff = torch.abs(grad_logits_pypto - grad_logits_torch)
            print(f"  Max abs grad diff: {grad_diff.max().item():.6f}")
            print(f"  Mean abs grad diff: {grad_diff.mean().item():.6f}")

            print("✅ Gradient matches!")

            # === Step 4: 性能对比 ===
            print("\n⏱️  Performance comparison:")
            print(f"  PyTorch Forward:  {torch_forward_time*1000:.3f} ms")
            print(f"  PyTorch Backward: {torch_backward_time*1000:.3f} ms")
            print(f"  PyTorch Total:    {torch_total_time*1000:.3f} ms")
            print(f"  PyPTO Forward:    {pypto_forward_time*1000:.3f} ms")
            print(f"  PyPTO Backward:   {pypto_backward_time*1000:.3f} ms")
            print(f"  PyPTO Total:      {pypto_total_time*1000:.3f} ms")
            
            speedup_forward = torch_forward_time / pypto_forward_time if pypto_forward_time > 0 else 0
            speedup_backward = torch_backward_time / pypto_backward_time if pypto_backward_time > 0 else 0
            speedup_total = torch_total_time / pypto_total_time if pypto_total_time > 0 else 0
            
            print(f"\n  Speedup:")
            print(f"    Forward:  {speedup_forward:.3f}x ({'faster' if speedup_forward > 1 else 'slower'})")
            print(f"    Backward: {speedup_backward:.3f}x ({'faster' if speedup_backward > 1 else 'slower'})")
            print(f"    Total:    {speedup_total:.3f}x ({'faster' if speedup_total > 1 else 'slower'})")

            print(f"🎉 Iter {i} PASSED!\n")


class FusedCrossEntropyPypto_new_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, tiling_m=16, tiling_n=1024):
        """
        logits: [M, N], fp16
        labels: [M], int32
        """
        M, _ = logits.shape
        device = logits.device
        labels = labels.to(torch.int32).contiguous()

        loss = torch.empty(M, dtype=torch.float32, device=device)
        lse = torch.empty(M, dtype=torch.float32, device=device)

        logits_at_label = logits[torch.arange(M, device=device), labels].view(M, 1)

        pto_logits = pypto.from_torch(logits, "logits")
        pto_logits_at_label = pypto.from_torch(logits_at_label, "logits_at_label")

        pto_lse = pypto.from_torch(lse, "lse")
        pto_loss_out = pypto.from_torch(loss, "loss_out")
        # Execute
        pypto_fused_ce_forward_v1_original(pto_logits, pto_logits_at_label, pto_lse, pto_loss_out, tiling_m=tiling_m, tiling_n=tiling_n)

        ctx.save_for_backward(logits, lse, labels)
        ctx.tiling_m = tiling_m
        ctx.tiling_n = tiling_n

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, lse, labels = ctx.saved_tensors
        tiling_m = ctx.tiling_m
        tiling_n = ctx.tiling_n
        grad_logits = torch.zeros_like(logits, memory_format=torch.contiguous_format, device=logits.device) # fp32
        labels_idexes = torch.arange(tiling_n, dtype=torch.int32, device=logits.device).reshape(1, tiling_n)
        # labels_idexes = torch.arange(tiling_n, dtype=torch.int32, device=logits.device).view(1, tiling_n)
        
        ones = torch.ones((tiling_m, tiling_n), dtype=torch.float32, device=logits.device)
        pto_logits = pypto.from_torch(logits, "logits")
        pto_lse = pypto.from_torch(lse.view(-1, 1), "lse")
        pto_labels = pypto.from_torch(labels.view(-1, 1), "labels")
        pto_labels_idexes = pypto.from_torch(labels_idexes, "labels_idexes")
        pto_ones = pypto.from_torch(ones, "ones")
        pto_grad_logits = pypto.from_torch(grad_logits, "grad_logits")
        # Execute
        pypto_fused_ce_backward_v1_original(pto_logits, pto_lse, pto_labels, pto_labels_idexes, pto_ones, pto_grad_logits, tiling_m=tiling_m, tiling_n=tiling_n)

        return grad_logits, None, None, None


def fast_pypto_fused_cross_entropy_loss(logits, labels, tiling_m=16, tiling_n=1024, version=0):
    """
    Unified interface for different versions of FusedCrossEntropyPypto.
    
    Args:
        logits: [M, N], fp16
        labels: [M], int32
        tiling_m: used only in version 2
        tiling_n: used in all versions (as main tiling)
        version: 0, 1, or 2
    """
    if version == 0:
        return FusedCrossEntropyPypto_original_v0.apply(logits, labels, tiling_n)
    elif version == 1:
        return FusedCrossEntropyPypto_new_v1.apply(logits, labels, tiling_m, tiling_n)
    else:
        raise ValueError(f"Unsupported version: {version}")


def reset_peak_memory(device):
    torch.npu.reset_peak_memory_stats(device)


def get_peak_memory(device):
    return torch.npu.max_memory_allocated(device) / (1024 ** 2)

def benchmark_torch_forward_backward(logits, labels, warmup=5, total_iter=10):
    device = logits.device
    fw_times = []
    bw_times = []
    reset_peak_memory(device)
    for i in range(total_iter):
        torch.npu.synchronize()
        start = time.perf_counter()
        torch.npu.synchronize()
        
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

        torch.npu.synchronize()
        mid = time.perf_counter()
        grad_logits = torch.autograd.grad(loss.sum(), logits)[0]
        torch.npu.synchronize()

        end = time.perf_counter()
        if i >= warmup:
            fw_times.append(mid - start)
            bw_times.append(end - mid)
        if i == total_iter - 1:
            loss_result = loss
    mem = get_peak_memory(logits.device)
    print("dtype of grad_logits:", grad_logits.dtype)
    return np.mean(fw_times) * 1000, np.mean(bw_times) * 1000, mem, loss_result, grad_logits


def benchmark_torch_only(total_tokens, vocab_size, device="npu:7", warmup=5, total_iter=10):
    torch.npu.set_device(device)
    torch.manual_seed(42)

    M, N = total_tokens, vocab_size
    fw_times = []
    bw_times = []
    mems = []

    # 先做 warmup 轮（不计入统计）
    for _ in range(warmup):
        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)
        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        fw, bw, mem, _, _ = benchmark_torch_forward_backward(logits, labels, warmup=0, total_iter=1)

    # 正式测量
    for i in range(total_iter):
        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)
        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        fw, bw, mem, _, _ = benchmark_torch_forward_backward(logits, labels, warmup=0, total_iter=1)
        fw_times.append(fw)
        bw_times.append(bw)
        mems.append(mem)

    return {
        "fw_mean": np.mean(fw_times),
        "bw_mean": np.mean(bw_times),
        "total_mean": np.mean(fw_times) + np.mean(bw_times),
        "mem_mean": np.mean(mems)
    }


def benchmark_pypto_forward_backward(logits, labels, tiling, version=0,warmup=5, total_iter=10):
    fw_times = []
    bw_times = []
    print("pypto forward-backward benchmark start, logits shape:", logits.shape, "labels shape:", labels.shape, "tiling:", tiling)
    reset_peak_memory(logits.device)
    for i in range(total_iter):
        torch.npu.synchronize()
        start = time.perf_counter()
        torch.npu.synchronize()

        # forward
        loss = fast_pypto_fused_cross_entropy_loss(
            logits, labels, tiling_m=16, tiling_n=tiling, version=version
        )
        torch.npu.synchronize()

        # backward
        mid = time.perf_counter()
        
        grad_logits = torch.autograd.grad(loss.sum(), logits)[0]
        torch.npu.synchronize()

        end = time.perf_counter()
        if i >= warmup:
            fw_times.append(mid - start)
            bw_times.append(end - mid)

        if i == total_iter - 1:
            loss_result = loss
    mem = get_peak_memory(logits.device)
    return np.mean(fw_times) * 1000, np.mean(bw_times) * 1000, mem, loss_result, grad_logits


def benchmark_pypto_only(total_tokens, vocab_size, tiling, device="npu:7", warmup=5, total_iter=10, version=0):
    torch.npu.set_device(device)
    pypto.runtime._device_init()
    torch.manual_seed(42)

    M, N = total_tokens, vocab_size
    fw_times = []
    bw_times = []
    mems = []

    # 先做 warmup 轮
    for _ in range(warmup):
        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)
        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        fw, bw, mem, _, _ = benchmark_pypto_forward_backward(logits, labels, tiling=tiling, warmup=0, total_iter=1, version=version)

    # 正式测量
    for i in range(total_iter):
        logits = torch.randn(M, N, dtype=torch.float16, device=device, requires_grad=True)
        labels = torch.randint(0, N, (M,), dtype=torch.int32, device=device)
        fw, bw, mem, _, _ = benchmark_pypto_forward_backward(logits, labels, tiling=tiling, warmup=0, total_iter=1, version=version)
        fw_times.append(fw)
        bw_times.append(bw)
        mems.append(mem)
    pypto.runtime._device_fini()
    return {
        "fw_mean": np.mean(fw_times),
        "bw_mean": np.mean(bw_times),
        "total_mean": np.mean(fw_times) + np.mean(bw_times),
        "mem_mean": np.mean(mems)
    }


def test_run_separate_benchmarks(total_tokens=12288, vocab_size=152576, tiling_options=[1024], device="npu:5", warmup=5, total_iter=10, version=1):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting SEPARATE benchmarks for total_tokens={total_tokens}, vocab_size={vocab_size}, device={device}, fused_ce_version={version}")

    # Step 1: Benchmark PyTorch
    print("\n=== 🔥 Benchmarking PyTorch (forward + backward) ===")
    torch_res = benchmark_torch_only(
        total_tokens=total_tokens,
        vocab_size=vocab_size,
        device=device,
        warmup=warmup,
        total_iter=total_iter
    )
    print(f"PyTorch Results:")
    print(f"  Forward:  {torch_res['fw_mean']:.3f} ms")
    print(f"  Backward: {torch_res['bw_mean']:.3f} ms")
    print(f"  Total:    {torch_res['total_mean']:.3f} ms")
    print(f"  Peak Mem: {torch_res['mem_mean']:.1f} MB")

    # Step 2: Benchmark pypto for each tiling
    for tiling in tiling_options:
        print(f"\n=== 🚀 Benchmarking pypto (tiling={tiling}) ===")
        pypto_res = benchmark_pypto_only(
            total_tokens=total_tokens,
            vocab_size=vocab_size,
            tiling=tiling,
            device=device,
            warmup=warmup,
            total_iter=total_iter,
            version=version
        )
        print("=" * 60)
        print(f"pytorch Results:")
        print(f"  Forward:  {torch_res['fw_mean']:.3f} ms")
        print(f"  Backward: {torch_res['bw_mean']:.3f} ms")
        print(f"  Total:    {torch_res['total_mean']:.3f} ms")
        print(f"  Peak Mem: {torch_res['mem_mean']:.1f} MB")
        print("=" * 60)
        print(f"pypto Results (tiling={tiling}):")
        print(f"  Forward:  {pypto_res['fw_mean']:.3f} ms")
        print(f"  Backward: {pypto_res['bw_mean']:.3f} ms")
        print(f"  Total:    {pypto_res['total_mean']:.3f} ms")
        print(f"  Peak Mem: {pypto_res['mem_mean']:.1f} MB")

        # Step 3: Compute ratios (pypto vs torch)
        speed_ratio = torch_res["total_mean"] / pypto_res["total_mean"]
        mem_ratio = pypto_res["mem_mean"] / torch_res["mem_mean"]

        print(f"\n📊 Comparison (pypto vs PyTorch, tiling={tiling}):")
        print(f"  Speed Ratio (torch_time / pypto_time): {speed_ratio:.3f}x {'(faster)' if speed_ratio > 1 else '(slower)'}")
        print(f"  Memory Ratio (pypto_mem / torch_mem): {mem_ratio:.1%} ({'better' if mem_ratio < 1 else 'worse'})")

        print("-" * 60)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All benchmarks completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", type=int, default=0, help="NPU device number (e.g., 5 for 'npu:5')")
    parser.add_argument("--version", type=int, choices=[0, 1], default=0, help="Fused CE version: 0 or 1")
    args = parser.parse_args()

    device = f"npu:{args.npu}"  
    vocab_size = 152576
    total_tokens_list = [12288]
    tiling_options = [1024]

    for total_tokens in total_tokens_list:
        test_run_separate_benchmarks(
            total_tokens=total_tokens,
            vocab_size=vocab_size,
            tiling_options=tiling_options,
            device=device,
            warmup=5,
            total_iter=10,
            version=args.version
        )