import os
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto

def moe_gating_topk_core(            
    x,
    bias,
    k,
    k_group,
    group_count,
    group_select_mode,
    renorm,
    norm_type,
    out_flag,
    routed_scaling_factor,
    eps) -> (pypto.Tensor, pypto.Tensor, pypto.Tensor):
 
        bs = x.shape[0]
        num_experts = x.shape[1]
        x = pypto.cast(x, pypto.DT_FP32)
        
        # Step 1: Apply normalization (sigmoid or softmax)
        if norm_type == 1:
            norm_out = pypto.sigmoid(x) 
        else:
            norm_out = pypto.softmax(x, -1) 
        original_norm_out =  norm_out

        # Step 2: Add bias if provided
        if bias is not None:    
            norm_out = pypto.add(norm_out, bias)

        # Step 3: Group selection
        if group_count > 1:
            group_unit = num_experts // group_count
            group = pypto.reshape(norm_out, [bs, group_count, group_unit])  
            pypto.set_vec_tile_shapes(bs, group_count, group_unit) 
            
            if group_select_mode == 1:
                # Use topk2 sum for group selection
                group_topk = pypto.topk(group, 2, -1, True)[0]
                group_topk = pypto.sum(group_topk, -1)
            else:
                # Use max for group selection
                group_topk = pypto.amax(group, -1, False)

            # Select top-k groups (smallest indices for largest values)
            group_topk, group_topk_id = pypto.topk(group_topk, k_group, -1, True)

            # Step 4: Create mask for selected groups
            mask = pypto.full([bs, group_count], 0.0, group_topk.dtype)
            topk_group_mask_scatter = pypto.scatter_(mask, 1, group_topk_id, 1.0)
            topk_group_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter, -1) 
            pypto.set_vec_tile_shapes(bs, group_count, num_experts) 
            expand = pypto.expand_clone(topk_group_unsqueeze, [bs, group_count, group_unit])
            reshape = pypto.reshape(expand, [bs, num_experts])

            # Mask non-selected experts
            pypto.set_vec_tile_shapes(bs, num_experts)
            twm_not = pypto.logical_not(reshape)
            norm_out = pypto.where(twm_not, 0.0, norm_out)
        
        # Step 5: Select top-k experts
        expect_out, expect_idx_out =  pypto.topk(norm_out, k, -1, True)
        
        # Step 6: Gather weights
        y = pypto.gather(original_norm_out, 1, expect_idx_out)

        # Step 7: Renormalize
        y_sum_eps = pypto.sum(y, -1, True) + eps
        y_out = pypto.div(y, y_sum_eps) * routed_scaling_factor

        return y_out, expect_idx_out, original_norm_out

def moe_gating_topk(    
    x_shape: tuple,
    bias_shape: tuple,
    k: int,
    k_group: int,
    group_count: int,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
    routed_scaling_factor: float,
    eps: float,
    run_mode: str = "npu"):

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    out_shape = [x_shape[0], k]  
    
    @pypto.frontend.jit(
        runtime_options={"run_mode": mode, 
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_function_inner_memory": 128,
        "stitch_cfgcache_size": 2500000,
        "device_sched_mode":1}, 
        
        debug_options={"runtime_debug_mode": 1})
    def moe_gating_topk_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP32),
        bias: pypto.Tensor(bias_shape, pypto.DT_FP32),
    ) -> (
        pypto.Tensor(out_shape, pypto.DT_FP32),
        pypto.Tensor(out_shape, pypto.DT_INT32),
        pypto.Tensor(x_shape, pypto.DT_FP32),
        ):
        y_out = pypto.tensor(out_shape, pypto.DT_FP32)
        expect_idx_out = pypto.tensor(out_shape, pypto.DT_INT32)
        norm_out = pypto.tensor(x_shape, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(64, 512)
        bs_size = 32
        bs_loop = (x_shape[0] + bs_size - 1) // bs_size
        for bs_index in pypto.loop(0, bs_loop, 1, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
            b_offset = bs_index * bs_size
            input_view = pypto.view(x, [bs_size, x_shape[1]], [b_offset, 0])
            #pypto.set_vec_tile_shapes(bs_size, x_shape[1])
            y_out_loop, expect_idx_out_loop, norm_out_loop = moe_gating_topk_core(
                input_view,
                bias,
                k,
                k_group,
                group_count,
                group_select_mode,
                renorm,
                norm_type,
                out_flag,
                routed_scaling_factor,
                eps
            )
            y_out[b_offset:, 0:] = y_out_loop
            expect_idx_out[b_offset:, 0:] = expect_idx_out_loop
            norm_out[b_offset:, 0:] = norm_out_loop
        return y_out, expect_idx_out, norm_out
    return moe_gating_topk_kernel
    
def moe_gating_topk_cpu(   
    x,
    bias, 
    k: int,
    k_group: int,
    group_count: int,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
    routed_scaling_factor: float,
    eps: float):

        if norm_type == 0:
            x, _,_ = softmax_func(x, -1)
        else:
            x = 1/ (1 + torch.exp(-x))

        original_x = x

        if bias is not None:
            x = x + bias

        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            print("grouped x shape:", x.shape) 
            if group_select_mode == 0:
                group_x = torch.amax(x, dim=-1)

            else:
                group_x, _ = torch.topk(x, 2, dim=-1)
                group_x = group_x[..., -2:].sum(dim=-1)
            indices = torch.argsort(-group_x, dim=-1, stable=True)[:, :k_group]
            mask = torch.ones((x.shape[0], group_count), dtype=torch.bool, device='cpu')
            mask.scatter_(1, indices, False)
            x = torch.where(mask.unsqueeze(-1), float('-inf'), x)
            x = x.reshape(x.shape[0], -1)
        indices = torch.argsort(-x, dim=-1, stable=True)
        indices = indices[:, :k]
        y = torch.gather(original_x, 1, indices)     

        if norm_type == 1:
            y /= (torch.sum(y, dim=-1, keepdim=True) + eps)

        y *= routed_scaling_factor 
        y2 = original_x
        return y, indices, y2

def test_moe_gating_topk():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    k = 4
    k_group = 1
    group_count = 4
    group_select_mode = 0
    renorm = 0
    norm_type = 1
    out_flag = True
    routed_scaling_factor = 1.0
    eps = 1e-20

    # 生成随机数据, 并发送到npu
    x = np.random.uniform(0, 2, (1148, 256)).astype(np.float32)
    bias = np.random.uniform(0, 2, (256,)).astype(np.float32)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()

    x_tensor_cpu = torch.tensor(x).cpu()
    bias_tensor_cpu = torch.tensor(bias).cpu()

    print(f"x_tensor_cpu", {x_tensor_cpu})

    y_cpu, expert_idx_cpu, out_cpu = moe_gating_topk_cpu(
        x=x_tensor_cpu,
        bias=bias_tensor_cpu,
        k=k,
        k_group=k_group, 
        group_count=group_count, 
        group_select_mode=group_select_mode, 
        renorm=renorm, 
        norm_type=norm_type, 
        out_flag=out_flag, 
        routed_scaling_factor=routed_scaling_factor, 
        eps=eps)

    print(f"y_cpu", {y_cpu})
    print(f"expert_idx_cpu", {expert_idx_cpu})
    print(f"out_cpu", {out_cpu}) 

    y_npu, expert_idx_npu, out_npu = moe_gating_topk(
        x_shape=x_tensor.shape, 
        bias_shape=bias_tensor.shape,
        k=k,
        k_group=k_group, 
        group_count=group_count, 
        group_select_mode=group_select_mode, 
        renorm=renorm, 
        norm_type=norm_type, 
        out_flag=out_flag, 
        routed_scaling_factor=routed_scaling_factor, 
        eps=eps)(x_tensor, bias_tensor)

    print(f"y_pypto", {y_npu})
    print(f"expert_idx_pypto", {expert_idx_npu})
    print(f"out_pypto", {out_npu}) 

    y_cpu_tensor_list = y_cpu.cpu().flatten().tolist()
    expert_idx_cpu_tensor_list = expert_idx_cpu.cpu().flatten().tolist()
    out_cpu_tensor_list = out_cpu.cpu().flatten().tolist()

    y_npu_tensor_list = y_npu.cpu().flatten().tolist()
    expert_idx_npu_tensor_list = expert_idx_npu.cpu().flatten().tolist()
    out_npu_tensor_list = out_npu.cpu().flatten().tolist()

    # y result
    assert_allclose(np.array(y_cpu_tensor_list),
                    np.array(y_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)

    # idx result
    assert_allclose(np.array(expert_idx_cpu_tensor_list),
                    np.array(expert_idx_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)

    # out result
    assert_allclose(np.array(out_cpu_tensor_list),
                    np.array(out_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)

def main():
    test_moe_gating_topk()

if __name__ == "__main__":
    main()