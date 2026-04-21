"""
Minimal reproduction of Assemble overwrite issue from gdr_origin.py.

Pattern: group = Nv // Nqk > 1
- Outer loop over nv_idx (0..Nv-1)
- nqk_idx = nv_idx // group (multiple nv_idx map to same nqk_idx)
- Each nv_idx computes result from its own input slice
- All write to the same output location: out[bs_ofs:bs_ofs+L, nqk_idx] = result
- Expected: last nv_idx's write wins (nv_idx=1 overwrites nv_idx=0)
- Bug: sometimes first write is kept instead
"""

import torch
import pypto


@pypto.frontend.jit(
    runtime_options={},
    debug_options={"runtime_debug_mode": 1},
    pass_options={
        "vec_nbuffer_setting": {-1: 4, -2: 1},
        "cube_l1_reuse_setting": {-1: 16},
    },
)
def simple_add_kernel(
    x_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    bias_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    act_seq_len: pypto.Tensor([], pypto.DT_INT32),
    cp_result: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
):
    # pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    _, nv, dim = x_in.shape
    nqk = out.shape[1]
    group = nv // nqk
    l = 128

    for b_idx in pypto.loop(act_seq_len.shape[0] - 1, name="LOOP_B", idx_name="b_idx"):
        dyn_seq = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(nv, name="LOOP_NV", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)

            state = pypto.view(bias_in, [1, dim], [nv_idx, 0])

            for inv_s_idx in pypto.loop(0, dyn_seq, l, name="LOOP_S", idx_name="i_idx", unroll_list=[1]):
                s_idx = dyn_seq - inv_s_idx - l
                bs_ofs = b_ofs + s_idx

                pypto.set_vec_tile_shapes(128, 128, 128)
                x_view = pypto.view(x_in, [l, 1, dim], [bs_ofs, nv_idx, 0])
                x_slice = pypto.reshape(x_view, [l, dim])

                result = x_slice + state

                cp_result[nv_idx] = result

                pypto.set_semantic_label("Assemble")
                pypto.set_vec_tile_shapes(128, 128)
                out[bs_ofs:bs_ofs + l, nqk_idx] = result


def main():
    torch.manual_seed(0)
    device_id = 0
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    T = 128
    Nv = 2
    Nqk = 1
    D = 128
    L = 128
    group = Nv // Nqk

    act_seq_len = torch.tensor([0, T], dtype=torch.int32, device=device)

    x = torch.randn(T, Nv, D, device=device, dtype=torch.float32)
    bias = torch.zeros(Nv, D, device=device, dtype=torch.float32)
    bias[0, :] = 0.0
    bias[1, :] = 1.0

    out = torch.zeros(T, Nqk, D, dtype=torch.float32, device=device)
    cp_result = torch.zeros(Nv, L, D, dtype=torch.float32, device=device)

    simple_add_kernel(x, bias, act_seq_len, cp_result, out)
    print('>>> pypto done')

    print("\n" + "=" * 60)
    print("Checkpoint: result per nv_idx (before Assemble)")
    print("=" * 60)
    for h in range(Nv):
        golden_h = x[:, h, :] + bias[h, :].unsqueeze(0)
        diff = (cp_result[h].cpu() - golden_h.cpu()).abs().max().item()
        match = torch.allclose(cp_result[h].cpu(), golden_h.cpu(), rtol=1e-3, atol=1e-3)
        print(f"  cp_result[h={h}] vs golden: {'PASS' if match else 'FAIL'}, max_diff={diff:.6e}")

    print("\n" + "=" * 60)
    print("Assemble overwrite verification")
    print("=" * 60)
    golden_out = x[:, 1, :] + bias[1, :].unsqueeze(0)
    out_cpu = out[:, 0, :].cpu()
    golden_cpu = golden_out.cpu()
    diff = (out_cpu - golden_cpu).abs().max().item()
    match = torch.allclose(out_cpu, golden_cpu, rtol=1e-3, atol=1e-3)
    print(f"  out vs golden (should be h=1): {'PASS' if match else 'FAIL'}, max_diff={diff:.6e}")

    for h in range(Nv):
        golden_h = x[:, h, :] + bias[h, :].unsqueeze(0)
        diff = (out_cpu - golden_h.cpu()).abs().max().item()
        match = torch.allclose(out_cpu, golden_h.cpu(), rtol=1e-3, atol=1e-3)
        print(f"  out vs h={h} result: {'MATCH' if match else 'NO MATCH'}, max_diff={diff:.6e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
