import os
import sys
import pypto
import torch

hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 4096, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc


def gen_hc_split_sinkhorn(comb_flag):
    t, _, _ = comb_flag.shape  # (t, 4, 4)

    conbine_size = 2
    ones = torch.ones([1, hc, hc], dtype=comb_flag.dtype)
    zeros = torch.zeros([1, conbine_size*hc, conbine_size*hc], dtype=comb_flag.dtype)

    for i in range(conbine_size):
        zeros[:, i*hc:(i+1)*hc, i*hc:(i+1)*hc] = ones

    conbine_times = (t + conbine_size - 1) // conbine_size
    res = torch.zeros_like(comb_flag)
    for i in range(conbine_times):
        comb_flag_view = comb_flag[i*conbine_size:(i+1)*conbine_size, :, :] # (conbine_size, 4, 4)
        comb_flag_set = torch.concat([comb_flag_view, comb_flag_view], -1).reshape([conbine_size // 2, 2*hc, 2*hc])    # (conbine_size // 2, 8, 8)
        comb_flag_set = comb_flag_set * zeros

        for _ in range(sinkhorn_iters - 1):
            row_sum = comb_flag_set.sum(-1, keepdim=True)       # (conbine_size // 2, 8, 1)
            comb_flag_set = comb_flag_set / (row_sum + hc_eps)  # (conbine_size // 2, 8, 8)
            col_sum = comb_flag_set.sum(-2, keepdim=True)       # (conbine_size // 2, 1, 8)
            comb_flag_set = comb_flag_set / (col_sum + hc_eps)  # (conbine_size // 2, 8, 8)

        for j in range(conbine_size):
            res[i*conbine_size+j:i*conbine_size+j+1, :, :] = comb_flag_set[:, j*hc:(j+1)*hc, j*hc:(j+1)*hc]

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True)   # (t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True)   # (t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (t, 4, 4)

    print("error ", res - comb_flag)
    print("zeros ", zeros)

    return comb_flag, zeros


def gen_sinkhorn_data(t=16):
    torch.manual_seed(42)
    print("t is ", t)
    x = torch.empty((t, hc, hc), dtype=torch.float32).uniform_(0.1, 0.9)
    res, tmp = gen_hc_split_sinkhorn(x)
    return x, tmp, res


def sinkorn(comb_flag):
    hc_split_sinkhorn_iters = 20

    for _ in range(hc_split_sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True)   # (tile_t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (tile_t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True)   # (tile_t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (tile_t, 4, 4)

    return comb_flag


def sinkorn_2(comb_flag, tmp):
    hc_split_sinkhorn_iters = 20
    tile_t, hc, _ = comb_flag.shape

    pypto.set_vec_tile_shapes(4, 32, 32)

    conbine_size = 2
    conbine_times = (tile_t + conbine_size - 1) // conbine_size
    pypto.set_vec_tile_shapes(4, 32, 32)
    res = pypto.tensor([tile_t, hc, hc], dtype=comb_flag.dtype)
    for i in range(conbine_times):
        comb_flag_view = comb_flag.view([conbine_size, hc, hc], [i*conbine_size, 0, 0]) # (conbine_size, 4, 4)
        comb_flag_set = pypto.concat([comb_flag_view, comb_flag_view], -1).reshape([conbine_size // 2, 2*hc, 2*hc])    # (conbine_size // 2, 8, 8)
        comb_flag_set = comb_flag_set * tmp

        for _ in range(hc_split_sinkhorn_iters - 1):
            row_sum = comb_flag_set.sum(-1, keepdim=True)       # (conbine_size // 2, 8, 1)
            comb_flag_set = comb_flag_set / (row_sum + hc_eps)  # (conbine_size // 2, 8, 8)
            col_sum = comb_flag_set.sum(-2, keepdim=True)       # (conbine_size // 2, 1, 8)
            comb_flag_set = comb_flag_set / (col_sum + hc_eps)  # (conbine_size // 2, 8, 8)

        for j in range(conbine_size):
            pypto.assemble(comb_flag_set[:, j*hc:(j+1)*hc, j*hc:(j+1)*hc ], [j, 0, 0], res)  # (conbine_size, 4, 4))

    return res


@pypto.jit(
    # pass_options={
    #     "vec_nbuffer_mode": 1,
    #     "mg_vec_parallel_lb": 4,
    #     "pg_parallel_lower_bound": 10,
    #     "pg_upper_bound": 10000*10
    # },
    runtime_options={
        "stitch_function_inner_memory": 512,
        "stitch_function_outcast_memory": 512,
        "device_sched_mode": 0,
        # for acl graph
        "stitch_cfgcache_size": 2500000
    },
    debug_options=dict(compile_debug_mode=1, runtime_debug_mode=1)
)
def sinkhorn_kernel(x: pypto.Tensor, tmp: pypto.Tensor, y: pypto.Tensor):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)

    t = x.shape[0]
    hc = x.shape[1]
    unroll_list = [2, 1]

    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        pypto.set_pass_options(sg_set_scope = (1, True, True, -1))
        tile_t = unrollLength

        pypto.set_vec_tile_shapes(4, 32, 32)

        comb_flag = pypto.view(x, [tile_t, hc, hc], [t_idx, 0, 0])

        if tile_t == 1:
            comb_flag = sinkorn(comb_flag)
        else:
            comb_flag = sinkorn_2(comb_flag, tmp)

        print(comb_flag.shape)
        pypto.assemble(comb_flag, [t_idx, 0, 0], y)
        pypto.set_pass_options(sg_set_scope = -1)



def test_sinkhorn(t=16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    x, tmp, y_gd = gen_sinkhorn_data(t)
    print("gen golden success !!!")
    print(y_gd.shape)

    y = torch.zeros_like(y_gd).npu()

    in_outs = {
        x.npu(): [0],
        tmp.npu(): None,
        y:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    sinkhorn_kernel(*pto_in_outs)
    pypto.runtime._device_synchronize()

    y = y.cpu()

if __name__ == "__main__":
    print("start test !!!")
    test_sinkhorn(4)



