import os
import sys
import torch
import torch_npu
from hc_pre_impl import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../deepseek_v32_exp/utils'))
from compare import compare

# t = bsz * seq, dynamic
hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 4096, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc
# x: [t,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [t,d]

def gen_rms_norm_denom(x):
    _, d = x.shape
    print("rms norm x.shape", x.shape)
    x = x.square()
    x = x.sum(-1, True) / d
    x = x + norm_eps
    x = x.sqrt()
    return x


def gen_sigmoid(x):
    x = -x
    x = x.exp()
    x = 1 / (1 + x)
    return x


def gen_hc_split_sinkhorn(x, hc_scale, hc_base):
    t, _ = x.shape # (t, 24)

    pre = x[:, :hc] * hc_scale[0] + hc_base[:, :hc] # (t, 4)
    pre = gen_sigmoid(pre) + hc_eps # (t, 4)

    post = x[:, hc: 2*hc] * hc_scale[1] + hc_base[:, hc: 2*hc] # (t, 4)
    post = 2.0 * gen_sigmoid(post) # (t, 4)

    comb_flag = (x[:, 2*hc: ] * hc_scale[2] + hc_base[:, 2*hc: ]).reshape(t, hc, hc) # (t, 4, 4)
    row_max = comb_flag.amax(-1, keepdim=True) # (t, 4, 1)
    comb_flag = (comb_flag - row_max).exp() # (t, 4, 4)

    row_sum = comb_flag.sum(-1, keepdim=True) # (t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps # (t, 4, 4)
    col_sum = comb_flag.sum(-2, keepdim=True) # (t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps) # (t, 4, 4)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True) # (t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps) # (t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True) # (t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps) # (t, 4, 4)
    return pre, post, comb_flag


@pypto.jit(
    host_options={"only_codegen": True}
    # for acl graph
    # runtime_options={"cfgcache_device_task_num": 100,
    #                  "cfgcache_root_task_num": 1000,
    #                  "cfgcache_leaf_task_num": 10000}
)
def hc_pre_kernel(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor,
):
    # pypto.set_debug_options(runtime_debug_mode=1)
    # pypto.set_debug_options(runtime_debug_mode=2)   ## for acl graph

    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]
    mix_hc = (2 + hc) * hc
    hc_eps = 1e-6

    ### check shape
    assert hc == 4, f"hc is {hc}, expected 4"
    assert d == 4096, f"d is {d}, expected 4096"
    assert mix_hc == hc_fn.shape[0], f"mix_hc is {hc_fn.shape[0]}, expected 24"
    assert hc_scale.shape[0] == 3, f"hc_scale.shape[0] is {hc_scale.shape[0]}, expected 3"

    # unroll_list = [16, 1]
    unroll_list=[1024, 256, 64, 32, 16, 8, 4, 2, 1]

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [t, hc*d], inplace=True)
        hc_base= pypto.reshape(hc_base_, [1, mix_hc], inplace=True)
    print("t in kernel is ", t)
    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        tile_t = unrollLength
        print("========================= tile_t: ", tile_t)
        print("========================= t_idx: ", t_idx)

        pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])
        tile_shapes_1 = [16, 512]
        tile_shape_2 = 64
        if tile_t <= 16:
            tile_shapes_1 = [2, 1024]
            tile_shape_2 = 128
            pypto.set_cube_tile_shapes([8, 8], [1024, 1024], [128, 128])
        elif tile_t <= 64:
            tile_shapes_1 = [8, 1024]
            tile_shape_2 = 32
        else:
            tile_shapes_1 = [16, 512]
            tile_shape_2 = 128

        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1])

        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        rms_res = rms_norm_denom(x_fp32)    ## (t, hc*d) -> (t, 1)

        pypto.set_vec_tile_shapes(tile_shape_2, 16)
        mm_res = pypto.matmul(x_view, hc_fn, pypto.DT_BF16, b_trans=True)   # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
        mm_res = pypto.cast(mm_res, pypto.DT_FP32)

        rms_res = mm_res / rms_res  ## t, mix_hc

        pre = rms_res[:, :hc] * (hc_scale[0:1].reshape([1, 1]).expand_clone([tile_t, 1])) + hc_base[:, :hc] # (tile_t, 4)
        pre = sigmoid(pre) + hc_eps # (tile_t, 4)

        pre_3d = pre.reshape([tile_t, hc, 1])
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d])
        pypto.set_vec_tile_shapes(tile_shape_2, 16, 16)

        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)
        res_bf16 = pypto.cast(res_fp32, pypto.DT_BF16)
        pypto.assemble(res_bf16, [t_idx, 0], y)

        post_ = rms_res[:, hc: 2*hc] * (hc_scale[1:2].reshape([1, 1]).expand_clone([tile_t, 1])) + hc_base[:, hc: 2*hc] # (tile_t, 4)
        post_ = sigmoid(post_) * 2.0 # (tile_t, 4)
        pypto.assemble(post_, [t_idx, 0], post)

        comb_ = hc_split_sinkhorn(rms_res, hc_scale, hc_base, hc, hc_eps)   # (tile_t, hc), (tile_t, hc), (tile_t, hc, hc)
        pypto.assemble(comb_, [t_idx, 0, 0], comb)


def check_input_output_shape_dtype(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    assert x.dim() == 3 and x.size(1) == 4 and x.size(2) == 4096,\
        f"expected x dim num {x.dim()}, x axis1 {x.size(1)}, x axis2 {x.size(2)}"
    assert hc_fn.dim() == 2 and hc_fn.size(0) == 24 and hc_fn.size(1) == 4 * 4096,\
        f"expected hc_fn dim num 2, hc_fn axis0 24, hc_fn axis1 12384"
    assert hc_scale.dim() == 1 and hc_scale.size(0) == 3, f"expected hc_scale dim num 1, hc_scale axis0 3"
    assert hc_base.dim() == 1 and hc_base.size(0) == 24, f"expected hc_scale dim num 1, hc_scale axis0 24"
    
    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert hc_fn.dtype == torch.bfloat16, f"hc_fn.dtype is {hc_fn.dtype}, expected torch.bfloat16"
    assert hc_scale.dtype == torch.float32, f"hc_scale.dtype is {hc_scale.dtype}, expected torch.float32"
    assert hc_base.dtype == torch.float32, f"hc_base.dtype is {hc_base.dtype}, expected torch.float32" 


@allow_in_graph
def npu_hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("x.shape in npu_hc_pre", x.shape)
    ### check dtype
    check_input_output_shape_dtype(x, hc_fn, hc_scale, hc_base)

    y = torch.zeros([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.zeros([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')
    comb = torch.zeros([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')

    in_outs = {
        x: [0],
        hc_fn: None,
        hc_scale: None,
        hc_base: None,
        y:[0],
        post:[0],
        comb:[0],
    }


def gen_hc_pre(x, hc_fn, hc_scale, hc_base):
    t = x.shape[0]
    x_16 = x.reshape((t, hc * d))
    hc_base = hc_base.reshape(1, mix_hc)
    x = x_16.to(torch.float32)

    hc_fn = hc_fn.to(torch.float32)

    res = torch.matmul(x, hc_fn.transpose(0, 1)) # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
    # mm_res = res

    res = res.to(torch.bfloat16).to(torch.float32)

    res = res / gen_rms_norm_denom(x) # (t, mix_hc) / (t, 1) = (t, mix_hc)
    mm_res = res

    pre, post, comb = gen_hc_split_sinkhorn(res, hc_scale, hc_base) # (t, hc), (t, hc), (t, hc, hc)
    mul_res = pre.reshape(t, hc, 1) * x.reshape(t, hc, d)
    res = mul_res.sum(-2) # (t,mul_res d)
    res = res.to(torch.bfloat16)
    return res, post, comb, mm_res

def gen_hc_pre_data(t = 16):
    print("t is ", t)
    x = torch.empty((t, hc, d), dtype=torch.bfloat16).uniform_(-1, 1)
    hc_fn = torch.empty((mix_hc, hc*d), dtype=torch.bfloat16).uniform_(-1, 1)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1)
    hc_base = torch.empty((mix_hc, ), dtype=torch.float32).uniform_(-1, 1)
    res, post, comb, mm_res = gen_hc_pre(x, hc_fn, hc_scale, hc_base)
    # print("res", res.shape, res)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)
    return x, hc_fn, hc_scale, hc_base, res, post, comb, mm_res

pyptolib = torch.library.Library("pypto", "FRAGMENT") 
pyptolib.define("hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(pyptolib, "hc_pre", "Meta")
def hc_pre(x, hc_fn, hc_scale, hc_base):
    y = torch.empty([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.empty([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    comb = torch.empty([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    return y, post, comb


@torch.library.impl(pyptolib, "hc_pre", "NPU")
def hc_pre(x, hc_fn, hc_scale, hc_base):
    return npu_hc_pre(x, hc_fn, hc_scale, hc_base)

class HC_PRE(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        return torch.ops.pypto.hc_pre(x, hc_fn, hc_scale, hc_base)

def test_hc_pre_inmodel(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)
    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    ### to device
    x = x.to(device=f'npu:{device_id}')
    hc_fn = hc_fn.to(device=f'npu:{device_id}')
    hc_scale = hc_scale.to(device=f'npu:{device_id}')
    hc_base = hc_base.to(device=f'npu:{device_id}')

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(HC_PRE(), dynamic=False, fullgraph=True, backend=npu_backend)
    y, post, comb = model(x, hc_fn, hc_scale, hc_base)
    pypto.runtime._device_synchronize()

    ### compare
    compare(y.cpu(), y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post.cpu(), post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb.cpu(), comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")

def test_hc_pre(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
    post = torch.zeros_like(post_gd).to(device=f'npu:{device_id}')
    comb = torch.zeros_like(comb_gd).to(device=f'npu:{device_id}')
    # mm_res = torch.zeros_like(mm_res_gd).to(device=f'npu:{device_id}')

    in_outs = {
        x.to(device=f'npu:{device_id}'): [0],
        hc_fn.to(device=f'npu:{device_id}'): None,
        hc_scale.to(device=f'npu:{device_id}'): None,
        hc_base.to(device=f'npu:{device_id}'): None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    hc_pre_kernel(*pto_in_outs)
    torch_npu.npu.synchronize()

    # mm_res = mm_res.cpu()
    y = y.cpu()
    post = post.cpu()
    comb = comb.cpu()

    # print("y", y.shape, y)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)

    # compare(mm_res, mm_res_gd, "mm_res", atol=0.0001, rtol=0.0078125)
    # print("mm_res compare success!!!")

    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")


if __name__ == "__main__":
    print("start test !!!")
    test_hc_pre_inmodel(16)
    # decode_t_list = {2048, 8192, }
    # for t_dyn in decode_t_list:
    #     test_hc_pre(t_dyn)
    # test_hc_pre(127)

