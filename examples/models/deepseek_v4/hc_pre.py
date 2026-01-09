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

