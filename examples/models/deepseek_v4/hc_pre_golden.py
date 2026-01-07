import torch

# t = bsz * seq, dynamic
hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 4096, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc
# x: [t,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [t,d]

def rms_norm_denom(x):
    _, d = x.shape
    print("rms norm x.shape", x.shape)
    x = x.square()
    x = x.sum(-1, True) / d
    x = x + norm_eps
    x = x.sqrt()
    return x


def sigmoid(x):
    x = -x
    x = x.exp()
    x = 1 / (1 + x)
    return x


def hc_split_sinkhorn(x, hc_scale, hc_base):
    t, _ = x.shape # (t, 24)

    pre = x[:, :hc] * hc_scale[0] + hc_base[:, :hc] # (t, 4)
    pre = sigmoid(pre) + hc_eps # (t, 4)

    post = x[:, hc: 2*hc] * hc_scale[1] + hc_base[:, hc: 2*hc] # (t, 4)
    post = 2.0 * sigmoid(post) # (t, 4)

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


def hc_pre(x, hc_fn, hc_scale, hc_base):
    t = x.shape[0]
    x_16 = x.reshape((t, hc * d))
    hc_base = hc_base.reshape(1, mix_hc)
    x = x_16.to(torch.float32)

    hc_fn = hc_fn.to(torch.float32)

    res = torch.matmul(x, hc_fn.transpose(0, 1)) # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
    # mm_res = res

    res = res.to(torch.bfloat16).to(torch.float32)

    res = res / rms_norm_denom(x) # (t, mix_hc) / (t, 1) = (t, mix_hc)
    mm_res = res

    pre, post, comb = hc_split_sinkhorn(res, hc_scale, hc_base) # (t, hc), (t, hc), (t, hc, hc)
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
    res, post, comb, mm_res = hc_pre(x, hc_fn, hc_scale, hc_base)
    # print("res", res.shape, res)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)

    return x, hc_fn, hc_scale, hc_base, res, post, comb, mm_res

if __name__ == "__main__":
    t = 16
    gen_hc_pre_data(t)
