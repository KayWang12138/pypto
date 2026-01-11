import pypto
import torch
import os
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import List

# Constants for shape dimensions
SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

@dataclass
class Rope3dTileConfig:
    # two_dim_tile: List[int]
    three_dim_tile: List[int]
    four_dim_tile: List[int]

def compressor(x, sin, cos, wkv, wgate, ape, weight, out, out1, start_pos, rope_head_dim, name, **kwargs):
    inputs = {
        x: [],
        sin: [],
        cos: [],
        wkv: [],
        wgate: [],
        ape: [],
        weight: []
    }
    outputs = {
        out: [],
        out1: []
    }
    
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    if name == "compressor":
        compressor_kernel(*pto_inputs, None, *pto_outputs, start_pos, rope_head_dim, name)
    if name == "indexer":
        hadamard = kwargs.get('hadamard')
        hadamard = pypto.from_torch(hadamard, dynamic_axis=[])
        compressor_kernel(*pto_inputs, hadamard, *pto_outputs, start_pos, rope_head_dim, name)
        
def overlap_trans(x: pypto.Tensor, value):
    b, s, r, d = x.shape
    d = d // 2
    x1 = pypto.view(x, [b, s, r, d], [0, 0, 0, d])
    x2 = pypto.view(x, [b, s - 1, r, d], [0, 0, 0, 0])
    y = pypto.full([b, 1, r, d], value, x.dtype)
    z = pypto.concat([y, x2], 1)
    z = pypto.concat([z, x1], 2)
    return z
    
def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum)
    return xdiv
    
def rms_norm(input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-6) -> pypto.Tensor:
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim = True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y
    
def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)
    
def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    x_dtype = x.dtype
    rope_dim = x.shape[2]
    
    pypto.set_vec_tile_shapes(1, 32, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.cast(x, pypto.DT_FP32)
    assert x_view.shape == cast_cos.shape
    
    res = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(res, x_dtype)
    return res

def interleaved_rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, rope_3d_config: Rope3dTileConfig) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE).

    Implements RoPE transformation for 3D tensors with shape (batch, heads, dim).
    The RoPE is applied independently to each head using broadcasted cos/sin values.

    Args:
        x: Input tensor of shape (batch, heads, rope_dim)
        cos: Cosine values for RoPE, shape (batch, rope_dim)
        sin: Sine values for RoPE, shape (batch, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    # assert (len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2)

    # pypto.set_vec_tile_shapes(*rope_3d_config.two_dim_tile) # (1, 64)
    pypto.set_vec_tile_shapes(*rope_3d_config.three_dim_tile) # (1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)
    # cast_cos = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    # cast_sin = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    pypto.set_vec_tile_shapes(*rope_3d_config.four_dim_tile)  # (1, 64, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embed, x.dtype)
    
@pypto.jit(
    host_options={"only_codegen": True},
    pass_options={}
)
def compressor_kernel(x, sin, cos, wkv, wgate, ape, weight, hadamard, out, out1, start_pos, rope_head_dim, name):
    pypto.set_options('profile_enable', True)
    pypto.set_debug_options(runtime_debug_mode=1)
    
    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    d = wkv.shape[1]//2
    
    for _ in pypto.loop(1):
        
        ## Matmul
        pypto.set_vec_tile_shapes(64, 64)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
        x = pypto.reshape(x, [b*s, h])
        x = pypto.cast(x, pypto.DT_FP32) ## b,s,h
        kv = pypto.matmul(x, wkv, pypto.DT_FP32) ## b*s,2d
        kv = pypto.reshape(kv, [b, s, 2*d]) ## b,s,2d
        score = pypto.matmul(x, wgate, pypto.DT_FP32)
        score = pypto.reshape(score, [b, s, 2*d]) ## b,s,2d
        
        ## Cut
        cutoff = s - (s % 4)
        cutlen = cutoff // 4
        kv = pypto.view(kv, [b, cutoff, 2*d], [0, 0, 0])
        kv = pypto.reshape(kv, [b, cutlen, 4, 2*d]) ## b,cut,4,2d
        score = pypto.view(score, [b, cutoff, 2*d], [0, 0, 0])
        score = pypto.reshape(score, [b, cutlen, 4, 2*d]) ## b,cut,4,2d
        pypto.set_vec_tile_shapes(1, 8, 4, 2*d)
        score = pypto.add(score, ape) ## b,cut,4,2d
        
        ## overlap_trans\softmax\mul\sum
        pypto.set_vec_tile_shapes(1, 8, 8, d)
        
        ## 两个overlap_trans分成不同图的时候，精度才对
        pypto.set_pass_options(sg_set_scope=1)
        kv = overlap_trans(kv, 0) ## b,cut,8,d
        pypto.set_pass_options(sg_set_scope=-1)
        
        pypto.set_pass_options(sg_set_scope=2)
        score = overlap_trans(score, float("-inf")) ## b,cut,8,d
        score = softmax(score, 2) ## b,cut,8,d
        pypto.set_pass_options(sg_set_scope=-1)
        
        kv = pypto.mul(kv, score) ## b,cut,8,d
        kv = pypto.sum(kv, 2) ## b,cut,d
        
        ## RMSNorm\RoPE\hadamard
        pypto.set_vec_tile_shapes(1, 8, d)
        kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,cut,d
        
        sin = pypto.view(sin, [b, cutlen, rope_head_dim], [0, 0, 0]) ## b, cut, 64
        cos = pypto.view(cos, [b, cutlen, rope_head_dim], [0, 0, 0]) ## b, cut, 64
        kv_nope = pypto.view(kv, [b, cutlen, d-rope_head_dim], [0, 0, 0])
        kv_rope = pypto.view(kv, [b, cutlen, rope_head_dim], [0, 0, d-rope_head_dim])
        #kv_rope = rope_3d(kv_rope, cos, sin)
        rope3d_tile_config = Rope3dTileConfig(
            [1, 64, 64],
            [1, 64, 128, 128]
        )
        kv_rope = interleaved_rope_3d(kv_rope, cos, sin, rope3d_tile_config)
        kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,cut,d

        if name == "indexer":
            kv = pypto.reshape(kv, [b*cutlen, d])
            kv = pypto.matmul(kv, hadamard, pypto.DT_FP32) ## b*cut,d
            kv = pypto.reshape(kv, [b, cutlen, d]) ## b,cut,d
        if name == "compressor":
            kv = pypto.cast(kv, pypto.DT_FP32)
            
        pypto.assemble(kv, [0, 0, 0], out)
        
        ## 更新kv_state和score_state
        
def overlap_transform(tensor: torch.Tensor, value):
    # tensor: [b,s,r,2d]
    b, s, ratio, d = tensor.size()
    d = d//2
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor
    
def RMSNorm(x, eps, weight):
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)
    
def apply_rotary_pos_emb_v2(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    input_dtype = x.dtype
    if input_dtype != torch.float32:
        x = x.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    
    b, s, d = x.shape
    x = x.reshape(b, s, d // 2, 2).permute(0, 1, 3, 2).reshape(b, s, d)
    
    x1, x2 = x.chunk(2, dim=-1)
    p = torch.cat((-x2, x1), dim=-1)
    
    x_embed = (x * cos) + (p * sin)
    x_embed = x_embed.to(input_dtype)
    return x_embed
    
def forward(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, start_pos, rope_head_dim, name, eps=1e-6):
    bsz, seqlen, _ = x.size()
    ratio, overlap, d = 4, 1, 512
    dtype = x.dtype
    x = x.float() ## b,s,h
    kv = torch.matmul(x, wkv) ## b,s,2d
    score = torch.matmul(x, wgate) ## b,s,2d
    if start_pos == 0:
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        assert cutoff==12 and ratio==4
        
        ##跳跃采样未适配
        sin = sin[:,:(cutoff//ratio)] ## b, cut, 64
        cos = cos[:,:(cutoff//ratio)] ## b, cut, 64
        
        offset = ratio if overlap else 0
        if overlap and cutoff >= ratio:
            kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff] ## b,4,2d
            score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + ape ## b,4,2d
        if remainder > 0:
            kv, kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1) ## b,4*cut,2d
            score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + ape[:remainder]
            score = score[:, :cutoff] ## b,4*cut,2d
        kv = kv.unflatten(1, (-1, ratio)) ## b,cut,4,2d
        score = score.unflatten(1, (-1, ratio)) + ape ## b,cut,4,2d
        if overlap:
            kv = overlap_transform(kv, 0) ## b,cut,8,d
            score = overlap_transform(score, float("-inf")) ## b,cut,8,d
        kv = (kv * score.softmax(dim=2)).sum(dim=2) ## b,cut,d
    else:
        should_compress = (start_pos + 1) % ratio == 0
        score += ape[start_pos % ratio]
        if overlap:
            kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv_state = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
                score_state = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
                score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
        else:
            kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv =  (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
    
    if not should_compress:
        return
    kv = RMSNorm(kv.to(dtype), eps, weight) ## b,cut,d
    kv_rope = kv[..., -rope_head_dim:].clone()
    kv_new = kv.clone()
    kv_new[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(kv_rope, sin, cos)
    if name == "indexer":
        kv = torch.matmul(kv_new.float(), hadamard.float()) ## b,cut,d
    if name == "compressor":
        kv = kv_new.float() ## b,cut,d
    # if start_pos == 0:
    #       kv_cache[:bsz, :seqlen // ratio] = kv
    # else:
    #       kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
    return kv
    
def gen_inputs(bsz, seq, h, d, rope_head_dim, device):
    torch.manual_seed(42)
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    sin = torch.rand((bsz, seq//4, rope_head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.rand((bsz, seq//4, rope_head_dim), dtype=torch.bfloat16, device=device)
    wkv = torch.rand((h, 2*d), dtype=torch.float32, device=device)
    wgate = torch.rand((h, 2*d), dtype=torch.float32, device=device)
    ape = torch.rand((4, 2*d), dtype=torch.float32, device=device)
    weight = torch.ones(d, dtype=torch.float32, device=device)
    kv_state = torch.zeros((bsz, seq, 2*d), dtype=torch.float32, device=device)
    score_state = torch.full((bsz, seq, 2*d), float("-inf"), dtype=torch.float32, device=device)
    hadamard = torch.rand((d, d), dtype=torch.bfloat16, device=device)*(d ** -0.5)
    return x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard
    
def test_indexer_comp():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)
    
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    
    bsz = 1
    seq = 12
    h = 4096
    d = 128
    start_pos = 0
    rope_head_dim = 64
    
    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, seq, h, d, rope_head_dim, device)
    kv =  forward(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, start_pos, rope_head_dim, "indexer")
    
    out = torch.zeros((bsz, seq//4, d), dtype=torch.float32, device=device)    
    out1 = torch.zeros((bsz, seq//4, 64), dtype=torch.float32, device=device)
    
    compressor(x, sin, cos, wkv, wgate, ape, weight, out, out1, start_pos, rope_head_dim, "indexer", hadamard=hadamard)
    assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    # assert_allclose(out1.cpu().float()[:,2,:].numpy(), expected.cpu().float()[:,2,:].numpy(), rtol=1e-2, atol=1e-2)
    # print("Compressor completed successfully")
    
def test_compressor():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)
    
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    
    bsz = 1
    seq = 12
    h = 4096
    d = 512
    start_pos = 0
    rope_head_dim = 64
    
    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, seq, h, d, rope_head_dim, device)
    kv =  forward(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, start_pos, rope_head_dim, "compressor")
    
    out = torch.zeros((bsz, seq//4, d), dtype=torch.float32, device=device)    
    out1 = torch.zeros((bsz, seq//4, 64), dtype=torch.float32, device=device)
    
    compressor(x, sin, cos, wkv, wgate, ape, weight, out, out1, start_pos, rope_head_dim, "compressor", hadamard=hadamard)
    assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    # assert_allclose(out1.cpu().float()[:,2,:].numpy(), expected.cpu().float()[:,2,:].numpy(), rtol=1e-2, atol=1e-2)
    # print("Compressor completed successfully")
    
# test_indexer_comp()
test_compressor()