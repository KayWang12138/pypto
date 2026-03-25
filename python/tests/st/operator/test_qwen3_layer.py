import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pypto
import torch

from st.pypto_test import TestBuilder

torch.manual_seed(0)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_ROOT = _REPO_ROOT / "output"


@dataclass
class PaTileConfig:
    head_num_q_tile: int
    c1_tile_shape: tuple
    v1_tile_shape: tuple
    c2_tile_shape: tuple
    v2_tile_shape: tuple


def _gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.rand(data_shape) < 0.5
    return (torch.rand(data_shape) * (max_value - min_value) + min_value).to(dtype)


def _convert_tensors_contiguous(tensor_list: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    for tensor in tensor_list:
        result.append(tensor if not isinstance(tensor, torch.Tensor) or tensor.is_contiguous() else tensor.contiguous())
    return result


def _align_up(value: int, base: int) -> int:
    if value <= 0:
        return base
    return ((value + base - 1) // base) * base


def _get_matmul_tile(m: int, k: int, n: int, dtype):
    m_tile = _align_up(min(max(m, 1), 64), 16)
    k_tile = _align_up(min(max(k, 1), 256), 16)
    n_tile = _align_up(min(max(n, 1), 256), 16)
    bytes_per_elem = 4 if dtype == pypto.DT_FP32 else 2
    l0b_bytes = 64 * 1024
    while k_tile * n_tile * bytes_per_elem > l0b_bytes:
        if n_tile >= k_tile and n_tile > 16:
            n_tile = max(16, n_tile // 2)
        elif k_tile > 16:
            k_tile = max(16, k_tile // 2)
        else:
            break
        k_tile = _align_up(k_tile, 16)
        n_tile = _align_up(n_tile, 16)
    return m_tile, k_tile, n_tile


def _set_matmul_tile(m: int, k: int, n: int, dtype):
    m_tile, k_tile, n_tile = _get_matmul_tile(m, k, n, dtype)
    pypto.set_cube_tile_shapes([m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile])
    pypto.set_matrix_size([m, k, n])


def _torch_dtype_to_pypto(dtype: torch.dtype):
    if dtype == torch.float16:
        return pypto.DT_FP16
    if dtype == torch.float32:
        return pypto.DT_FP32
    if dtype == torch.bfloat16:
        return pypto.DT_BF16
    raise ValueError(f"Unsupported torch dtype for frontend.jit kernel: {dtype}")


def _build_prolog_act_seq_list(params) -> List[int]:
    b = params["b"]
    s = params["s"]
    block_size = params["block_size"]
    if b == 1:
        return [block_size * 2 + s]
    return [block_size * 2 + s, block_size * 2 - 1 + s] + [block_size * 2 + s] * (b - 2)


def _build_attention_act_seq_list(params) -> List[int]:
    skv = params["skv"]
    b = params["b"]
    return [skv] * b if isinstance(skv, int) else [int(x) for x in skv]


def _get_cache_layout(act_seq_list: List[int], block_size: int):
    block_num_per_batch = [int(math.ceil(x / block_size)) for x in act_seq_list]
    max_block_num_per_batch = int(max(block_num_per_batch))
    block_num = int(sum(block_num_per_batch))
    return block_num * block_size, max_block_num_per_batch


def _collect_output_dirs():
    if not _OUTPUT_ROOT.exists():
        return set()
    return {path.resolve() for path in _OUTPUT_ROOT.glob("output_*") if path.is_dir()}


def _rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp32 * rstd
    if gamma is not None:
        out = out * gamma.to(torch.float32)
    return out.to(x.dtype)


def _rope_rearrange_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape)


def _rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_fp32 = _rope_rearrange_torch(x.to(torch.float32))
    cos_u = cos.to(torch.float32).unsqueeze(2)
    sin_u = sin.to(torch.float32).unsqueeze(2)
    return (x_fp32 * cos_u + _rotate_half_torch(x_fp32) * sin_u).to(x.dtype)


def _build_block_table(act_seq_list: List[int], block_size: int):
    block_num_per_batch = [int(math.ceil(x / block_size)) for x in act_seq_list]
    block_num = int(sum(block_num_per_batch))
    max_block_num_per_batch = int(max(block_num_per_batch))
    block_table = torch.full([len(act_seq_list), max_block_num_per_batch], -1, dtype=torch.int32)
    block_id = 0
    for batch_idx, block_num_in_batch in enumerate(block_num_per_batch):
        for block_idx in range(block_num_in_batch):
            block_table[batch_idx, block_idx] = block_id
            block_id += 1
    return block_table, block_num_per_batch, block_num


def _build_cache_from_bsnd(k_bsnd: torch.Tensor, v_bsnd: torch.Tensor, act_seq_list: List[int], block_size: int):
    b, _, n_kv, d = k_bsnd.shape
    block_table, block_num_per_batch, block_num = _build_block_table(act_seq_list, block_size)
    k_cache = torch.zeros([block_num, block_size, n_kv, d], dtype=k_bsnd.dtype)
    v_cache = torch.zeros([block_num, block_size, n_kv, d], dtype=v_bsnd.dtype)
    for batch_idx, seq_len in enumerate(act_seq_list):
        for block_idx in range(block_num_per_batch[batch_idx]):
            global_block_id = int(block_table[batch_idx, block_idx].item())
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            if end > start:
                k_cache[global_block_id, 0 : (end - start), :, :] = k_bsnd[batch_idx, start:end, :, :]
                v_cache[global_block_id, 0 : (end - start), :, :] = v_bsnd[batch_idx, start:end, :, :]
    return (
        block_table,
        block_num_per_batch,
        k_cache.reshape(block_num * block_size, n_kv * d),
        v_cache.reshape(block_num * block_size, n_kv * d),
    )


def _build_cache_position_from_act_seq_list(act_seq_list: List[int], s: int):
    b = len(act_seq_list)
    cache_position = torch.zeros([b, s], dtype=torch.int32)
    for batch_idx, act_seq in enumerate(act_seq_list):
        start_pos = int(act_seq) - int(s)
        for token_idx in range(s):
            cache_position[batch_idx, token_idx] = start_pos + token_idx
    return cache_position


def _build_cache_index(cache_position: torch.Tensor, block_table: torch.Tensor, block_size: int):
    b, s = cache_position.shape
    cache_index = torch.zeros([b, s], dtype=torch.int32)
    for batch_idx in range(b):
        for token_idx in range(s):
            pos = int(cache_position[batch_idx, token_idx].item())
            block_idx_in_batch = pos // block_size
            offset_in_block = pos % block_size
            global_block_id = int(block_table[batch_idx, block_idx_in_batch].item())
            cache_index[batch_idx, token_idx] = global_block_id * block_size + offset_in_block
    return cache_index


def _build_rope_from_cache_position(cache_position: torch.Tensor, d: int, dtype):
    pos_fp32 = cache_position.to(torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d)))
    freqs = pos_fp32.unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)


def _build_pa_rows_from_cache(
    cache: torch.Tensor, block_table: torch.Tensor, block_size: int, act_seq: int, kv_idx: int, d: int
):
    rows = []
    for pos in range(act_seq):
        block_idx_in_batch = pos // block_size
        offset_in_block = pos % block_size
        global_block_id = int(block_table[block_idx_in_batch].item())
        row_idx = global_block_id * block_size + offset_in_block
        rows.append(cache[row_idx, kv_idx * d : (kv_idx + 1) * d])
    return torch.stack(rows, dim=0)


def _rotate_half_graph(x: pypto.Tensor) -> pypto.Tensor:
    rank = x.dim
    shape = list(x.shape)
    half = x.shape[-1] // 2
    left_shape = shape[:-1] + [half]
    zeros = [0] * rank
    left = pypto.view(x, left_shape, zeros)
    right = pypto.view(x, left_shape, zeros[:-1] + [half])
    out = pypto.tensor(shape, pypto.DT_FP32, "qwen3_rotate_half")
    pypto.assemble(pypto.mul(right, -1.0), zeros, out)
    pypto.assemble(left, zeros[:-1] + [half], out)
    return out


def _rope_rearrange_graph(x: pypto.Tensor) -> pypto.Tensor:
    shape = list(x.shape)
    half = x.shape[-1] // 2
    tile_mid = max(16, min(64, int(shape[-1])))
    if x.dim == 2:
        pypto.set_vec_tile_shapes(1, tile_mid, tile_mid)
    elif x.dim == 3:
        pypto.set_vec_tile_shapes(1, min(8, int(shape[1])), tile_mid, tile_mid)
    elif x.dim == 4:
        pypto.set_vec_tile_shapes(1, min(8, int(shape[1])), tile_mid, tile_mid)
    x_view = pypto.reshape(x, shape[:-1] + [half, 2])
    x_trans = pypto.transpose(x_view, x_view.dim - 2, x_view.dim - 1)
    return pypto.reshape(x_trans, shape)


def _apply_rope_graph(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    b, s, n, d = list(x.shape)
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    x_3d = pypto.reshape(x_fp32, [b * s, n, d])
    cos_3d = pypto.reshape(pypto.cast(cos, pypto.DT_FP32), [b * s, 1, d])
    sin_3d = pypto.reshape(pypto.cast(sin, pypto.DT_FP32), [b * s, 1, d])
    x_rearranged = _rope_rearrange_graph(x_3d)
    tile_last = max(16, min(64, d))
    pypto.set_vec_tile_shapes(1, min(8, n), tile_last)
    rotated = _rotate_half_graph(x_rearranged)
    x_embed = pypto.add(pypto.mul(x_rearranged, cos_3d), pypto.mul(rotated, sin_3d))
    pypto.set_vec_tile_shapes(1, 1, min(8, n), tile_last)
    return pypto.cast(pypto.reshape(x_embed, [b, s, n, d]), x.dtype)


def _build_frontend_rope_helpers(b: int, s: int, heads: int, d: int, pto_dtype):
    half = d // 2
    tile_last = max(16, min(64, d))
    tile_half = max(16, min(64, half))

    @pypto.frontend.function
    def rope_rearrange(
        x: pypto.tensor((b, s, heads, d), pypto.DT_FP32)
    ) -> pypto.tensor((b, s, heads, d), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_view = pypto.reshape(x, [b, s, heads, half, 2])
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_half, 2)
        x_trans = pypto.transpose(x_view, x_view.dim - 2, x_view.dim - 1)
        out = pypto.reshape(x_trans, [b, s, heads, d])
        return out

    @pypto.frontend.function
    def rotate_half(x: pypto.tensor((b, s, heads, d), pypto.DT_FP32)) -> pypto.tensor((b, s, heads, d), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        left = x[:, :, :, :half]
        right = x[:, :, :, half:]
        neg_right = pypto.mul(right, -1.0)
        out = pypto.concat([neg_right, left], -1)
        return out

    @pypto.frontend.function
    def apply_rope(
        x: pypto.tensor((b, s, heads, d), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
    ) -> pypto.tensor((b, s, heads, d), pto_dtype):
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_fp32 = pypto.cast(x, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 1, tile_last)
        x_rearranged = rope_rearrange(x_fp32)
        pypto.set_vec_tile_shapes(1, 1, tile_last)
        cos_4d = pypto.reshape(pypto.cast(cos, pypto.DT_FP32), [b, s, 1, d])
        pypto.set_vec_tile_shapes(1, 1, tile_last)
        sin_4d = pypto.reshape(pypto.cast(sin, pypto.DT_FP32), [b, s, 1, d])
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        rotated = rotate_half(x_rearranged)
        out_fp32 = pypto.add(pypto.mul(x_rearranged, cos_4d), pypto.mul(rotated, sin_4d))
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        out = pypto.cast(out_fp32, pto_dtype)
        return out

    return rotate_half, rope_rearrange, apply_rope


def _build_flat_frontend_rope_helper(b: int, s: int, heads: int, d: int, pto_dtype):
    half = d // 2
    tile_last = max(16, min(64, d))
    tile_half = max(16, min(64, half))

    @pypto.frontend.function
    def apply_rope(
        x: pypto.tensor((b, s, heads, d), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
    ) -> pypto.tensor((b, s, heads, d), pto_dtype):
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_fp32 = pypto.cast(x, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_view = pypto.reshape(x_fp32, [b, s, heads, half, 2])
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_half, 2)
        x_trans = pypto.transpose(x_view, x_view.dim - 2, x_view.dim - 1)
        x_rearranged = pypto.reshape(x_trans, [b, s, heads, d])

        pypto.set_vec_tile_shapes(1, 1, tile_last)
        cos_4d = pypto.reshape(pypto.cast(cos, pypto.DT_FP32), [b, s, 1, d])
        pypto.set_vec_tile_shapes(1, 1, tile_last)
        sin_4d = pypto.reshape(pypto.cast(sin, pypto.DT_FP32), [b, s, 1, d])

        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        left = x_rearranged[:, :, :, :half]
        right = x_rearranged[:, :, :, half:]
        neg_right = pypto.mul(right, -1.0)
        rotated = pypto.concat([neg_right, left], -1)

        out_fp32 = pypto.add(pypto.mul(x_rearranged, cos_4d), pypto.mul(rotated, sin_4d))
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        return pypto.cast(out_fp32, pto_dtype)

    return apply_rope


def _build_mlp_core_frontend(m: int, h: int, inter: int, pto_dtype):
    mm1_tile = _get_matmul_tile(m, h, inter, pto_dtype)
    mm2_tile = _get_matmul_tile(m, inter, h, pto_dtype)

    @pypto.frontend.function
    def mlp_core(
        hidden_states: pypto.tensor((m, h), pto_dtype),
        gate_w: pypto.tensor((h, inter), pto_dtype),
        up_w: pypto.tensor((h, inter), pto_dtype),
        down_w: pypto.tensor((inter, h), pto_dtype),
    ) -> pypto.tensor((m, h), pto_dtype):
        pypto.set_vec_tile_shapes(1, h)
        pypto.set_cube_tile_shapes([mm1_tile[0], mm1_tile[0]], [mm1_tile[1], mm1_tile[1]], [mm1_tile[2], mm1_tile[2]])
        pypto.set_matrix_size([m, h, inter])
        gate = pypto.matmul(hidden_states, gate_w, pto_dtype)

        pypto.set_cube_tile_shapes([mm1_tile[0], mm1_tile[0]], [mm1_tile[1], mm1_tile[1]], [mm1_tile[2], mm1_tile[2]])
        pypto.set_matrix_size([m, h, inter])
        up = pypto.matmul(hidden_states, up_w, pto_dtype)

        gate_sig = pypto.cast(pypto.sigmoid(pypto.cast(gate, pypto.DT_FP32)), pto_dtype)
        gate_act = pypto.mul(gate, gate_sig)
        inter_fp32 = pypto.mul(pypto.cast(gate_act, pypto.DT_FP32), pypto.cast(up, pypto.DT_FP32))
        inter_dt = pypto.cast(inter_fp32, pto_dtype)

        pypto.set_cube_tile_shapes([mm2_tile[0], mm2_tile[0]], [mm2_tile[1], mm2_tile[1]], [mm2_tile[2], mm2_tile[2]])
        pypto.set_matrix_size([m, inter, h])
        output = pypto.matmul(inter_dt, down_w, pto_dtype)
        return output

    return mlp_core


def qwen3_mlp_graph(params, hidden_states, gate_w, up_w, down_w, mlp_out):
    m = params["b"] * params["s"]
    h = params["h"]
    inter = params["inter"]
    dtype = hidden_states.dtype

    pypto.set_vec_tile_shapes(1, h)
    _set_matmul_tile(m, h, inter, dtype)
    gate = pypto.matmul(hidden_states, gate_w, dtype)
    _set_matmul_tile(m, h, inter, dtype)
    up = pypto.matmul(hidden_states, up_w, dtype)

    gate_sig = pypto.cast(pypto.sigmoid(pypto.cast(gate, pypto.DT_FP32)), dtype)
    gate_act = pypto.mul(gate, gate_sig)
    inter_fp32 = pypto.mul(pypto.cast(gate_act, pypto.DT_FP32), pypto.cast(up, pypto.DT_FP32))
    inter_dt = pypto.cast(inter_fp32, dtype)

    _set_matmul_tile(m, inter, h, dtype)
    mlp_out[:] = pypto.matmul(inter_dt, down_w, dtype)


def build_qwen3_mlp_frontend_jit(params, run_mode):
    m = params["b"] * params["s"]
    h = params["h"]
    inter = params["inter"]
    pto_dtype = _torch_dtype_to_pypto(params["dtype"])
    mlp_core = _build_mlp_core_frontend(m, h, inter, pto_dtype)

    @pypto.frontend.jit(
        runtime_options={
            "run_mode": run_mode,
            "stitch_function_num_initial": 1,
            "stitch_function_num_step": 1,
            "stitch_function_inner_memory": 1,
            "stitch_function_outcast_memory": 1,
            "stitch_function_size": 2048,
            "valid_shape_optimize": 1,
        },
    )
    def qwen3_mlp_frontend_jit(
        hidden_states: pypto.tensor((m, h), pto_dtype),
        gate_w: pypto.tensor((h, inter), pto_dtype),
        up_w: pypto.tensor((h, inter), pto_dtype),
        down_w: pypto.tensor((inter, h), pto_dtype),
    ) -> pypto.tensor((m, h), pto_dtype):
        output = mlp_core(hidden_states, gate_w, up_w, down_w)
        return output

    return qwen3_mlp_frontend_jit


def build_qwen3_paged_attention_prolog_frontend_jit(params, run_mode):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    half = d // 2
    tile_last = max(16, min(64, d))
    tile_half = max(16, min(64, half))
    hidden_size = n_q * d
    kv_hidden = n_kv * d
    pto_dtype = _torch_dtype_to_pypto(params["dtype"])
    q_mm_tile = _get_matmul_tile(b * s, hidden_size, hidden_size, pto_dtype)
    kv_mm_tile = _get_matmul_tile(b * s, hidden_size, kv_hidden, pto_dtype)
    act_seq_list = _build_prolog_act_seq_list(params)
    cache_rows, _ = _get_cache_layout(act_seq_list, params["block_size"])

    def apply_rope_no_nested(x, cos, sin, heads):
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_fp32 = pypto.cast(x, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        x_view = pypto.reshape(x_fp32, [b, s, heads, half, 2])
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_half, 2)
        x_trans = pypto.transpose(x_view, x_view.dim - 2, x_view.dim - 1)
        x_rearranged = pypto.reshape(x_trans, [b, s, heads, d])

        pypto.set_vec_tile_shapes(1, 1, tile_last)
        cos_4d = pypto.reshape(pypto.cast(cos, pypto.DT_FP32), [b, s, 1, d])
        pypto.set_vec_tile_shapes(1, 1, tile_last)
        sin_4d = pypto.reshape(pypto.cast(sin, pypto.DT_FP32), [b, s, 1, d])

        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        left = x_rearranged[:, :, :, :half]
        right = x_rearranged[:, :, :, half:]
        neg_right = pypto.mul(right, -1.0)
        rotated = pypto.concat([neg_right, left], -1)

        out_fp32 = pypto.add(pypto.mul(x_rearranged, cos_4d), pypto.mul(rotated, sin_4d))
        pypto.set_vec_tile_shapes(1, 1, min(8, heads), tile_last)
        return pypto.cast(out_fp32, pto_dtype)

    @pypto.frontend.jit(
        runtime_options={"run_mode": run_mode},
    )
    def qwen3_paged_attention_prolog_frontend_jit(
        hidden_states: pypto.tensor((b * s, hidden_size), pto_dtype),
        attn_q_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_q_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_k_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_k_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_v_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_v_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_q_norm_w: pypto.tensor((d,), pto_dtype),
        attn_k_norm_w: pypto.tensor((d,), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
        cache_index: pypto.tensor((b, s), pypto.DT_INT32),
        key_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        value_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
    ) -> (
        pypto.tensor((b * s * n_q, d), pto_dtype),
        pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        pypto.tensor((cache_rows, kv_hidden), pto_dtype),
    ):
        # Inline the prolog body to avoid the new parser's multi-output nested function issue.
        pypto.set_cube_tile_shapes(
            [q_mm_tile[0], q_mm_tile[0]],
            [q_mm_tile[1], q_mm_tile[1]],
            [q_mm_tile[2], q_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, hidden_size])
        q = pypto.matmul(hidden_states, attn_q_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, hidden_size)
        q = pypto.cast(
            pypto.add(
                pypto.cast(q, pypto.DT_FP32),
                pypto.cast(attn_q_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        k = pypto.matmul(hidden_states, attn_k_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        k = pypto.cast(
            pypto.add(
                pypto.cast(k, pypto.DT_FP32),
                pypto.cast(attn_k_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        v = pypto.matmul(hidden_states, attn_v_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        v = pypto.cast(
            pypto.add(
                pypto.cast(v, pypto.DT_FP32),
                pypto.cast(attn_v_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
        q_flat = pypto.reshape(q, [b * s * n_q, d])
        q_norm = pypto.rms_norm(q_flat, attn_q_norm_w)

        pypto.set_vec_tile_shapes(min(16, b * s * n_kv), d)
        k_flat = pypto.reshape(k, [b * s * n_kv, d])
        k_norm = pypto.rms_norm(k_flat, attn_k_norm_w)

        q_4d = pypto.reshape(q_norm, [b, s, n_q, d])
        k_4d = pypto.reshape(k_norm, [b, s, n_kv, d])
        v_4d = pypto.reshape(v, [b, s, n_kv, d])

        q_embed = apply_rope_no_nested(q_4d, cos, sin, n_q)
        k_embed = apply_rope_no_nested(k_4d, cos, sin, n_kv)
        query_out = pypto.reshape(q_embed, [b * s * n_q, d])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_rows = pypto.reshape(k_embed, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_rows = pypto.reshape(v_4d, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_cache_out = pypto.scatter_update(key_cache, -2, cache_index, key_rows)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_cache_out = pypto.scatter_update(value_cache, -2, cache_index, value_rows)
        return query_out, key_cache_out, value_cache_out

    return qwen3_paged_attention_prolog_frontend_jit


def build_qwen3_paged_attention_frontend_jit(params, run_mode):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    block_size = params["block_size"]
    tile_config = params["pa_tile_config"]
    max_unroll_times = params["max_unroll_times"]
    pto_dtype = _torch_dtype_to_pypto(params["dtype"])
    act_seq_list = _build_attention_act_seq_list(params)
    cache_rows, block_cols = _get_cache_layout(act_seq_list, block_size)
    group = n_q // n_kv
    n_tile = tile_config.head_num_q_tile
    n_loop = n_q // n_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    softmax_scale = float(1.0 / math.sqrt(d))

    @pypto.frontend.jit(
        runtime_options={"run_mode": run_mode},
    )
    def qwen3_paged_attention_frontend_jit(
        query: pypto.tensor((b * s * n_q, d), pto_dtype),
        key_cache: pypto.tensor((cache_rows, n_kv * d), pto_dtype),
        value_cache: pypto.tensor((cache_rows, n_kv * d), pto_dtype),
        block_table: pypto.tensor((b, block_cols), pypto.DT_INT32),
        act_seqs: pypto.tensor((b,), pypto.DT_INT32),
    ) -> pypto.tensor((b * s * n_q, d), pypto.DT_FP32):
        attention_out = pypto.tensor((b * s * n_q, d), pypto.DT_FP32)
        block_size_sym = pypto.symbolic_scalar(block_size)
        for b_idx in pypto.loop(0, b, 1, name="QWEN3_PA_L0_B", idx_name="b_idx"):
            cur_act_seq = act_seqs[b_idx]
            cur_act_seq.as_variable()
            for s1_idx in pypto.loop(0, s, 1, name="QWEN3_PA_L1_S1", idx_name="s1_idx"):
                cur_seq = (cur_act_seq - s + 1 + s1_idx).max(0)
                cur_seq.as_variable()
                bn_per_batch = (cur_seq + block_size - 1) // block_size
                bn_per_batch.as_variable()
                for n_idx in pypto.loop(0, n_loop, 1, name="QWEN3_PA_L2_N", idx_name="n_idx"):
                    kv_idx = (n_idx * n_tile) // group
                    kv_idx.as_variable()
                    cur_offset = b_idx * s * n_q + s1_idx * n_q + n_idx * n_tile
                    oi_offset = [cur_offset, 0]
                    oi_update = pypto.tensor((n_tile, d), pypto.DT_FP32)
                    li_update = pypto.tensor((n_tile, 1), pypto.DT_FP32)
                    mi_update = pypto.tensor((n_tile, 1), pypto.DT_FP32)

                    for bn in pypto.loop(
                        0,
                        bn_per_batch,
                        1,
                        name="QWEN3_PA_L3_BN",
                        idx_name="bn",
                        unroll_List={max_unroll_times},
                    ):
                        valid_s2 = (cur_seq - bn * block_size).min(block_size_sym)
                        qi = pypto.view(query, [n_tile, d], [cur_offset, 0])
                        cur_block_idx = block_table[b_idx, bn]
                        cur_block_idx.as_variable()
                        kj = pypto.view(
                            key_cache,
                            [block_size, d],
                            [cur_block_idx * block_size, kv_idx * d],
                            valid_shape=[valid_s2, d],
                        )
                        vj = pypto.view(
                            value_cache,
                            [block_size, d],
                            [cur_block_idx * block_size, kv_idx * d],
                            valid_shape=[valid_s2, d],
                        )

                        pypto.set_cube_tile_shapes(
                            [c1_tile[0], c1_tile[1]],
                            [c1_tile[2], c1_tile[3]],
                            [c1_tile[4], c1_tile[5]],
                        )
                        pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_dt = pypto.cast(tilda_pij, pto_dtype)
                        tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                        if pypto.is_loop_begin(bn):
                            pypto.set_cube_tile_shapes(
                                [c2_tile[0], c2_tile[1]],
                                [c2_tile[2], c2_tile[3]],
                                [c2_tile[4], c2_tile[5]],
                            )
                            pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                            oi_tmp = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.is_loop_end(bn):
                                pypto.assemble(pypto.div(oi_tmp, tilda_lij), oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = tilda_lij
                                mi_update[:] = tilda_mij
                        else:
                            mi_new = pypto.maximum(mi_update, tilda_mij)
                            t2 = pypto.exp(pypto.sub(mi_update, mi_new))
                            t4 = pypto.exp(pypto.sub(tilda_mij, mi_new))
                            li_new = pypto.add(pypto.mul(t2, li_update), pypto.mul(t4, tilda_lij))
                            q3 = pypto.mul(oi_update, t2)
                            pypto.set_cube_tile_shapes(
                                [c2_tile[0], c2_tile[1]],
                                [c2_tile[2], c2_tile[3]],
                                [c2_tile[4], c2_tile[5]],
                            )
                            pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                            q1 = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = pypto.add(q3, pypto.mul(q1, t4))
                            if pypto.is_loop_end(bn):
                                pypto.assemble(pypto.div(oi_tmp, li_new), oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new
        return attention_out

    return qwen3_paged_attention_frontend_jit


def build_qwen3_layer_frontend_jit(params, run_mode):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    block_size = params["block_size"]
    inter = params["inter"]
    hidden_size = n_q * d
    kv_hidden = n_kv * d
    pto_dtype = _torch_dtype_to_pypto(params["dtype"])
    act_seq_list = _build_attention_act_seq_list(params)
    cache_rows, block_cols = _get_cache_layout(act_seq_list, block_size)
    q_mm_tile = _get_matmul_tile(b * s, hidden_size, hidden_size, pto_dtype)
    kv_mm_tile = _get_matmul_tile(b * s, hidden_size, kv_hidden, pto_dtype)
    o_mm_tile = _get_matmul_tile(b * s, hidden_size, hidden_size, pypto.DT_FP32)
    mlp_core = _build_mlp_core_frontend(b * s, hidden_size, inter, pto_dtype)
    apply_rope_q = _build_flat_frontend_rope_helper(b, s, n_q, d, pto_dtype)
    apply_rope_k = _build_flat_frontend_rope_helper(b, s, n_kv, d, pto_dtype)
    tile_config = params["pa_tile_config"]
    max_unroll_times = params["max_unroll_times"]
    group = n_q // n_kv
    n_tile = tile_config.head_num_q_tile
    n_loop = n_q // n_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    softmax_scale = float(1.0 / math.sqrt(d))

    @pypto.frontend.function
    def paged_attention_prolog_core(
        hidden_states: pypto.tensor((b * s, hidden_size), pto_dtype),
        attn_q_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_q_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_k_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_k_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_v_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_v_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_q_norm_w: pypto.tensor((d,), pto_dtype),
        attn_k_norm_w: pypto.tensor((d,), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
        cache_index: pypto.tensor((b, s), pypto.DT_INT32),
        key_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        value_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
    ) -> (
        pypto.tensor((b * s * n_q, d), pto_dtype),
        pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        pypto.tensor((cache_rows, kv_hidden), pto_dtype),
    ):
        pypto.set_cube_tile_shapes(
            [q_mm_tile[0], q_mm_tile[0]],
            [q_mm_tile[1], q_mm_tile[1]],
            [q_mm_tile[2], q_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, hidden_size])
        q = pypto.matmul(hidden_states, attn_q_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, hidden_size)
        q = pypto.cast(
            pypto.add(
                pypto.cast(q, pypto.DT_FP32),
                pypto.cast(attn_q_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        k = pypto.matmul(hidden_states, attn_k_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        k = pypto.cast(
            pypto.add(
                pypto.cast(k, pypto.DT_FP32),
                pypto.cast(attn_k_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        v = pypto.matmul(hidden_states, attn_v_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        v = pypto.cast(
            pypto.add(
                pypto.cast(v, pypto.DT_FP32),
                pypto.cast(attn_v_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
        q_flat = pypto.reshape(q, [b * s * n_q, d])
        q_norm = pypto.rms_norm(q_flat, attn_q_norm_w)
        pypto.set_vec_tile_shapes(min(16, b * s * n_kv), d)
        k_flat = pypto.reshape(k, [b * s * n_kv, d])
        k_norm = pypto.rms_norm(k_flat, attn_k_norm_w)

        q_4d = pypto.reshape(q_norm, [b, s, n_q, d])
        k_4d = pypto.reshape(k_norm, [b, s, n_kv, d])
        v_4d = pypto.reshape(v, [b, s, n_kv, d])
        q_embed = apply_rope_q(q_4d, cos, sin)
        k_embed = apply_rope_k(k_4d, cos, sin)

        query_out = pypto.reshape(q_embed, [b * s * n_q, d])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_rows = pypto.reshape(k_embed, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_rows = pypto.reshape(v_4d, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_cache_out = pypto.scatter_update(key_cache, -2, cache_index, key_rows)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_cache_out = pypto.scatter_update(value_cache, -2, cache_index, value_rows)
        return query_out, key_cache_out, value_cache_out

    @pypto.frontend.function
    def paged_attention_core(
        query: pypto.tensor((b * s * n_q, d), pto_dtype),
        key_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        value_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        block_table: pypto.tensor((b, block_cols), pypto.DT_INT32),
        act_seqs: pypto.tensor((b,), pypto.DT_INT32),
    ) -> pypto.tensor((b * s * n_q, d), pypto.DT_FP32):
        attention_out = pypto.tensor((b * s * n_q, d), pypto.DT_FP32)
        block_size_sym = pypto.symbolic_scalar(block_size)
        for b_idx in pypto.loop(0, b, 1, name="QWEN3_LAYER_PA_L0_B", idx_name="b_idx"):
            cur_act_seq = act_seqs[b_idx]
            cur_act_seq.as_variable()
            for s1_idx in pypto.loop(0, s, 1, name="QWEN3_LAYER_PA_L1_S1", idx_name="s1_idx"):
                cur_seq = (cur_act_seq - s + 1 + s1_idx).max(0)
                cur_seq.as_variable()
                bn_per_batch = (cur_seq + block_size - 1) // block_size
                bn_per_batch.as_variable()
                for n_idx in pypto.loop(0, n_loop, 1, name="QWEN3_LAYER_PA_L2_N", idx_name="n_idx"):
                    kv_idx = (n_idx * n_tile) // group
                    kv_idx.as_variable()
                    cur_offset = b_idx * s * n_q + s1_idx * n_q + n_idx * n_tile
                    oi_update = pypto.tensor((n_tile, d), pypto.DT_FP32)
                    li_update = pypto.tensor((n_tile, 1), pypto.DT_FP32)
                    mi_update = pypto.tensor((n_tile, 1), pypto.DT_FP32)

                    for bn in pypto.loop(
                        0,
                        bn_per_batch,
                        1,
                        name="QWEN3_LAYER_PA_L3_BN",
                        idx_name="bn",
                        unroll_List={1},
                    ):
                        valid_s2 = (cur_seq - bn * block_size).min(block_size_sym)
                        qi = pypto.view(query, [n_tile, d], [cur_offset, 0])
                        cur_block_idx = block_table[b_idx, bn]
                        cur_block_idx.as_variable()
                        kj = pypto.view(
                            key_cache,
                            [block_size, d],
                            [cur_block_idx * block_size, kv_idx * d],
                            valid_shape=[valid_s2, d],
                        )
                        vj = pypto.view(
                            value_cache,
                            [block_size, d],
                            [cur_block_idx * block_size, kv_idx * d],
                            valid_shape=[valid_s2, d],
                        )

                        pypto.set_cube_tile_shapes(
                            [c1_tile[0], c1_tile[1]],
                            [c1_tile[2], c1_tile[3]],
                            [c1_tile[4], c1_tile[5]]
                        )
                        pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_dt = pypto.cast(tilda_pij, pto_dtype)
                        tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                        if pypto.is_loop_begin(bn):
                            pypto.set_cube_tile_shapes(
                                [c2_tile[0], c2_tile[1]],
                                [c2_tile[2], c2_tile[3]],
                                [c2_tile[4], c2_tile[5]]
                            )
                            pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                            oi_tmp = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.is_loop_end(bn):
                                attention_out[cur_offset : cur_offset + n_tile, :] = pypto.div(oi_tmp, tilda_lij)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = tilda_lij
                                mi_update[:] = tilda_mij
                        else:
                            mi_new = pypto.maximum(mi_update, tilda_mij)
                            t2 = pypto.exp(pypto.sub(mi_update, mi_new))
                            t4 = pypto.exp(pypto.sub(tilda_mij, mi_new))
                            li_new = pypto.add(pypto.mul(t2, li_update), pypto.mul(t4, tilda_lij))
                            q3 = pypto.mul(oi_update, t2)
                            pypto.set_cube_tile_shapes(
                                [c2_tile[0], c2_tile[1]],
                                [c2_tile[2], c2_tile[3]],
                                [c2_tile[4], c2_tile[5]]
                            )
                            pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                            q1 = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = pypto.add(q3, pypto.mul(q1, t4))
                            if pypto.is_loop_end(bn):
                                attention_out[cur_offset : cur_offset + n_tile, :] = pypto.div(oi_tmp, li_new)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new
        return attention_out

    @pypto.frontend.function
    def attn_post_core(
        hidden_states: pypto.tensor((b * s, hidden_size), pto_dtype),
        pa_out: pypto.tensor((b * s * n_q, d), pypto.DT_FP32),
        attn_o_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_o_b: pypto.tensor((hidden_size,), pto_dtype),
    ) -> (
        pypto.tensor((b * s, hidden_size), pto_dtype),
        pypto.tensor((b * s, hidden_size), pto_dtype),
    ):
        pypto.set_vec_tile_shapes(1, hidden_size)
        context = pypto.reshape(pa_out, [b * s, hidden_size])

        pypto.set_cube_tile_shapes(
            [o_mm_tile[0], o_mm_tile[0]],
            [o_mm_tile[1], o_mm_tile[1]],
            [o_mm_tile[2], o_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, hidden_size])
        pypto.set_vec_tile_shapes(1, hidden_size)
        attn_o_w_fp32 = pypto.cast(attn_o_w, pypto.DT_FP32)
        o_proj_fp32 = pypto.matmul(context, attn_o_w_fp32, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, hidden_size)
        o_proj_fp32 = pypto.add(o_proj_fp32, pypto.cast(attn_o_b, pypto.DT_FP32))
        pypto.set_vec_tile_shapes(1, hidden_size)
        o_proj = pypto.cast(o_proj_fp32, pto_dtype)
        pypto.set_vec_tile_shapes(1, hidden_size)
        h1 = pypto.cast(
            pypto.add(
                pypto.cast(hidden_states, pypto.DT_FP32),
                pypto.cast(o_proj, pypto.DT_FP32)
            ),
            pto_dtype
        )
        pypto.set_vec_tile_shapes(1, hidden_size)
        norm2 = pypto.rms_norm(h1)
        return h1, norm2

    @pypto.frontend.jit(
        runtime_options={"run_mode": run_mode},
    )
    def qwen3_layer_frontend_jit(
        hidden_states: pypto.tensor((b * s, hidden_size), pto_dtype),
        attn_q_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_q_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_k_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_k_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_v_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_v_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_o_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_o_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_q_norm_w: pypto.tensor((d,), pto_dtype),
        attn_k_norm_w: pypto.tensor((d,), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
        cache_index: pypto.tensor((b, s), pypto.DT_INT32),
        key_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        value_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        block_table: pypto.tensor((b, block_cols), pypto.DT_INT32),
        act_seqs: pypto.tensor((b,), pypto.DT_INT32),
        gate_w: pypto.tensor((hidden_size, inter), pto_dtype),
        up_w: pypto.tensor((hidden_size, inter), pto_dtype),
        down_w: pypto.tensor((inter, hidden_size), pto_dtype),
    ) -> pypto.tensor((b * s, hidden_size), pto_dtype):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_vec_tile_shapes(1, hidden_size)
        norm1 = pypto.rms_norm(hidden_states)

        pypto.set_cube_tile_shapes(
            [q_mm_tile[0], q_mm_tile[0]],
            [q_mm_tile[1], q_mm_tile[1]],
            [q_mm_tile[2], q_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, hidden_size])
        q = pypto.matmul(hidden_states, attn_q_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, hidden_size)
        q = pypto.cast(
            pypto.add(
                pypto.cast(q, pypto.DT_FP32),
                pypto.cast(attn_q_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        k = pypto.matmul(norm1, attn_k_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        k = pypto.cast(
            pypto.add(
                pypto.cast(k, pypto.DT_FP32),
                pypto.cast(attn_k_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_cube_tile_shapes(
            [kv_mm_tile[0], kv_mm_tile[0]],
            [kv_mm_tile[1], kv_mm_tile[1]],
            [kv_mm_tile[2], kv_mm_tile[2]]
        )
        pypto.set_matrix_size([b * s, hidden_size, kv_hidden])
        v = pypto.matmul(hidden_states, attn_v_w, pto_dtype)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        v = pypto.cast(
            pypto.add(
                pypto.cast(v, pypto.DT_FP32),
                pypto.cast(attn_v_b, pypto.DT_FP32)
            ),
            pto_dtype
        )

        pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
        q_flat = pypto.reshape(q, [b * s * n_q, d])
        q_norm = pypto.rms_norm(q_flat, attn_q_norm_w)
        pypto.set_vec_tile_shapes(min(16, b * s * n_kv), d)
        k_flat = pypto.reshape(k, [b * s * n_kv, d])
        k_norm = pypto.rms_norm(k_flat, attn_k_norm_w)

        q_4d = pypto.reshape(q_norm, [b, s, n_q, d])
        k_4d = pypto.reshape(k_norm, [b, s, n_kv, d])
        v_4d = pypto.reshape(v, [b, s, n_kv, d])
        q_embed = apply_rope_q(q_4d, cos, sin)
        k_embed = apply_rope_k(k_4d, cos, sin)

        query_out = pypto.reshape(q_embed, [b * s * n_q, d])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_rows = pypto.reshape(k_embed, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_rows = pypto.reshape(v_4d, [b * s, kv_hidden])
        pypto.set_vec_tile_shapes(1, kv_hidden)
        key_cache_tmp = pypto.scatter_update(key_cache, -2, cache_index, key_rows)
        pypto.set_vec_tile_shapes(1, kv_hidden)
        value_cache_tmp = pypto.scatter_update(value_cache, -2, cache_index, value_rows)

        pa_out = paged_attention_core(query_out, key_cache_tmp, value_cache_tmp, block_table, act_seqs)
        h1, norm2 = attn_post_core(hidden_states, pa_out, attn_o_w, attn_o_b)
        mlp_out = mlp_core(norm2, gate_w, up_w, down_w)
        pypto.set_vec_tile_shapes(1, hidden_size)
        return pypto.cast(pypto.add(pypto.cast(h1, pypto.DT_FP32), pypto.cast(mlp_out, pypto.DT_FP32)), pto_dtype)

    @pypto.frontend.jit(
        runtime_options={
            "run_mode": run_mode,
            "stitch_function_num_initial": 1,
            "stitch_function_num_step": 1,
            "stitch_function_inner_memory": 1,
            "stitch_function_outcast_memory": 1,
            "stitch_function_size": 2048,
            "valid_shape_optimize": 1,
        },
    )
    def qwen3_layer_frontend_graph_jit(
        hidden_states: pypto.tensor((b * s, hidden_size), pto_dtype),
        attn_q_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_q_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_k_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_k_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_v_w: pypto.tensor((hidden_size, kv_hidden), pto_dtype),
        attn_v_b: pypto.tensor((kv_hidden,), pto_dtype),
        attn_o_w: pypto.tensor((hidden_size, hidden_size), pto_dtype),
        attn_o_b: pypto.tensor((hidden_size,), pto_dtype),
        attn_q_norm_w: pypto.tensor((d,), pto_dtype),
        attn_k_norm_w: pypto.tensor((d,), pto_dtype),
        cos: pypto.tensor((b, s, d), pto_dtype),
        sin: pypto.tensor((b, s, d), pto_dtype),
        cache_index: pypto.tensor((b, s), pypto.DT_INT32),
        key_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        value_cache: pypto.tensor((cache_rows, kv_hidden), pto_dtype),
        block_table: pypto.tensor((b, block_cols), pypto.DT_INT32),
        act_seqs: pypto.tensor((b,), pypto.DT_INT32),
        gate_w: pypto.tensor((hidden_size, inter), pto_dtype),
        up_w: pypto.tensor((hidden_size, inter), pto_dtype),
        down_w: pypto.tensor((inter, hidden_size), pto_dtype),
    ) -> pypto.tensor((b * s, hidden_size), pto_dtype):
        layer_out = pypto.tensor((b * s, hidden_size), pto_dtype, "qwen3_layer_frontend_out")
        pypto.set_vec_tile_shapes(1, hidden_size)
        qwen3_layer_graph(
            params,
            hidden_states,
            attn_q_w,
            attn_q_b,
            attn_k_w,
            attn_k_b,
            attn_v_w,
            attn_v_b,
            attn_o_w,
            attn_o_b,
            attn_q_norm_w,
            attn_k_norm_w,
            cos,
            sin,
            cache_index,
            key_cache,
            value_cache,
            block_table,
            act_seqs,
            gate_w,
            up_w,
            down_w,
            layer_out,
        )
        return layer_out

    return qwen3_layer_frontend_graph_jit


def qwen3_paged_attention_prolog_graph(
    params,
    hidden_states,
    attn_q_w,
    attn_q_b,
    attn_k_w,
    attn_k_b,
    attn_v_w,
    attn_v_b,
    attn_q_norm_w,
    attn_k_norm_w,
    cos,
    sin,
    cache_index,
    key_cache,
    value_cache,
    query_out,
    key_cache_out,
    value_cache_out,
):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    hidden_size = n_q * d
    kv_hidden = n_kv * d
    dtype = hidden_states.dtype

    _set_matmul_tile(b * s, hidden_size, hidden_size, dtype)
    q = pypto.matmul(hidden_states, attn_q_w, dtype)
    q = pypto.cast(pypto.add(pypto.cast(q, pypto.DT_FP32), pypto.cast(attn_q_b, pypto.DT_FP32)), dtype)

    _set_matmul_tile(b * s, hidden_size, kv_hidden, dtype)
    k = pypto.matmul(hidden_states, attn_k_w, dtype)
    k = pypto.cast(pypto.add(pypto.cast(k, pypto.DT_FP32), pypto.cast(attn_k_b, pypto.DT_FP32)), dtype)

    _set_matmul_tile(b * s, hidden_size, kv_hidden, dtype)
    v = pypto.matmul(hidden_states, attn_v_w, dtype)
    v = pypto.cast(pypto.add(pypto.cast(v, pypto.DT_FP32), pypto.cast(attn_v_b, pypto.DT_FP32)), dtype)

    pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
    q = pypto.reshape(q, [b * s * n_q, d])
    q = pypto.rms_norm(q, attn_q_norm_w)

    pypto.set_vec_tile_shapes(min(16, b * s * n_kv), d)
    k = pypto.reshape(k, [b * s * n_kv, d])
    k = pypto.rms_norm(k, attn_k_norm_w)

    tile_last = max(16, min(64, d))
    pypto.set_vec_tile_shapes(1, 1, min(8, n_q), tile_last)
    q_4d = pypto.reshape(q, [b, s, n_q, d])
    pypto.set_vec_tile_shapes(1, 1, min(8, n_kv), tile_last)
    k_4d = pypto.reshape(k, [b, s, n_kv, d])
    pypto.set_vec_tile_shapes(1, 1, min(8, n_kv), tile_last)
    v_4d = pypto.reshape(v, [b, s, n_kv, d])

    q_embed = _apply_rope_graph(q_4d, cos, sin)
    k_embed = _apply_rope_graph(k_4d, cos, sin)

    pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
    query_out[:] = pypto.reshape(q_embed, [b * s * n_q, d])
    pypto.set_vec_tile_shapes(min(16, b * s), kv_hidden)
    key_rows = pypto.reshape(k_embed, [b * s, kv_hidden])
    pypto.set_vec_tile_shapes(min(16, b * s), kv_hidden)
    value_rows = pypto.reshape(v_4d, [b * s, kv_hidden])

    key_cache_out[:] = pypto.scatter_update(key_cache, -2, cache_index, key_rows)
    value_cache_out[:] = pypto.scatter_update(value_cache, -2, cache_index, value_rows)


def qwen3_paged_attention_graph(params, query, key_cache, value_cache, block_table, act_seqs, attention_out):
    n_q = params["n"]
    n_kv = params["n_kv"]
    block_size = params["block_size"]
    tile_config = params["pa_tile_config"]
    max_unroll_times = params["max_unroll_times"]
    group = n_q // n_kv
    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    dtype = query.dtype
    d = query.shape[1]
    batch_size = block_table.shape[0]
    s1_size = query.shape[0] // batch_size // n_q
    n_loop = n_q // n_tile
    block_size_sym = pypto.symbolic_scalar(block_size)
    softmax_scale = float(1.0 / math.sqrt(params["d"]))

    for b_idx in pypto.loop(0, batch_size, 1, name="QWEN3_PA_L0_B", idx_name="b_idx"):
        cur_act_seq = act_seqs[b_idx]
        cur_act_seq.as_variable()
        for s1_idx in pypto.loop(0, s1_size, 1, name="QWEN3_PA_L1_S1", idx_name="s1_idx"):
            cur_seq = (cur_act_seq - s1_size + 1 + s1_idx).max(0)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + block_size - 1) // block_size
            bn_per_batch.as_variable()
            for n_idx in pypto.loop(0, n_loop, 1, name="QWEN3_PA_L2_N", idx_name="n_idx"):
                kv_idx = (n_idx * n_tile) // group
                kv_idx.as_variable()
                cur_offset = b_idx * s1_size * n_q + s1_idx * n_q + n_idx * n_tile
                oi_offset = [cur_offset, 0]
                oi_update = pypto.tensor([n_tile, d], pypto.DT_FP32, "qwen3_pa_oi")
                li_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "qwen3_pa_li")
                mi_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "qwen3_pa_mi")

                for bn in pypto.loop(
                    0,
                    bn_per_batch,
                    1,
                    name="QWEN3_PA_L3_BN",
                    idx_name="bn",
                    unroll_List={max_unroll_times},
                ):
                    valid_s2 = (cur_seq - bn * block_size).min(block_size_sym)
                    qi = pypto.view(query, [n_tile, d], [cur_offset, 0])
                    cur_block_idx = block_table[b_idx, bn]
                    cur_block_idx.as_variable()
                    kj = pypto.view(
                        key_cache,
                        [block_size, d],
                        [cur_block_idx * block_size, kv_idx * d],
                        valid_shape=[valid_s2, d],
                    )
                    vj = pypto.view(
                        value_cache,
                        [block_size, d],
                        [cur_block_idx * block_size, kv_idx * d],
                        valid_shape=[valid_s2, d],
                    )

                    pypto.set_cube_tile_shapes(
                        [c1_tile[0], c1_tile[1]],
                        [c1_tile[2], c1_tile[3]],
                        [c1_tile[4], c1_tile[5]],
                    )
                    pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                    sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    sij_scale = pypto.mul(sij, softmax_scale)
                    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                    tsub = pypto.sub(sij_scale, tilda_mij)
                    tilda_pij = pypto.exp(tsub)
                    tilda_pij_dt = pypto.cast(tilda_pij, dtype)
                    tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                    if pypto.cond(pypto.is_loop_begin(bn)):
                        pypto.set_cube_tile_shapes(
                            [c2_tile[0], c2_tile[1]],
                            [c2_tile[2], c2_tile[3]],
                            [c2_tile[4], c2_tile[5]],
                        )
                        pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                        oi_tmp = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        if pypto.cond(pypto.is_loop_end(bn)):
                            pypto.assemble(pypto.div(oi_tmp, tilda_lij), oi_offset, attention_out)
                        else:
                            oi_update[:] = oi_tmp
                            li_update[:] = tilda_lij
                            mi_update[:] = tilda_mij
                    else:
                        mi_new = pypto.maximum(mi_update, tilda_mij)
                        t2 = pypto.exp(pypto.sub(mi_update, mi_new))
                        t4 = pypto.exp(pypto.sub(tilda_mij, mi_new))
                        li_new = pypto.add(pypto.mul(t2, li_update), pypto.mul(t4, tilda_lij))
                        q3 = pypto.mul(oi_update, t2)
                        pypto.set_cube_tile_shapes(
                            [c2_tile[0], c2_tile[1]],
                            [c2_tile[2], c2_tile[3]],
                            [c2_tile[4], c2_tile[5]],
                        )
                        pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
                        q1 = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_tmp = pypto.add(q3, pypto.mul(q1, t4))
                        if pypto.cond(pypto.is_loop_end(bn)):
                            pypto.assemble(pypto.div(oi_tmp, li_new), oi_offset, attention_out)
                        else:
                            oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


def qwen3_layer_graph(
    params,
    hidden_states,
    attn_q_w,
    attn_q_b,
    attn_k_w,
    attn_k_b,
    attn_v_w,
    attn_v_b,
    attn_o_w,
    attn_o_b,
    attn_q_norm_w,
    attn_k_norm_w,
    cos,
    sin,
    cache_index,
    key_cache,
    value_cache,
    block_table,
    act_seqs,
    gate_w,
    up_w,
    down_w,
    layer_out,
):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    hidden_size = n_q * d
    dtype = hidden_states.dtype
    kv_hidden = n_kv * d

    norm1 = pypto.rms_norm(hidden_states)
    query_out = pypto.tensor([b * s * n_q, d], dtype, "qwen3_layer_query")
    key_cache_tmp = pypto.tensor(list(key_cache.shape), key_cache.dtype, "qwen3_layer_key_cache")
    value_cache_tmp = pypto.tensor(list(value_cache.shape), value_cache.dtype, "qwen3_layer_value_cache")
    qwen3_paged_attention_prolog_graph(
        params,
        norm1,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_q_norm_w,
        attn_k_norm_w,
        cos,
        sin,
        cache_index,
        key_cache,
        value_cache,
        query_out,
        key_cache_tmp,
        value_cache_tmp,
    )

    pa_out = pypto.tensor([b * s * n_q, d], pypto.DT_FP32, "qwen3_layer_pa")
    qwen3_paged_attention_graph(params, query_out, key_cache_tmp, value_cache_tmp, block_table, act_seqs, pa_out)

    context = pypto.reshape(pa_out, [b * s, hidden_size])
    _set_matmul_tile(b * s, hidden_size, hidden_size, pypto.DT_FP32)
    o_proj_fp32 = pypto.matmul(context, pypto.cast(attn_o_w, pypto.DT_FP32), pypto.DT_FP32)
    o_proj_fp32 = pypto.add(o_proj_fp32, pypto.cast(attn_o_b, pypto.DT_FP32))
    o_proj = pypto.cast(o_proj_fp32, dtype)
    h1 = pypto.cast(pypto.add(pypto.cast(hidden_states, pypto.DT_FP32), pypto.cast(o_proj, pypto.DT_FP32)), dtype)
    norm2 = pypto.rms_norm(h1)

    mlp_tmp = pypto.tensor([b * s, hidden_size], dtype, "qwen3_layer_mlp")
    mlp_params = {"b": b, "s": s, "h": hidden_size, "inter": params["inter"]}
    qwen3_mlp_graph(mlp_params, norm2, gate_w, up_w, down_w, mlp_tmp)
    layer_out[:] = pypto.cast(pypto.add(pypto.cast(h1, pypto.DT_FP32), pypto.cast(mlp_tmp, pypto.DT_FP32)), dtype)


def qwen3_rmsnorm_graph(params, hidden_states, norm_out):
    norm_out[:] = pypto.rms_norm(hidden_states)


def qwen3_layer_post_graph(params, hidden_states, pa_out, attn_o_w, attn_o_b, gate_w, up_w, down_w, layer_out):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    d = params["d"]
    hidden_size = n_q * d
    dtype = hidden_states.dtype

    context = pypto.reshape(pa_out, [b * s, hidden_size])
    _set_matmul_tile(b * s, hidden_size, hidden_size, pypto.DT_FP32)
    o_proj_fp32 = pypto.matmul(context, pypto.cast(attn_o_w, pypto.DT_FP32), pypto.DT_FP32)
    o_proj_fp32 = pypto.add(o_proj_fp32, pypto.cast(attn_o_b, pypto.DT_FP32))
    o_proj = pypto.cast(o_proj_fp32, dtype)
    h1 = pypto.cast(pypto.add(pypto.cast(hidden_states, pypto.DT_FP32), pypto.cast(o_proj, pypto.DT_FP32)), dtype)
    norm2 = pypto.rms_norm(h1)

    mlp_tmp = pypto.tensor([b * s, hidden_size], dtype, "qwen3_layer_post_mlp")
    mlp_params = {"b": b, "s": s, "h": hidden_size, "inter": params["inter"]}
    qwen3_mlp_graph(mlp_params, norm2, gate_w, up_w, down_w, mlp_tmp)
    layer_out[:] = pypto.cast(pypto.add(pypto.cast(h1, pypto.DT_FP32), pypto.cast(mlp_tmp, pypto.DT_FP32)), dtype)


class CountBasedCompareTestBuilder(TestBuilder):
    def _assert_count_based_close(
        self, expected: torch.Tensor, actual: torch.Tensor, eps: float, zero_count_threshold: int = 1000
    ):
        threshold = int(expected.numel() * eps)
        err_count = 0
        zero_count = 0
        chunk = 1_000_000
        exp_flat = expected.reshape(-1)
        act_flat = actual.reshape(-1)
        for start in range(0, exp_flat.numel(), chunk):
            end = min(start + chunk, exp_flat.numel())
            exp_chunk = exp_flat[start:end].to(torch.float32)
            act_chunk = act_flat[start:end].to(torch.float32)
            diff = (exp_chunk - act_chunk).abs()
            rel = torch.where(
                exp_chunk != 0,
                diff / exp_chunk.abs(),
                torch.where(diff == 0, torch.zeros_like(diff), torch.full_like(diff, float("inf"))),
            )
            err_count += torch.logical_and(diff > eps, rel > eps).sum().item()
            zero_count += torch.logical_and(act_chunk.abs() <= 1e-6, exp_chunk.abs() > 1e-6).sum().item()
            if err_count > threshold or zero_count > zero_count_threshold:
                raise AssertionError(
                    f"Count-based compare failed: err_count={err_count}, threshold={threshold}, "
                    f"zero_count={zero_count}, zero_threshold={zero_count_threshold}"
                )

    def run_pto(self, kernel, tiling, on_board: bool = True):
        if on_board:
            torch.npu.set_device(self.device_id)

        pypto.set_vec_tile_shapes(tiling, tiling)
        with pypto.function("MAIN", *self.input_pto_list, *self.output_pto_list) as rlf:
            for _ in rlf:
                kernel(self.params, *self.input_pto_list, *self.output_pto_list)
            del rlf

        if on_board:
            pto_input_data = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(self.input_data_list)]
            pto_output_data = [
                pypto.from_torch(tensor, f"OUT_{idx}")
                for idx, tensor in enumerate(self.output_data_list)
            ]
            pypto.runtime._device_run_once_data_from_host(*pto_input_data, *pto_output_data)
            for idx in range(len(self.golden_output)):
                self._assert_count_based_close(
                    self.golden_output[idx].cpu(), self.output_data_list[idx].cpu(), self.atol_value
                )

    def _assert_new_swimlane_output(self, before_dirs):
        after_dirs = _collect_output_dirs()
        new_dirs = sorted(after_dirs - before_dirs)
        if not new_dirs:
            raise AssertionError("No new output/output_* directory created for runtime_debug_mode=1.")
        latest_dir = max(new_dirs, key=lambda path: path.stat().st_mtime_ns)
        for filename in ("merged_swimlane.json", "program.json", "dyn_topo.txt"):
            path = latest_dir / filename
            if not path.exists():
                raise AssertionError(f"Missing debug artifact: {path}")
            if path.stat().st_size <= 0:
                raise AssertionError(f"Empty debug artifact: {path}")
        return latest_dir

    def _run_frontend_jit_kernel(self, kernel, inputs, on_board: bool = True):
        before_dirs = _collect_output_dirs() if on_board else set()
        if on_board:
            torch.npu.set_device(self.device_id)
            device = torch.device(f"npu:{self.device_id}")
        else:
            device = torch.device("cpu")

        device_inputs = [tensor.to(device).contiguous() for tensor in inputs]
        outputs = kernel(*device_inputs)

        if on_board:
            torch.npu.synchronize()
            self._assert_new_swimlane_output(before_dirs)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return outputs


class Qwen3PagedAttentionPrologRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_paged_attention_prolog_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        b = self.params["b"]
        s = self.params["s"]
        n_q = self.params["n"]
        n_kv = self.params["n_kv"]
        d = self.params["d"]
        block_size = self.params["block_size"]
        dtype = self.params["dtype"]
        hidden_size = n_q * d
        kv_hidden = n_kv * d

        hidden_states = _gen_uniform_data([b * s, hidden_size], -1, 1, dtype)
        attn_q_w = _gen_uniform_data([hidden_size, hidden_size], -1, 1, dtype)
        attn_q_b = _gen_uniform_data([hidden_size], -1, 1, dtype)
        attn_k_w = _gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype)
        attn_k_b = _gen_uniform_data([kv_hidden], -1, 1, dtype)
        attn_v_w = _gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype)
        attn_v_b = _gen_uniform_data([kv_hidden], -1, 1, dtype)
        attn_q_norm_w = _gen_uniform_data([d], -1, 1, dtype)
        attn_k_norm_w = _gen_uniform_data([d], -1, 1, dtype)

        if b == 1:
            act_seq_list = [block_size * 2 + s]
        else:
            act_seq_list = [block_size * 2 + s, block_size * 2 - 1 + s] + [block_size * 2 + s] * (b - 2)
        cache_position = _build_cache_position_from_act_seq_list(act_seq_list, s)
        cos, sin = _build_rope_from_cache_position(cache_position, d, dtype)
        block_table, _, block_num = _build_block_table(act_seq_list, block_size)

        key_cache = _gen_uniform_data([block_num * block_size, kv_hidden], -1, 1, dtype)
        value_cache = _gen_uniform_data([block_num * block_size, kv_hidden], -1, 1, dtype)
        cache_index = _build_cache_index(cache_position, block_table, block_size)
        self._cache_index = cache_index

        inputs = _convert_tensors_contiguous(
            [
                hidden_states,
                attn_q_w,
                attn_q_b,
                attn_k_w,
                attn_k_b,
                attn_v_w,
                attn_v_b,
                attn_q_norm_w,
                attn_k_norm_w,
                cos,
                sin,
                cache_index,
                key_cache,
                value_cache,
            ]
        )
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-3, atol=1e-3)
        return inputs

    def run_pto(self, kernel, tiling, on_board: bool = True):
        if on_board:
            torch.npu.set_device(self.device_id)

        pypto.set_vec_tile_shapes(tiling, tiling)
        with pypto.function("MAIN", *self.input_pto_list, *self.output_pto_list) as rlf:
            for _ in rlf:
                kernel(self.params, *self.input_pto_list, *self.output_pto_list)
            del rlf

        if on_board:
            pto_input_data = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(self.input_data_list)]
            pto_output_data = [
                pypto.from_torch(tensor, f"OUT_{idx}")
                for idx, tensor in enumerate(self.output_data_list)
            ]
            pypto.runtime._device_run_once_data_from_host(*pto_input_data, *pto_output_data)

            self._assert_count_based_close(self.golden_output[0].cpu(), self.output_data_list[0].cpu(), self.atol_value)

            row_index = self._cache_index.reshape(-1).to(torch.int64)
            golden_k_rows = self.golden_output[1].index_select(0, row_index)
            actual_k_rows = self.output_data_list[1].index_select(0, row_index)
            self._assert_count_based_close(golden_k_rows.cpu(), actual_k_rows.cpu(), self.atol_value)

            golden_v_rows = self.golden_output[2].index_select(0, row_index)
            actual_v_rows = self.output_data_list[2].index_select(0, row_index)
            self._assert_count_based_close(golden_v_rows.cpu(), actual_v_rows.cpu(), self.atol_value)

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3PagedAttentionPrologRunner directly uses a frontend.jit kernel.")

        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None, None, None))
        prolog_kernel = build_qwen3_paged_attention_prolog_frontend_jit(
            self.params,
            pypto.RunMode.NPU if on_board else pypto.RunMode.SIM,
        )
        query_out, key_cache_out, value_cache_out = self._run_frontend_jit_kernel(prolog_kernel, self.inputs, on_board)
        query_out = query_out.cpu()
        key_cache_out = key_cache_out.cpu()
        value_cache_out = value_cache_out.cpu()

        self._assert_count_based_close(self.golden_output[0].cpu(), query_out, self.atol_value)
        row_index = self._cache_index.reshape(-1).to(torch.int64)
        golden_k_rows = self.golden_output[1].index_select(0, row_index)
        actual_k_rows = key_cache_out.index_select(0, row_index)
        self._assert_count_based_close(golden_k_rows.cpu(), actual_k_rows, self.atol_value)

        golden_v_rows = self.golden_output[2].index_select(0, row_index)
        actual_v_rows = value_cache_out.index_select(0, row_index)
        self._assert_count_based_close(golden_v_rows.cpu(), actual_v_rows, self.atol_value)

    def golden(
        self,
        params,
        hidden_states,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_q_norm_w,
        attn_k_norm_w,
        cos,
        sin,
        cache_index,
        key_cache,
        value_cache,
        query_out=None,
        key_cache_out=None,
        value_cache_out=None,
    ):
        b = params["b"]
        s = params["s"]
        n_q = params["n"]
        n_kv = params["n_kv"]
        d = params["d"]
        kv_hidden = n_kv * d

        q = torch.matmul(hidden_states.to(torch.float32), attn_q_w.to(torch.float32)).to(hidden_states.dtype)
        q = (q.to(torch.float32) + attn_q_b.to(torch.float32)).to(hidden_states.dtype)
        k = torch.matmul(hidden_states.to(torch.float32), attn_k_w.to(torch.float32)).to(hidden_states.dtype)
        k = (k.to(torch.float32) + attn_k_b.to(torch.float32)).to(hidden_states.dtype)
        v = torch.matmul(hidden_states.to(torch.float32), attn_v_w.to(torch.float32)).to(hidden_states.dtype)
        v = (v.to(torch.float32) + attn_v_b.to(torch.float32)).to(hidden_states.dtype)

        q = _rmsnorm_torch(q.reshape(b * s * n_q, d), attn_q_norm_w).reshape(b, s, n_q, d)
        k = _rmsnorm_torch(k.reshape(b * s * n_kv, d), attn_k_norm_w).reshape(b, s, n_kv, d)
        v = v.reshape(b, s, n_kv, d)

        q_embed = _apply_rope_torch(q, cos, sin)
        k_embed = _apply_rope_torch(k, cos, sin)
        query_out = q_embed.reshape(b * s * n_q, d)

        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()
        key_rows = k_embed.reshape(b * s, kv_hidden)
        value_rows = v.reshape(b * s, kv_hidden)
        for batch_idx in range(b):
            for token_idx in range(s):
                row = int(cache_index[batch_idx, token_idx].item())
                src = batch_idx * s + token_idx
                key_cache_out[row, :] = key_rows[src, :]
                value_cache_out[row, :] = value_rows[src, :]

        return query_out, key_cache_out, value_cache_out


class Qwen3PagedAttentionRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_paged_attention_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        b = self.params["b"]
        s = self.params["s"]
        n_q = self.params["n"]
        n_kv = self.params["n_kv"]
        d = self.params["d"]
        block_size = self.params["block_size"]
        dtype = self.params["dtype"]
        skv = self.params["skv"]

        actual_seq_list = [skv] * b if isinstance(skv, int) else [int(x) for x in skv]
        act_seqs = torch.tensor(actual_seq_list, dtype=torch.int32)
        max_seq = int(max(actual_seq_list))
        q_4d = _gen_uniform_data([b, s, n_q, d], -1, 1, dtype)
        k_bsnd = _gen_uniform_data([b, max_seq, n_kv, d], -1, 1, dtype)
        v_bsnd = _gen_uniform_data([b, max_seq, n_kv, d], -1, 1, dtype)
        block_table, _, key_cache, value_cache = _build_cache_from_bsnd(k_bsnd, v_bsnd, actual_seq_list, block_size)
        query = q_4d.reshape(b * s * n_q, d)

        self._q_4d = q_4d
        self._k_bsnd = k_bsnd
        self._v_bsnd = v_bsnd

        inputs = _convert_tensors_contiguous([query, key_cache, value_cache, block_table, act_seqs])
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-3, atol=1e-3)
        return inputs

    def golden(
        self,
        params,
        query,
        key_cache,
        value_cache,
        block_table,
        act_seqs,
        paged_attention_out=None,
    ):
        b = params["b"]
        s = params["s"]
        n_q = params["n"]
        n_kv = params["n_kv"]
        d = params["d"]
        block_size = params["block_size"]
        group = n_q // n_kv
        q_4d = self._q_4d.to(torch.float32)
        out = torch.zeros([b, s, n_q, d], dtype=torch.float32)
        scale = 1.0 / math.sqrt(d)
        for batch_idx in range(b):
            cur_block_table = block_table[batch_idx]
            act_seq = int(act_seqs[batch_idx].item())
            for token_idx in range(s):
                cur_seq = max(act_seq - s + 1 + token_idx, 0)
                for kv_idx in range(n_kv):
                    q_group = q_4d[batch_idx, token_idx, kv_idx * group : (kv_idx + 1) * group, :]
                    k_cur = _build_pa_rows_from_cache(
                        key_cache, cur_block_table, block_size, cur_seq, kv_idx, d
                    ).to(torch.float32)
                    v_cur = _build_pa_rows_from_cache(
                        value_cache, cur_block_table, block_size, cur_seq, kv_idx, d
                    ).to(torch.float32)
                    scores = torch.matmul(q_group, k_cur.transpose(-1, -2)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    out_group = torch.matmul(probs, v_cur)
                    out[batch_idx, token_idx, kv_idx * group : (kv_idx + 1) * group, :] = out_group
        return (out.reshape(b * s * n_q, d),)

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3PagedAttentionRunner directly uses a frontend.jit kernel.")

        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        attention_kernel = build_qwen3_paged_attention_frontend_jit(
            self.params,
            pypto.RunMode.NPU if on_board else pypto.RunMode.SIM,
        )
        (attention_out,) = self._run_frontend_jit_kernel(attention_kernel, self.inputs, on_board)
        self._assert_count_based_close(self.golden_output[0].cpu(), attention_out.cpu(), self.atol_value)


class Qwen3MLPRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_mlp_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        b = self.params["b"]
        s = self.params["s"]
        h = self.params["h"]
        inter = self.params["inter"]
        dtype = self.params["dtype"]
        hidden_states = _gen_uniform_data([b * s, h], -1, 1, dtype)
        gate_w = _gen_uniform_data([h, inter], -1, 1, dtype)
        up_w = _gen_uniform_data([h, inter], -1, 1, dtype)
        down_w = _gen_uniform_data([inter, h], -1, 1, dtype)
        inputs = _convert_tensors_contiguous([hidden_states, gate_w, up_w, down_w])
        self.setup_inputs(*inputs)
        self.set_tol(rtol=8e-3, atol=8e-3)
        return inputs

    def golden(self, params, hidden_states, gate_w, up_w, down_w, mlp_out=None):
        gate = torch.matmul(hidden_states.to(torch.float32), gate_w.to(torch.float32)).to(hidden_states.dtype)
        up = torch.matmul(hidden_states.to(torch.float32), up_w.to(torch.float32)).to(hidden_states.dtype)
        gate_act = gate * torch.sigmoid(gate.to(torch.float32)).to(hidden_states.dtype)
        inter = (gate_act.to(torch.float32) * up.to(torch.float32)).to(hidden_states.dtype)
        out = torch.matmul(inter.to(torch.float32), down_w.to(torch.float32)).to(hidden_states.dtype)
        return (out,)

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3MLPRunner directly uses a frontend.jit kernel.")

        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        mlp_kernel = build_qwen3_mlp_frontend_jit(
            self.params,
            pypto.RunMode.NPU if on_board else pypto.RunMode.SIM,
        )

        if on_board:
            torch.npu.set_device(self.device_id)
        (actual,) = self._run_frontend_jit_kernel(mlp_kernel, self.inputs, on_board)
        self._assert_count_based_close(self.golden_output[0].cpu(), actual.cpu(), self.atol_value)


class Qwen3LayerRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_layer_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        b = self.params["b"]
        s = self.params["s"]
        n_q = self.params["n"]
        n_kv = self.params["n_kv"]
        d = self.params["d"]
        block_size = self.params["block_size"]
        inter = self.params["inter"]
        skv = self.params["skv"]
        dtype = self.params["dtype"]
        hidden_size = n_q * d
        kv_hidden = n_kv * d

        hidden_states = _gen_uniform_data([b * s, hidden_size], -1, 1, dtype)
        attn_q_w = _gen_uniform_data([hidden_size, hidden_size], -1, 1, dtype)
        attn_q_b = _gen_uniform_data([hidden_size], -1, 1, dtype)
        attn_k_w = _gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype)
        attn_k_b = _gen_uniform_data([kv_hidden], -1, 1, dtype)
        attn_v_w = _gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype)
        attn_v_b = _gen_uniform_data([kv_hidden], -1, 1, dtype)
        attn_o_w = _gen_uniform_data([hidden_size, hidden_size], -1, 1, dtype)
        attn_o_b = _gen_uniform_data([hidden_size], -1, 1, dtype)
        attn_q_norm_w = _gen_uniform_data([d], -1, 1, dtype)
        attn_k_norm_w = _gen_uniform_data([d], -1, 1, dtype)
        cos = _gen_uniform_data([b, s, d], -1, 1, dtype)
        sin = _gen_uniform_data([b, s, d], -1, 1, dtype)

        cache_index = torch.zeros([b, s], dtype=torch.int32)
        for batch_idx in range(b):
            for token_idx in range(s):
                cache_index[batch_idx, token_idx] = skv - s + token_idx + batch_idx * block_size

        act_seq_list = [skv] * b
        act_seqs = torch.tensor(act_seq_list, dtype=torch.int32)
        self._act_seq_list = act_seq_list
        group = n_q // n_kv
        self._group = group

        block_table, _, block_num = _build_block_table(act_seq_list, block_size)
        k_bsnd = _gen_uniform_data([b, skv, n_kv, d], -1, 1, dtype)
        v_bsnd = _gen_uniform_data([b, skv, n_kv, d], -1, 1, dtype)
        self._k_bsnd = k_bsnd
        self._v_bsnd = v_bsnd
        key_cache = torch.zeros([block_num * block_size, kv_hidden], dtype=dtype)
        value_cache = torch.zeros([block_num * block_size, kv_hidden], dtype=dtype)
        for batch_idx, seq_len in enumerate(act_seq_list):
            block_num_in_batch = int(math.ceil(seq_len / block_size))
            for block_idx in range(block_num_in_batch):
                global_block_id = int(block_table[batch_idx, block_idx].item())
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if end > start:
                    key_cache[global_block_id * block_size : global_block_id * block_size + (end - start), :] = (
                        k_bsnd[batch_idx, start:end, :, :].reshape(end - start, kv_hidden)
                    )
                    value_cache[
                        global_block_id * block_size : global_block_id * block_size + (end - start), :
                    ] = v_bsnd[batch_idx, start:end, :, :].reshape(end - start, kv_hidden)

        gate_w = _gen_uniform_data([hidden_size, inter], -1, 1, dtype)
        up_w = _gen_uniform_data([hidden_size, inter], -1, 1, dtype)
        down_w = _gen_uniform_data([inter, hidden_size], -1, 1, dtype)

        inputs = _convert_tensors_contiguous(
            [
                hidden_states,
                attn_q_w,
                attn_q_b,
                attn_k_w,
                attn_k_b,
                attn_v_w,
                attn_v_b,
                attn_o_w,
                attn_o_b,
                attn_q_norm_w,
                attn_k_norm_w,
                cos,
                sin,
                cache_index,
                key_cache,
                value_cache,
                block_table,
                act_seqs,
                gate_w,
                up_w,
                down_w,
            ]
        )
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-1, atol=1e-1)
        return inputs

    def _run_kernel_once(self, name, kernel, inputs, outputs, on_board: bool = True):
        input_pto_list = []
        output_pto_list = []
        for idx, item in enumerate(inputs):
            dtype = self.dtype_conversion(str(item.dtype))
            input_pto_list.append(pypto.tensor(item.shape, dtype, f"{name}_IN_{idx}"))
        for idx, item in enumerate(outputs):
            dtype = self.dtype_conversion(str(item.dtype))
            output_pto_list.append(pypto.tensor(item.shape, dtype, f"{name}_OUT_{idx}"))

        if on_board:
            torch.npu.set_device(self.device_id)
            pypto.runtime._device_init()
        try:
            pypto.set_vec_tile_shapes(self.tiling, self.tiling)
            with pypto.function(name, *input_pto_list, *output_pto_list) as rlf:
                for _ in rlf:
                    kernel(self.params, *input_pto_list, *output_pto_list)

            pto_input_data = [pypto.from_torch(tensor, f"{name}_IN_{idx}") for idx, tensor in enumerate(inputs)]
            pto_output_data = [pypto.from_torch(tensor, f"{name}_OUT_{idx}") for idx, tensor in enumerate(outputs)]
            pypto.runtime._device_run_once_data_from_host(*pto_input_data, *pto_output_data)
        finally:
            if on_board:
                pypto.runtime._device_fini()

    def _paged_attention_torch(self, query, key_cache, value_cache, block_table, act_seqs):
        b = self.params["b"]
        s = self.params["s"]
        n_q = self.params["n"]
        n_kv = self.params["n_kv"]
        d = self.params["d"]
        block_size = self.params["block_size"]
        group = n_q // n_kv
        q_4d = query.reshape(b, s, n_q, d).to(torch.float32)
        out = torch.zeros([b, s, n_q, d], dtype=torch.float32)
        scale = 1.0 / math.sqrt(d)
        for batch_idx in range(b):
            cur_block_table = block_table[batch_idx]
            act_seq = int(act_seqs[batch_idx].item())
            for token_idx in range(s):
                cur_seq = max(act_seq - s + 1 + token_idx, 0)
                for kv_idx in range(n_kv):
                    q_group = q_4d[batch_idx, token_idx, kv_idx * group : (kv_idx + 1) * group, :]
                    k_cur = _build_pa_rows_from_cache(key_cache, cur_block_table, block_size, cur_seq, kv_idx, d).to(
                        torch.float32
                    )
                    v_cur = _build_pa_rows_from_cache(
                        value_cache, cur_block_table, block_size, cur_seq, kv_idx, d
                    ).to(torch.float32)
                    scores = torch.matmul(q_group, k_cur.transpose(-1, -2)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    out_group = torch.matmul(probs, v_cur)
                    out[batch_idx, token_idx, kv_idx * group : (kv_idx + 1) * group, :] = out_group
        return out.reshape(b * s * n_q, d)

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3LayerRunner directly uses a frontend.jit kernel.")

        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        layer_kernel = build_qwen3_layer_frontend_jit(
            self.params,
            pypto.RunMode.NPU if on_board else pypto.RunMode.SIM,
        )
        self._run_frontend_jit_kernel(layer_kernel, self.inputs, on_board)
        layer_out = self.golden_output[0]
        self._assert_count_based_close(self.golden_output[0].cpu(), layer_out.cpu(), self.atol_value)

    def golden(
        self,
        params,
        hidden_states,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_o_w,
        attn_o_b,
        attn_q_norm_w,
        attn_k_norm_w,
        cos,
        sin,
        cache_index,
        key_cache,
        value_cache,
        block_table,
        act_seqs,
        gate_w,
        up_w,
        down_w,
        layer_out=None,
    ):
        b = params["b"]
        s = params["s"]
        n_q = params["n"]
        n_kv = params["n_kv"]
        d = params["d"]
        hidden_size = n_q * d
        kv_hidden = n_kv * d

        residual = hidden_states
        norm1 = _rmsnorm_torch(hidden_states)

        q = torch.matmul(norm1.to(torch.float32), attn_q_w.to(torch.float32)).to(hidden_states.dtype)
        q = (q.to(torch.float32) + attn_q_b.to(torch.float32)).to(hidden_states.dtype)
        k = torch.matmul(norm1.to(torch.float32), attn_k_w.to(torch.float32)).to(hidden_states.dtype)
        k = (k.to(torch.float32) + attn_k_b.to(torch.float32)).to(hidden_states.dtype)
        v = torch.matmul(norm1.to(torch.float32), attn_v_w.to(torch.float32)).to(hidden_states.dtype)
        v = (v.to(torch.float32) + attn_v_b.to(torch.float32)).to(hidden_states.dtype)

        q = _rmsnorm_torch(q.reshape(b * s * n_q, d), attn_q_norm_w).reshape(b, s, n_q, d)
        k = _rmsnorm_torch(k.reshape(b * s * n_kv, d), attn_k_norm_w).reshape(b, s, n_kv, d)
        v = v.reshape(b, s, n_kv, d)

        q_embed = _apply_rope_torch(q, cos, sin)
        k_embed = _apply_rope_torch(k, cos, sin)
        k_bsnd_updated = self._k_bsnd.clone()
        v_bsnd_updated = self._v_bsnd.clone()
        for batch_idx in range(b):
            for token_idx in range(s):
                pos = params["skv"] - s + token_idx
                k_bsnd_updated[batch_idx, pos, :, :] = k_embed[batch_idx, token_idx, :, :]
                v_bsnd_updated[batch_idx, pos, :, :] = v[batch_idx, token_idx, :, :]

        scale = 1.0 / math.sqrt(d)
        attn_out = torch.zeros([b, s, n_q, d], dtype=torch.float32)
        for batch_idx in range(b):
            act_seq = int(act_seqs[batch_idx].item())
            for token_idx in range(s):
                cur_seq = max(act_seq - s + 1 + token_idx, 0)
                for kv_idx in range(n_kv):
                    q_group = q_embed[batch_idx, token_idx, kv_idx * self._group : (kv_idx + 1) * self._group, :].to(
                        torch.float32
                    )
                    k_cur = k_bsnd_updated[batch_idx, :cur_seq, kv_idx, :].to(torch.float32)
                    v_cur = v_bsnd_updated[batch_idx, :cur_seq, kv_idx, :].to(torch.float32)
                    scores = torch.matmul(q_group, k_cur.transpose(-1, -2)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    out_group = torch.matmul(probs, v_cur)
                    attn_out[batch_idx, token_idx, kv_idx * self._group : (kv_idx + 1) * self._group, :] = out_group

        context = attn_out.reshape(b * s, hidden_size).to(hidden_states.dtype)
        o_proj = torch.matmul(context.to(torch.float32), attn_o_w.to(torch.float32)).to(hidden_states.dtype)
        o_proj = (o_proj.to(torch.float32) + attn_o_b.to(torch.float32)).to(hidden_states.dtype)
        h1 = (residual.to(torch.float32) + o_proj.to(torch.float32)).to(hidden_states.dtype)
        norm2 = _rmsnorm_torch(h1)

        gate = torch.matmul(norm2.to(torch.float32), gate_w.to(torch.float32)).to(hidden_states.dtype)
        up = torch.matmul(norm2.to(torch.float32), up_w.to(torch.float32)).to(hidden_states.dtype)
        gate_act = gate * torch.sigmoid(gate.to(torch.float32)).to(hidden_states.dtype)
        gate_up = (gate_act.to(torch.float32) * up.to(torch.float32)).to(hidden_states.dtype)
        mlp_out = torch.matmul(gate_up.to(torch.float32), down_w.to(torch.float32)).to(hidden_states.dtype)
        layer_out = (h1.to(torch.float32) + mlp_out.to(torch.float32)).to(hidden_states.dtype)

        return (layer_out,)


class TestQwen3Atten:
    def test_qwen3_paged_attention_prolog_b32_s1_n32_kv8_d128_blk2048_bf16(self):
        params = {
            "b": 32,
            "s": 1,
            "n": 32,
            "n_kv": 8,
            "d": 128,
            "block_size": 2048,
            "dtype": torch.bfloat16,
        }
        Qwen3PagedAttentionPrologRunner(params)()

    def test_qwen3_paged_attention_b32_s1_n32_kv8_d128_blk2048_skv2048_bf16(self):
        params = {
            "b": 32,
            "s": 1,
            "n": 32,
            "n_kv": 8,
            "d": 128,
            "block_size": 2048,
            "skv": 2048,
            "dtype": torch.bfloat16,
            "max_unroll_times": 8,
            "pa_tile_config": PaTileConfig(
                head_num_q_tile=4,
                c1_tile_shape=(4, 4, 128, 128, 256, 256),
                v1_tile_shape=(4, 2048),
                c2_tile_shape=(4, 4, 128, 128, 128, 128),
                v2_tile_shape=(4, 512),
            ),
        }
        Qwen3PagedAttentionRunner(params)()

    def test_qwen3_mlp_b32_s1_h4096_inter12288_bf16(self):
        params = {
            "b": 32,
            "s": 1,
            "h": 4096,
            "inter": 12288,
            "dtype": torch.bfloat16,
        }
        Qwen3MLPRunner(params)()

    def test_qwen3_layer_b32_s1_n32_kv8_d128_blk2048_inter12288_skv2048_bf16(self):
        params = {
            "b": 32,
            "s": 1,
            "n": 32,
            "n_kv": 8,
            "d": 128,
            "block_size": 2048,
            "inter": 12288,
            "skv": 2048,
            "dtype": torch.bfloat16,
            "max_unroll_times": 8,
            "pa_tile_config": PaTileConfig(
                head_num_q_tile=4,
                c1_tile_shape=(4, 4, 128, 128, 256, 256),
                v1_tile_shape=(4, 2048),
                c2_tile_shape=(4, 4, 128, 128, 128, 128),
                v2_tile_shape=(4, 512),
            ),
        }
        Qwen3LayerRunner(params)()

