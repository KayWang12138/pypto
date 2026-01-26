import sys
import math
import logging
import json
from pathlib import Path
from typing import List

import torch
import numpy as np
from bfloat16 import bfloat16

# NOTE: qwen3 目录层级比 deepseekv3/nsa 少一层，直接写 parents[4] 会在不同脚本里指向不同目录。
# 这里兼容两种层级：优先 scripts/helper，其次 cmake/helper（如果未来存在）。
for _helper_dir in [
    Path(__file__).parents[3].joinpath("helper"),  # .../cmake/scripts/helper
    Path(__file__).parents[4].joinpath("helper"),  # .../cmake/helper (fallback)
]:
    if _helper_dir.exists() and str(_helper_dir) not in sys.path:
        sys.path.append(str(_helper_dir))
from config_gen import TestBase

torch.manual_seed(0)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


########################################Qwen3Attention###################################
class Qwen3AttentionTest(TestBase):
    def __init__(self):
        super().__init__()
        self.name = "Qwen3AttentionTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            n=int,
            n_kv=int,
            d=int,
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d

        hidden_states = torch.rand(
            [self.b * self.s, hidden_size], dtype=self.dtype
        ).uniform_(-1, 1)

        attn_q_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_q_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)

        attn_k_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_k_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)

        attn_v_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_v_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)

        attn_o_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_o_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)

        attn_q_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)
        attn_k_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)

        self.setup_input_tensors(locals())
        return (
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
        )

    def core(
        self,
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
    ):
        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d

        x_2d = hidden_states

        x_fp32 = x_2d.to(torch.float32)
        # 和C++的Matmul输入输出对齐
        q = torch.matmul(x_fp32, attn_q_w.to(torch.float32)).to(self.dtype)
        q = (q.to(torch.float32) + attn_q_b.to(torch.float32)).to(self.dtype)

        k = torch.matmul(x_fp32, attn_k_w.to(torch.float32)).to(self.dtype)
        k = (k.to(torch.float32) + attn_k_b.to(torch.float32)).to(self.dtype)

        v = torch.matmul(x_fp32, attn_v_w.to(torch.float32)).to(self.dtype)
        v = (v.to(torch.float32) + attn_v_b.to(torch.float32)).to(self.dtype)

        # RMSNorm
        q = q.reshape(self.b * self.s * self.n, self.d)
        q = torch.nn.functional.normalize(q.to(torch.float32), dim=-1) * math.sqrt(
            self.d
        )

        # q = (q.to(torch.float32) * attn_q_norm_w.to(torch.float32)).to(self.dtype)# 错误
        q = q.to(self.dtype) * attn_q_norm_w  # 正确

        q = q.reshape(self.b, self.s, self.n, self.d)

        k = k.reshape(self.b * self.s * self.n_kv, self.d)
        k = torch.nn.functional.normalize(k.to(torch.float32), dim=-1) * math.sqrt(
            self.d
        )
        k = k.to(self.dtype) * attn_k_norm_w
        k = k.reshape(self.b, self.s, self.n_kv, self.d)

        # Attention
        q = q.transpose(1, 2)  # [B, N, S, D]
        k = k.transpose(1, 2)  # [B, N_kv, S, D]
        v = v.reshape(self.b, self.s, self.n_kv, self.d).transpose(1, 2)

        # score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d)
        score = torch.matmul(
            q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)
        ).to(self.dtype)

        score /= math.sqrt(self.d)

        # 下面两个写法都可以。但最好是先转换为FP32，保险一点
        probs = torch.softmax(score.to(torch.float32), dim=-1).to(self.dtype)
        # probs = torch.softmax(score, dim=-1)

        context = torch.matmul(probs.to(torch.float32), v.to(torch.float32)).to(
            self.dtype
        )  # [B, N, S, D]
        context = context.transpose(1, 2).reshape(self.b * self.s, hidden_size)

        # 2D [B*S, H]
        out = torch.matmul(context.to(torch.float32), attn_o_w.to(torch.float32)).to(
            self.dtype
        )
        out = (out.to(torch.float32) + attn_o_b.to(torch.float32)).to(self.dtype)
        attention_out = out
        self.setup_output({"attention_out": attention_out})


########################################Qwen3PagedAttentionProlog###################################
class Qwen3PagedAttentionPrologTest(TestBase):
    """
    Qwen3 PagedAttention Prolog (decode):
      - 输入 hidden_states(2D) 做 Q/K/V 投影
      - 对 Q/K 做 RMSNorm + 乘 norm weight
      - 对 Q/K 做 RoPE（使用 cos/sin: [B,S,D]，与C++ ApplyRotaryPosEmbV2一致的半维布局）
      - 将 roped K 与 V 写入 paged KV cache（2D: [blockNum*blockSize, nKv*D]）
      - 输出 query_out(2D: [B*S*nQ, D]) + key_cache_out/value_cache_out
    """

    def __init__(self):
        super().__init__()
        self.name = "Qwen3PagedAttentionPrologTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            n=int,  # nQ
            n_kv=int,
            d=int,
            block_size=int,
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        assert self.n % self.n_kv == 0
        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d

        hidden_states = torch.rand(
            [self.b * self.s, hidden_size], dtype=self.dtype
        ).uniform_(-1, 1)

        attn_q_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_q_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)

        attn_k_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_k_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)

        attn_v_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_v_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)

        attn_q_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)
        attn_k_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)

        # 生成 act_seq_list + cache_position（仅用于构造 RoPE 的 cos/sin 与 cache_index，不作为算子输入）
        if self.b == 1:
            act_seq_list = [self.block_size * 2 + self.s]
        else:
            act_seq_list = [
                self.block_size * 2 + self.s,
                self.block_size * 2 - 1 + self.s,
            ] + [self.block_size * 2 + self.s] * (self.b - 2)

        cache_position = torch.zeros([self.b, self.s], dtype=torch.int32)
        for bi, act_seq in enumerate(act_seq_list):
            start_pos = int(act_seq) - int(self.s)
            for si in range(self.s):
                cache_position[bi, si] = start_pos + si

        # RoPE cos/sin: [B,S,D]，按 half-layout 生成 (freqs + freqs)
        pos_fp32 = cache_position.to(torch.float32)  # [B,S]
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.d, 2, dtype=torch.float32) / float(self.d))
        )  # [D/2]
        freqs = pos_fp32.unsqueeze(-1) * inv_freq  # [B,S,D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,S,D]
        cos = torch.cos(emb).to(self.dtype)
        sin = torch.sin(emb).to(self.dtype)

        # block_table + paged cache
        block_num_per_batch = [
            int(math.ceil(x / self.block_size)) for x in act_seq_list
        ]
        block_num = int(sum(block_num_per_batch))
        max_block_num_per_batch = int(max(block_num_per_batch))

        block_table = torch.full(
            [self.b, max_block_num_per_batch], -1, dtype=torch.int32
        )
        block_id = 0
        for bi, bn in enumerate(block_num_per_batch):
            for j in range(bn):
                block_table[bi, j] = block_id
                block_id += 1

        key_cache = torch.rand(
            [block_num * self.block_size, kv_hidden], dtype=self.dtype
        ).uniform_(-1, 1)
        value_cache = torch.rand(
            [block_num * self.block_size, kv_hidden], dtype=self.dtype
        ).uniform_(-1, 1)

        # cache_index: [B,S] 行索引（flatten后的cache行号），用于 C++ ScatterUpdate(PA_BSND)
        # NOTE: 注意这里的kv cache的shape为二维（ifa_pa.py中实现的是三维）,[block_num * block_size, n_kv * d]
        cache_index = torch.zeros([self.b, self.s], dtype=torch.int32)
        for bi in range(self.b):
            for si in range(self.s):
                pos = int(cache_position[bi, si].item())
                blk_in_batch = pos // self.block_size
                tail = pos % self.block_size
                gid = int(block_table[bi, blk_in_batch].item())  # gid为物理块号
                cache_index[bi, si] = (
                    gid * self.block_size + tail
                )  # 当前batch的当前query token对应的物理KV cache的flatten后的行号

        self.setup_input_tensors(
            {
                "hidden_states": hidden_states,
                "attn_q_w": attn_q_w,
                "attn_q_b": attn_q_b,
                "attn_k_w": attn_k_w,
                "attn_k_b": attn_k_b,
                "attn_v_w": attn_v_w,
                "attn_v_b": attn_v_b,
                "attn_q_norm_w": attn_q_norm_w,
                "attn_k_norm_w": attn_k_norm_w,
                "cos": cos,
                "sin": sin,
                "cache_index": cache_index,
                "key_cache": key_cache,
                "value_cache": value_cache,
            }
        )
        return (
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
        )

    def core(
        self,
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
    ):  # noqa: ARG002
        eps = 1e-6

        def rmsnorm_2d(x_2d: torch.Tensor) -> torch.Tensor:
            x_fp32 = x_2d.to(torch.float32)
            rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
            return (x_fp32 * rstd).to(self.dtype)

        def rope_rearrange(x_4d: torch.Tensor) -> torch.Tensor:
            # 对齐 C++ ApplyRotaryPosEmbV2: reshape(d/2,2) -> transpose -> reshape
            b, s, h, d = x_4d.shape
            x = x_4d.reshape(b, s, h, d // 2, 2).transpose(-1, -2).reshape(b, s, h, d)
            return x

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1]
            return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)

        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d
        x_fp32 = hidden_states.to(torch.float32)

        q = torch.matmul(x_fp32, attn_q_w.to(torch.float32)).to(self.dtype)
        q = (q.to(torch.float32) + attn_q_b.to(torch.float32)).to(self.dtype)

        k = torch.matmul(x_fp32, attn_k_w.to(torch.float32)).to(self.dtype)
        k = (k.to(torch.float32) + attn_k_b.to(torch.float32)).to(self.dtype)

        v = torch.matmul(x_fp32, attn_v_w.to(torch.float32)).to(self.dtype)
        v = (v.to(torch.float32) + attn_v_b.to(torch.float32)).to(self.dtype)

        # Q RMSNorm
        q = q.reshape(self.b * self.s * self.n, self.d)
        q = rmsnorm_2d(q) * attn_q_norm_w
        q = q.reshape(self.b, self.s, self.n, self.d)

        # K RMSNorm
        k = k.reshape(self.b * self.s * self.n_kv, self.d)
        k = rmsnorm_2d(k) * attn_k_norm_w
        k = k.reshape(self.b, self.s, self.n_kv, self.d)

        v = v.reshape(self.b, self.s, self.n_kv, self.d)

        # NOTE: 测试通过（1）
        # query_out = v.reshape(self.b * self.s * self.n_kv, self.d)
        # self.setup_output(
        #     {
        #         "query_out": query_out,
        #         "key_cache_out": key_cache.clone(),
        #         "value_cache_out": value_cache.clone(),
        #     }
        # )
        # return

        # RoPE (half-layout cos/sin, unsqueeze head dim)
        cos_u = cos.to(torch.float32).unsqueeze(2)  # [B,S,1,D]
        sin_u = sin.to(torch.float32).unsqueeze(2)

        q_r = rope_rearrange(q.to(torch.float32))
        k_r = rope_rearrange(k.to(torch.float32))
        q_embed = (q_r * cos_u + rotate_half(q_r) * sin_u).to(self.dtype)
        k_embed = (k_r * cos_u + rotate_half(k_r) * sin_u).to(self.dtype)

        query_out = q_embed.reshape(self.b * self.s * self.n, self.d)

        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()

        # 写入cache：row = cache_index[b, s]
        k_rows = k_embed.reshape(self.b * self.s, kv_hidden)
        v_rows = v.reshape(self.b * self.s, kv_hidden)

        # NOTE: 测试通过（2）max diff: 0
        # query_out = v_rows.reshape(self.b * self.s * self.n_kv, self.d)
        # self.setup_output(
        #     {
        #         "query_out": query_out,
        #         "key_cache_out": key_cache.clone(),
        #         "value_cache_out": value_cache.clone(),
        #     }
        # )
        # return

        # NOTE: 测试通过（3）max diff: 0
        # query_out = cache_index # [B,S]
        # self.setup_output(
        #     {
        #         "query_out": query_out,
        #         "key_cache_out": key_cache.clone(),
        #         "value_cache_out": value_cache.clone(),
        #     }
        # )
        # return

        for bi in range(self.b):
            for si in range(self.s):
                row = int(cache_index[bi, si].item())
                src = bi * self.s + si
                key_cache_out[row, :] = k_rows[src, :]
                value_cache_out[row, :] = v_rows[src, :]

        self.setup_output(
            {
                "query_out": query_out,
                "key_cache_out": key_cache_out,
                "value_cache_out": value_cache_out,
            }
        )


########################################Qwen3PagedAttention###################################
class Qwen3PagedAttentionTest(TestBase):
    """
    Qwen3 PagedAttention (decode):
      - query/out 都是2D: [B * S_q * nQ, D]
      - key/value cache 是2D: [blockNum * blockSize, nKv * D]
      - 支持GQA: nQ > nKv, group = nQ / nKv, 每个KV head共享group个Q heads
      - 不包含 RoPE/nope 拆分（认为Q/K已在外部完成 RMSNorm+RoPE 并写入cache）
    """

    def __init__(self):
        super().__init__()
        self.name = "Qwen3PagedAttentionTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,  # batch size
            s=int,  # 本次算子调用里每条sequence要算的 query token 数
            n=int,  # nQ
            n_kv=int,
            d=int,
            block_size=int,
            skv=(int, list),  # KV 上下文长度（act_seqs）。支持 int 或 list[int](len==b)
            dtype=torch.dtype,
        )
        params = self.load_parameters(param_set)
        # C++ TestDataLoader 仅支持 scalar 参数；list 类型需序列化为 string（否则解析会失败）
        if isinstance(params.get("skv"), list):
            params["skv"] = json.dumps(params["skv"])
        return params

    def define_input_tensors(self):
        assert self.n % self.n_kv == 0
        group = self.n // self.n_kv
        assert group > 0

        # 生成的 batch 内每条序列的有效长度（上下文长度）
        # skv: KV 序列长度（act_seqs），支持：
        # - skv=int：所有 batch 相同
        # - skv=list[int]：每个 batch 不同（len==b）
        skv = self.skv
        if isinstance(skv, int):
            actual_seq_len = [skv] * self.b
        elif isinstance(skv, list):
            if len(skv) == self.b:
                actual_seq_len = skv
            else:
                raise RuntimeError("unsupported skv list length")
        else:
            raise RuntimeError("unsupported skv data type")

        actual_seq_len = [int(x) for x in actual_seq_len]
        if any(x < int(self.s) for x in actual_seq_len):
            raise RuntimeError(
                f"invalid skv: expect skv >= s, got skv={actual_seq_len}, s={self.s}"
            )

        act_seq_list = actual_seq_len
        act_seqs = torch.tensor(act_seq_list, dtype=torch.int32)
        max_seq = int(max(act_seq_list))

        # Q: [B, S_q, nQ, D] -> 2D flatten
        q_4d = torch.rand([self.b, self.s, self.n, self.d], dtype=self.dtype).uniform_(
            -1, 1
        )
        query = q_4d.reshape(self.b * self.s * self.n, self.d)

        # 原始序列K/V（按token连续存放，后续再做paged散列）
        k_bsnd = torch.rand(
            [self.b, max_seq, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        v_bsnd = torch.rand(
            [self.b, max_seq, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        # 保存下来供core计算golden
        self._q_4d = q_4d
        self._k_bsnd = k_bsnd
        self._v_bsnd = v_bsnd
        self._group = group

        # 构造 blockTable + paged cache
        # 在一个batch中的每一个seq需要的block数量
        block_num_per_batch = [
            int(math.ceil(x / self.block_size)) for x in act_seq_list
        ]
        block_num = int(sum(block_num_per_batch))
        max_block_num_per_batch = int(max(block_num_per_batch))

        block_table = torch.zeros([self.b, max_block_num_per_batch], dtype=torch.int32)
        block_id = 0
        for bi, bn in enumerate(block_num_per_batch):
            for j in range(bn):
                block_table[bi, j] = block_id
                block_id += 1

        k_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        v_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        # 填充虚假KV Cache
        for bi, seq_len in enumerate(act_seq_list):
            bn = block_num_per_batch[bi]  # 当前seq需要的block数量
            for j in range(bn):
                gid = int(block_table[bi, j].item())  # 当前block的id
                # 从原始KV序列中获取当前block的KV数据
                start = j * self.block_size  # 当前block在原始KV张量中的起始位置
                end = min(
                    start + self.block_size, seq_len
                )  # 当前block在原始KV张量中的结束位置
                if end > start:
                    # 从原始序列K/V中获取当前block的KV数据，填充虚假KV Cache
                    k_cache[gid, 0 : (end - start), :, :] = k_bsnd[bi, start:end, :, :]
                    v_cache[gid, 0 : (end - start), :, :] = v_bsnd[bi, start:end, :, :]

        key_cache = k_cache.reshape(block_num * self.block_size, self.n_kv * self.d)
        value_cache = v_cache.reshape(block_num * self.block_size, self.n_kv * self.d)

        self.setup_input_tensors(
            {
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "block_table": block_table,
                "act_seqs": act_seqs,
            }
        )
        return query, key_cache, value_cache, block_table, act_seqs

    def core(
        self, query, key_cache, value_cache, block_table, act_seqs
    ):  # noqa: ARG002
        # golden: full attention (no paged logic needed here; paged仅是存储方式)
        q_4d = self._q_4d.to(torch.float32)  # [B,S,nQ,D]
        k_bsnd = self._k_bsnd.to(torch.float32)  # [B,Skv,nKv,D]
        v_bsnd = self._v_bsnd.to(torch.float32)
        group = self._group

        softmax_scale = 1.0 / math.sqrt(self.d)
        out = torch.zeros([self.b, self.s, self.n, self.d], dtype=torch.float32)

        for bi in range(self.b):
            act_seq = int(act_seqs[bi].item())
            for s_idx in range(
                self.s
            ):  # 遍历本次算子调用里每条sequence要算的每个query token
                # 支持s>1的轻量因果长度（与C++实现保持一致）
                # 当前query对应的kv cache的最后一个index（由于在prolog中，当前query对应的kv已经append到了kv cache的最后，所以这里对应最后的self.s个kv cache。由于有mask，所以当前query只能跟0到cur_seq-1的kv cache做attention）
                # 第 s_idx 个 query 只能 attend 到 [0, cur_seq) 的 K/V（随 s_idx 递增，包含自身）
                cur_seq = max(act_seq - self.s + 1 + s_idx, 0)
                for kv in range(self.n_kv):
                    q_grp = q_4d[
                        bi, s_idx, kv * group : (kv + 1) * group, :
                    ]  # [group, D] 当前一个kv头对应的多个query头
                    k_cur = k_bsnd[bi, :cur_seq, kv, :]  # [cur_seq, D]
                    v_cur = v_bsnd[bi, :cur_seq, kv, :]  # [cur_seq, D]

                    scores = (q_grp @ k_cur.T) * softmax_scale  # [group, cur_seq]
                    probs = torch.softmax(scores, dim=-1)  # fp32
                    out_grp = probs @ v_cur  # [group, D]
                    out[bi, s_idx, kv * group : (kv + 1) * group, :] = out_grp

        paged_attention_out = out.reshape(self.b * self.s * self.n, self.d)
        self.setup_output({"paged_attention_out": paged_attention_out})


class Qwen3PagedAttentionWindowTest(TestBase):
    """
    Qwen3 PagedAttention with sliding window:
      - 与 Qwen3PagedAttentionTest 相同，但只关注最后 window_size 个 token
    """

    def __init__(self):
        super().__init__()
        self.name = "Qwen3PagedAttentionWindowTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            n=int,  # nQ
            n_kv=int,
            d=int,
            block_size=int,
            window_size=int,
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        assert self.n % self.n_kv == 0
        group = self.n // self.n_kv
        assert group > 0
        assert self.window_size > 0

        # 生成每个batch的有效长度（固定策略，包含尾块）
        if self.b == 1:
            act_seq_list = [self.block_size * 3 + 1]
        else:
            act_seq_list = [self.block_size * 3 + 1, self.block_size * 3 - 1] + [
                self.block_size * 3 + 1
            ] * (self.b - 2)
        act_seqs = torch.tensor(act_seq_list, dtype=torch.int32)
        max_seq = int(max(act_seq_list))

        # Q: [B, S_q, nQ, D] -> 2D flatten
        q_4d = torch.rand([self.b, self.s, self.n, self.d], dtype=self.dtype).uniform_(
            -1, 1
        )
        query = q_4d.reshape(self.b * self.s * self.n, self.d)

        # 原始序列K/V（按token连续存放，后续再做paged散列）
        k_bsnd = torch.rand(
            [self.b, max_seq, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        v_bsnd = torch.rand(
            [self.b, max_seq, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        self._q_4d = q_4d
        self._k_bsnd = k_bsnd
        self._v_bsnd = v_bsnd
        self._group = group

        # 构造 blockTable + paged cache
        block_num_per_batch = [
            int(math.ceil(x / self.block_size)) for x in act_seq_list
        ]
        block_num = int(sum(block_num_per_batch))
        max_block_num_per_batch = int(max(block_num_per_batch))

        block_table = torch.zeros([self.b, max_block_num_per_batch], dtype=torch.int32)
        block_id = 0
        for bi, bn in enumerate(block_num_per_batch):
            for j in range(bn):
                block_table[bi, j] = block_id
                block_id += 1

        k_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        v_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        for bi, seq_len in enumerate(act_seq_list):
            bn = block_num_per_batch[bi]
            for j in range(bn):
                gid = int(block_table[bi, j].item())
                start = j * self.block_size
                end = min(start + self.block_size, seq_len)
                if end > start:
                    k_cache[gid, 0 : (end - start), :, :] = k_bsnd[bi, start:end, :, :]
                    v_cache[gid, 0 : (end - start), :, :] = v_bsnd[bi, start:end, :, :]

        key_cache = k_cache.reshape(block_num * self.block_size, self.n_kv * self.d)
        value_cache = v_cache.reshape(block_num * self.block_size, self.n_kv * self.d)

        self.setup_input_tensors(
            {
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "block_table": block_table,
                "act_seqs": act_seqs,
            }
        )
        return query, key_cache, value_cache, block_table, act_seqs

    def core(
        self, query, key_cache, value_cache, block_table, act_seqs
    ):  # noqa: ARG002
        q_4d = self._q_4d.to(torch.float32)  # [B,S,nQ,D]
        k_bsnd = self._k_bsnd.to(torch.float32)  # [B,Skv,nKv,D]
        v_bsnd = self._v_bsnd.to(torch.float32)
        group = self._group

        softmax_scale = 1.0 / math.sqrt(self.d)
        out = torch.zeros([self.b, self.s, self.n, self.d], dtype=torch.float32)

        for bi in range(self.b):
            act_seq = int(act_seqs[bi].item())
            for s_idx in range(self.s):
                cur_seq = max(act_seq - self.s + 1 + s_idx, 0)
                start = max(
                    0, cur_seq - int(self.window_size)
                )  # 在这里多了start，计算起始位置，只关注最后window_size个token
                for kv in range(self.n_kv):
                    q_grp = q_4d[
                        bi, s_idx, kv * group : (kv + 1) * group, :
                    ]  # [group, D]
                    k_cur = k_bsnd[bi, start:cur_seq, kv, :]  # [win, D]
                    v_cur = v_bsnd[bi, start:cur_seq, kv, :]  # [win, D]

                    scores = (q_grp @ k_cur.T) * softmax_scale  # [group, win]
                    probs = torch.softmax(scores, dim=-1)
                    out_grp = probs @ v_cur
                    out[bi, s_idx, kv * group : (kv + 1) * group, :] = out_grp

        paged_attention_out = out.reshape(self.b * self.s * self.n, self.d)
        self.setup_output({"paged_attention_out": paged_attention_out})


########################################Qwen3MLP###################################
class Qwen3MLPTest(TestBase):
    def __init__(self):
        super().__init__()
        self.name = "Qwen3MLPTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            h=int,
            inter=int,
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        hidden_size = self.h
        inter_size = self.inter

        hidden_states = torch.rand(
            [self.b * self.s, hidden_size], dtype=self.dtype
        ).uniform_(-1, 1)

        gate_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        up_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        down_w = torch.rand([inter_size, hidden_size], dtype=self.dtype).uniform_(-1, 1)

        self.setup_input_tensors(locals())
        return hidden_states, gate_w, up_w, down_w

    def core(self, hidden_states, gate_w, up_w, down_w):
        x_fp32 = hidden_states.to(torch.float32)

        gate = torch.matmul(x_fp32, gate_w.to(torch.float32)).to(self.dtype)

        up = torch.matmul(x_fp32, up_w.to(torch.float32)).to(self.dtype)
        gate_act = gate * (torch.sigmoid(gate.to(torch.float32))).to(self.dtype)  # SiLU

        inter = gate_act * up

        out = torch.matmul(inter.to(torch.float32), down_w.to(torch.float32)).to(
            self.dtype
        )
        mlp_out = out
        self.setup_output({"mlp_out": mlp_out})


########################################Qwen3Layer###################################
class Qwen3LayerTest(TestBase):
    def __init__(self):
        super().__init__()
        self.name = "Qwen3LayerTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            h=int,
            n=int,
            n_kv=int,
            inter=int,
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        hidden_size = self.h
        kv_hidden = self.n_kv * self.h // self.n
        d = self.h // self.n
        inter_size = self.inter

        hidden_states = torch.rand(
            [self.b * self.s, hidden_size], dtype=self.dtype
        ).uniform_(-1, 1)

        attn_q_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_q_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)
        attn_k_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_k_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)
        attn_v_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_v_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)
        attn_o_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_o_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)
        attn_q_norm_w = torch.rand([d], dtype=self.dtype).uniform_(-1, 1)
        attn_k_norm_w = torch.rand([d], dtype=self.dtype).uniform_(-1, 1)

        gate_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        up_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        down_w = torch.rand([inter_size, hidden_size], dtype=self.dtype).uniform_(-1, 1)

        self.setup_input_tensors(locals())
        return (
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
            gate_w,
            up_w,
            down_w,
        )

    def core(
        self,
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
        gate_w,
        up_w,
        down_w,
    ):
        # Qwen3 Layer（Pre-Norm）:
        #   x -> RMSNorm -> Attention -> +residual -> RMSNorm -> MLP -> +residual
        # 复用 Qwen3AttentionTest.core 与 Qwen3MLPTest.core，保证底层 attention/mlp 逻辑一致
        hidden_size = self.h
        d = self.h // self.n

        # residual 0
        residual = hidden_states

        # RMSNorm 1 (before attention)
        attn_in = torch.nn.functional.normalize(
            hidden_states.to(torch.float32), dim=-1
        ) * math.sqrt(hidden_size)
        attn_in = attn_in.to(self.dtype)

        attn_test = Qwen3AttentionTest()
        attn_test.b = self.b
        attn_test.s = self.s
        attn_test.n = self.n
        attn_test.n_kv = self.n_kv
        attn_test.d = d
        attn_test.dtype = self.dtype
        attn_test.core(
            attn_in,
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
        )
        attn_out = attn_test.golden_outputs["attention_out"][0]

        # residual 1
        hidden_states = (residual.to(torch.float32) + attn_out.to(torch.float32)).to(
            self.dtype
        )
        residual = hidden_states

        # RMSNorm 2 (before MLP)
        mlp_in = torch.nn.functional.normalize(
            hidden_states.to(torch.float32), dim=-1
        ) * math.sqrt(hidden_size)
        mlp_in = mlp_in.to(self.dtype)

        mlp_test = Qwen3MLPTest()
        mlp_test.dtype = self.dtype
        mlp_test.core(mlp_in, gate_w, up_w, down_w)
        mlp_out = mlp_test.golden_outputs["mlp_out"][0]

        # residual 2
        out = (residual.to(torch.float32) + mlp_out.to(torch.float32)).to(self.dtype)

        self.setup_output({"layer_out": out})


########################################Qwen3LayerNew###################################
class Qwen3LayerNewTest(TestBase):
    """
    Qwen3LayerNew (decode / paged-attn, Pre-Norm):
      hiddenStates -> RMSNorm -> Prolog (QKV + RoPE + write KV cache)
        -> PagedAttention -> OProj -> +residual
        -> RMSNorm -> MLP -> +residual -> layerOut

    输入:
      - hiddenStates: [B*S, H], H=nQ*d
      - Attention 权重: qWeight, qBias, kWeight, kBias, vWeight, vBias, oWeight, oBias
      - Norm 权重: qNormWeight, kNormWeight
      - RoPE: cos, sin [B, S, d]
      - KV Cache: keyCache, valueCache [blockNum*blockSize, nKv*d]
      - PagedAttention: cacheIndex, blockTable, actSeqs
      - MLP 权重: gateWeight, upWeight, downWeight

    输出:
      - layerOut: [B*S, H]
      - keyCacheOut, valueCacheOut: in-place 更新的 KV cache
    """

    def __init__(self):
        super().__init__()
        self.name = "Qwen3LayerNewTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b=int,
            s=int,
            n=int,  # nQ
            n_kv=int,
            d=int,
            block_size=int,
            inter=int,  # intermediate size
            skv=int,  # KV sequence length
            dtype=torch.dtype,
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d
        inter_size = self.inter

        # hiddenStates: [B*S, H]
        hidden_states = torch.rand(
            [self.b * self.s, hidden_size], dtype=self.dtype
        ).uniform_(-1, 1)

        # Attention weights
        attn_q_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_q_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)
        attn_k_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_k_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)
        attn_v_w = torch.rand([hidden_size, kv_hidden], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_v_b = torch.rand([kv_hidden], dtype=self.dtype).uniform_(-1, 1)
        attn_o_w = torch.rand([hidden_size, hidden_size], dtype=self.dtype).uniform_(
            -1, 1
        )
        attn_o_b = torch.rand([hidden_size], dtype=self.dtype).uniform_(-1, 1)

        # Norm weights
        attn_q_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)
        attn_k_norm_w = torch.rand([self.d], dtype=self.dtype).uniform_(-1, 1)

        # RoPE: cos, sin [B, S, D]
        cos = torch.rand([self.b, self.s, self.d], dtype=self.dtype).uniform_(-1, 1)
        sin = torch.rand([self.b, self.s, self.d], dtype=self.dtype).uniform_(-1, 1)

        # cache_index: [B, S]
        cache_index = torch.zeros([self.b, self.s], dtype=torch.int32)
        for bi in range(self.b):
            for si in range(self.s):
                cache_index[bi, si] = self.skv - self.s + si + bi * self.block_size

        # KV Cache: blockTable + paged cache
        assert self.n % self.n_kv == 0
        group = self.n // self.n_kv
        act_seq_list = [self.skv] * self.b
        block_num_per_batch = [
            int(math.ceil(x / self.block_size)) for x in act_seq_list
        ]
        block_num = int(sum(block_num_per_batch))
        max_block_num_per_batch = int(max(block_num_per_batch))

        block_table = torch.zeros([self.b, max_block_num_per_batch], dtype=torch.int32)
        block_id = 0
        for bi, bn in enumerate(block_num_per_batch):
            for j in range(bn):
                block_table[bi, j] = block_id
                block_id += 1

        # 原始 K/V 序列数据（供golden计算使用）
        k_bsnd = torch.rand(
            [self.b, self.skv, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        v_bsnd = torch.rand(
            [self.b, self.skv, self.n_kv, self.d], dtype=self.dtype
        ).uniform_(-1, 1)
        self._k_bsnd = k_bsnd
        self._v_bsnd = v_bsnd
        self._group = group

        # 构造 paged cache
        k_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        v_cache = torch.zeros(
            [block_num, self.block_size, self.n_kv, self.d], dtype=self.dtype
        )
        for bi, seq_len in enumerate(act_seq_list):
            bn = block_num_per_batch[bi]
            for j in range(bn):
                gid = int(block_table[bi, j].item())
                start = j * self.block_size
                end = min(start + self.block_size, seq_len)
                if end > start:
                    k_cache[gid, 0 : (end - start), :, :] = k_bsnd[bi, start:end, :, :]
                    v_cache[gid, 0 : (end - start), :, :] = v_bsnd[bi, start:end, :, :]

        key_cache = k_cache.reshape(block_num * self.block_size, kv_hidden)
        value_cache = v_cache.reshape(block_num * self.block_size, kv_hidden)

        act_seqs = torch.tensor(act_seq_list, dtype=torch.int32)

        # MLP weights
        gate_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        up_w = torch.rand([hidden_size, inter_size], dtype=self.dtype).uniform_(-1, 1)
        down_w = torch.rand([inter_size, hidden_size], dtype=self.dtype).uniform_(-1, 1)

        self.setup_input_tensors(
            {
                "hidden_states": hidden_states,
                "attn_q_w": attn_q_w,
                "attn_q_b": attn_q_b,
                "attn_k_w": attn_k_w,
                "attn_k_b": attn_k_b,
                "attn_v_w": attn_v_w,
                "attn_v_b": attn_v_b,
                "attn_o_w": attn_o_w,
                "attn_o_b": attn_o_b,
                "attn_q_norm_w": attn_q_norm_w,
                "attn_k_norm_w": attn_k_norm_w,
                "cos": cos,
                "sin": sin,
                "cache_index": cache_index,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "block_table": block_table,
                "act_seqs": act_seqs,
                "gate_w": gate_w,
                "up_w": up_w,
                "down_w": down_w,
            }
        )
        return (
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
        )

    def core(
        self,
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
    ):
        """
        Qwen3LayerNew golden computation:
          1. RMSNorm (pre-attention)
          2. Prolog: QKV projection + RMSNorm(Q,K) + RoPE + write KV cache
          3. PagedAttention (streaming softmax)
          4. OProj + residual 1
          5. RMSNorm (pre-MLP)
          6. MLP (SwiGLU)
          7. residual 2
        """
        eps = 1e-6
        hidden_size = self.n * self.d
        kv_hidden = self.n_kv * self.d
        group = self._group
        k_bsnd = self._k_bsnd
        v_bsnd = self._v_bsnd

        def rmsnorm_2d(x_2d: torch.Tensor) -> torch.Tensor:
            x_fp32 = x_2d.to(torch.float32)
            rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
            return (x_fp32 * rstd).to(self.dtype)

        def rope_rearrange(x_4d: torch.Tensor) -> torch.Tensor:
            b, s, h, d = x_4d.shape
            x = x_4d.reshape(b, s, h, d // 2, 2).transpose(-1, -2).reshape(b, s, h, d)
            return x

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1]
            return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)

        # residual 0
        residual = hidden_states

        # ========== RMSNorm 1 (pre-attention) ==========
        norm1 = rmsnorm_2d(hidden_states)  # [B*S, H]

        # ========== Prolog: QKV + RMSNorm(Q,K) + RoPE ==========
        x_fp32 = norm1.to(torch.float32)

        q = torch.matmul(x_fp32, attn_q_w.to(torch.float32)).to(self.dtype)
        q = (q.to(torch.float32) + attn_q_b.to(torch.float32)).to(self.dtype)

        k = torch.matmul(x_fp32, attn_k_w.to(torch.float32)).to(self.dtype)
        k = (k.to(torch.float32) + attn_k_b.to(torch.float32)).to(self.dtype)

        v = torch.matmul(x_fp32, attn_v_w.to(torch.float32)).to(self.dtype)
        v = (v.to(torch.float32) + attn_v_b.to(torch.float32)).to(self.dtype)

        # Q RMSNorm
        q = q.reshape(self.b * self.s * self.n, self.d)
        q = rmsnorm_2d(q) * attn_q_norm_w
        q = q.reshape(self.b, self.s, self.n, self.d)

        # K RMSNorm
        k = k.reshape(self.b * self.s * self.n_kv, self.d)
        k = rmsnorm_2d(k) * attn_k_norm_w
        k = k.reshape(self.b, self.s, self.n_kv, self.d)

        v = v.reshape(self.b, self.s, self.n_kv, self.d)

        # RoPE
        cos_u = cos.to(torch.float32).unsqueeze(2)  # [B,S,1,D]
        sin_u = sin.to(torch.float32).unsqueeze(2)

        q_r = rope_rearrange(q.to(torch.float32))
        k_r = rope_rearrange(k.to(torch.float32))
        q_embed = (q_r * cos_u + rotate_half(q_r) * sin_u).to(self.dtype)
        k_embed = (k_r * cos_u + rotate_half(k_r) * sin_u).to(self.dtype)

        # Write to KV cache (update k_bsnd/v_bsnd for attention computation)
        # 将 prolog 产生的新 K/V 写入缓存（覆盖最后 S 个位置）
        k_bsnd_updated = k_bsnd.clone()
        v_bsnd_updated = v_bsnd.clone()
        for bi in range(self.b):
            for si in range(self.s):
                pos = self.skv - self.s + si  # 写入位置
                k_bsnd_updated[bi, pos, :, :] = k_embed[bi, si, :, :]
                v_bsnd_updated[bi, pos, :, :] = v[bi, si, :, :]

        # ========== PagedAttention ==========
        softmax_scale = 1.0 / math.sqrt(self.d)
        attn_out = torch.zeros([self.b, self.s, self.n, self.d], dtype=torch.float32)

        for bi in range(self.b):
            act_seq = int(act_seqs[bi].item())
            for s_idx in range(self.s):
                cur_seq = max(act_seq - self.s + 1 + s_idx, 0)
                for kv_idx in range(self.n_kv):
                    q_grp = q_embed[
                        bi, s_idx, kv_idx * group : (kv_idx + 1) * group, :
                    ].to(torch.float32)
                    k_cur = k_bsnd_updated[bi, :cur_seq, kv_idx, :].to(torch.float32)
                    v_cur = v_bsnd_updated[bi, :cur_seq, kv_idx, :].to(torch.float32)

                    scores = (q_grp @ k_cur.T) * softmax_scale  # [group, cur_seq]
                    probs = torch.softmax(scores, dim=-1)
                    out_grp = probs @ v_cur  # [group, D]
                    attn_out[bi, s_idx, kv_idx * group : (kv_idx + 1) * group, :] = (
                        out_grp
                    )

        # context: [B*S*nQ, D] -> [B*S, H]
        context = attn_out.reshape(self.b * self.s, hidden_size).to(self.dtype)

        # ========== OProj ==========
        o_proj = torch.matmul(context.to(torch.float32), attn_o_w.to(torch.float32)).to(
            self.dtype
        )
        o_proj = (o_proj.to(torch.float32) + attn_o_b.to(torch.float32)).to(self.dtype)

        # ========== Residual 1 ==========
        h1 = (residual.to(torch.float32) + o_proj.to(torch.float32)).to(self.dtype)
        residual = h1

        # ========== RMSNorm 2 (pre-MLP) ==========
        norm2 = rmsnorm_2d(h1)

        # ========== MLP (SwiGLU) ==========
        mlp_in = norm2.to(torch.float32)
        gate = torch.matmul(mlp_in, gate_w.to(torch.float32))
        up = torch.matmul(mlp_in, up_w.to(torch.float32))
        gate_act = torch.nn.functional.silu(gate)
        gate_up = (gate_act * up).to(self.dtype)
        mlp_out = torch.matmul(gate_up.to(torch.float32), down_w.to(torch.float32)).to(
            self.dtype
        )

        # ========== Residual 2 ==========
        layer_out = (residual.to(torch.float32) + mlp_out.to(torch.float32)).to(
            self.dtype
        )

        # ========== Output KV cache (in-place update) ==========
        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()
        k_rows = k_embed.reshape(self.b * self.s, kv_hidden)
        v_rows = v.reshape(self.b * self.s, kv_hidden)
        for bi in range(self.b):
            for si in range(self.s):
                row = int(cache_index[bi, si].item())
                src = bi * self.s + si
                key_cache_out[row, :] = k_rows[src, :]
                value_cache_out[row, :] = v_rows[src, :]

        self.setup_output(
            {
                "layer_out": layer_out,
                "key_cache_out": key_cache_out,
                "value_cache_out": value_cache_out,
            }
        )


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestQwen3Atten.TestQwen3Atten_B_1_S_16_N_2_D_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_8_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_2048_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_1_D_128_BLK_128_SKV_256_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_8_D_128_BLK_128_SKV_256_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_512_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_512_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_2048_SKV_2048_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionWindow_B_2_S_1_N_4_KV_2_D_16_BLK_16_WIN_16_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_16_H_32_INTER_64_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_1024_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_6144_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_2048_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_4096_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_32_S_1_H_4096_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3Layer_B_1_S_16_H_32_N_2_KV_2_INTER_64_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_4096_INTER_12288_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_1024_INTER_12288_SKV_1024_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_2048_INTER_12288_SKV_2048_BF16",
    ]
)
def gen_qwen3_attention_data(case_name: str, output: Path) -> bool:
    cases = {
        "TestQwen3Atten.TestQwen3Atten_B_1_S_16_N_2_D_16_BF16": (
            Qwen3AttentionTest,
            (1, 16, 2, 2, 16, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16": (
            Qwen3PagedAttentionPrologTest,
            (2, 1, 4, 2, 16, 16, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16": (
            Qwen3PagedAttentionPrologTest,
            (2, 1, 32, 8, 128, 16, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_8_S_1_N_32_KV_8_D_128_BLK_16_BF16": (
            Qwen3PagedAttentionPrologTest,
            (8, 1, 32, 8, 128, 16, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_4096_BF16": (  # Qwen3-8B aligned
            Qwen3PagedAttentionPrologTest,
            (32, 1, 32, 8, 128, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_2048_BF16": (
            Qwen3PagedAttentionPrologTest,
            (32, 1, 32, 8, 128, 2048, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16": (
            Qwen3PagedAttentionTest,
            (2, 1, 4, 2, 16, 16, [33, 31], torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16": (  # qwen3-8b
            Qwen3PagedAttentionTest,
            (
                2,
                1,
                32,
                8,
                128,
                16,
                256,  # skv=256（4096 太大，swimlane json 会非常大）
                torch.bfloat16,
            ),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_1_D_128_BLK_128_SKV_256_BF16": (  # time: 45us AI Core:27.72%
            Qwen3PagedAttentionTest,
            (4, 1, 32, 1, 128, 128, 256, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_8_D_128_BLK_128_SKV_256_BF16": (
            Qwen3PagedAttentionTest,
            (4, 1, 32, 8, 128, 128, 256, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_512_SKV_4096_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 32, 1, 128, 512, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_4096_SKV_4096_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 32, 1, 128, 4096, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_128_BLK_4096_SKV_4096_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 128, 1, 128, 4096, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_512_BLK_4096_SKV_4096_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 128, 1, 512, 4096, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_4096_SKV_4096_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 32, 8, 128, 4096, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_2048_SKV_2048_BF16": (
            Qwen3PagedAttentionTest,
            (32, 1, 32, 8, 128, 2048, 2048, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3PagedAttentionWindow_B_2_S_1_N_4_KV_2_D_16_BLK_16_WIN_16_BF16": (
            Qwen3PagedAttentionWindowTest,
            (2, 1, 4, 2, 16, 16, 16, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_1_S_16_H_32_INTER_64_BF16": (
            Qwen3MLPTest,
            (1, 16, 32, 64, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_1024_BF16": (
            Qwen3MLPTest,
            (1, 1, 1024, 1024, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_6144_BF16": (
            Qwen3MLPTest,
            (1, 1, 1024, 6144, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_2048_INTER_12288_BF16": (
            Qwen3MLPTest,
            (1, 1, 2048, 12288, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_4096_INTER_12288_BF16": (
            Qwen3MLPTest,
            (1, 1, 4096, 12288, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3MLP_B_32_S_1_H_4096_INTER_12288_BF16": (
            Qwen3MLPTest,
            (32, 1, 4096, 12288, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3Layer_B_1_S_16_H_32_N_2_KV_2_INTER_64_BF16": (
            Qwen3LayerTest,
            (1, 16, 32, 2, 2, 64, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_4096_INTER_12288_SKV_4096_BF16": (
            Qwen3LayerNewTest,
            (32, 1, 32, 8, 128, 4096, 12288, 4096, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_1024_INTER_12288_SKV_1024_BF16": (
            Qwen3LayerNewTest,
            (32, 1, 32, 8, 128, 1024, 12288, 1024, torch.bfloat16),
        ),
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_2048_INTER_12288_SKV_2048_BF16": (
            Qwen3LayerNewTest,
            (32, 1, 32, 8, 128, 2048, 12288, 2048, torch.bfloat16),
        ),
    }

    if case_name in cases:
        test_cls, params = cases[case_name]
        test = test_cls()
        test.name = case_name
        test.run(params, output.parent)
        return True

    logging.error("Can't get func to gen golden, Case(%s)", case_name)
    return False


def main() -> bool:
    case_name_list: List[str] = [
        "TestQwen3Atten.TestQwen3Atten_B_1_S_16_N_2_D_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_8_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionProlog_B_32_S_1_N_32_KV_8_D_128_BLK_2048_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_4_KV_2_D_16_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_2_S_1_N_32_KV_8_D_128_BLK_16_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_1_D_128_BLK_128_SKV_256_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_4_S_1_N_32_KV_8_D_128_BLK_128_SKV_256_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_512_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_1_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_128_KV_1_D_512_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_4096_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3PagedAttention_B_32_S_1_N_32_KV_8_D_128_BLK_2048_SKV_2048_BF16",
        "TestQwen3Atten.TestQwen3PagedAttentionWindow_B_2_S_1_N_4_KV_2_D_16_BLK_16_WIN_16_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_16_H_32_INTER_64_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_1024_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_1024_INTER_6144_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_2048_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_1_S_1_H_4096_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3MLP_B_32_S_1_H_4096_INTER_12288_BF16",
        "TestQwen3Atten.TestQwen3Layer_B_1_S_16_H_32_N_2_KV_2_INTER_64_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_4096_INTER_12288_SKV_4096_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_1024_INTER_12288_SKV_1024_BF16",
        "TestQwen3Atten.TestQwen3LayerNew_B_32_S_1_N_32_KV_8_D_128_BLK_2048_INTER_12288_SKV_2048_BF16",
    ]
    ret: bool = True
    g_src_root = Path(__file__).parents[7]
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_qwen3_attention_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
