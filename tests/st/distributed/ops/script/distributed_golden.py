#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import dataclasses
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


np.random.seed(0)
torch.manual_seed(0)

STR_TO_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float,
    "bfloat16": torch.bfloat16,
}

def get_dtype(dtype_str: str):
    if dtype_str not in STR_TO_DTYPE:
        return False, None
    else:
        return True, STR_TO_DTYPE[dtype_str]


@dataclasses.dataclass
class MoeDispatchCase:
    batch_size: int
    hidden_size: int
    share_expert_num: int
    routing_expert_num: int
    top_k: int
    rank_size: int
    dtype: torch.dtype
    is_dynamic: bool

    def __post_init__(self):
        if self.share_expert_num + self.routing_expert_num != self.rank_size:
            raise ValueError(
                "(share_expert_num + routing_expert_num) != rank_size: "
                f"{self.share_expert_num} + {self.routing_expert_num} != {self.rank_size}"
            )


class DistributedTestGolden:
    @staticmethod
    def tensor_bf16_tofile(t: torch.tensor, filename: str):
        input_file_bin = open(filename, "wb")
        for each in t:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        input_file_bin.close()

    @staticmethod
    def save_tensor(tensor, save_dir: Path, save_name: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / save_name
        if tensor.dtype == torch.bfloat16:
            # view只改变形状，元素总数和存储顺序不变
            tensor.view(torch.int16).numpy().tofile(save_path)
        else:
            tensor.numpy().tofile(save_path)

    @staticmethod
    def split_case_name(case_name: str, dim: int):
        parts = case_name.split('_')  # 按 '_' 分割字符串
        if len(parts) < dim + 2:
            raise ValueError(f"case_name {case_name} format is error.")
        rank = int(parts[-1])
        shape = []
        for i in range(-1 - dim, -1):
            shape.append(int(parts[i]))
        _, dtype = get_dtype(parts[-2 - dim])

        result = {
            "dtype": dtype,
            "shape": shape,
            "rank": rank
        }
        has_in_dtype, in_dtype = get_dtype(parts[-3 - dim])
        if has_in_dtype:
            result['in_dtype'] = in_dtype

        logging.info(f"Case {case_name}, case info: {result}")
        return result

    @staticmethod
    def get_dtype_num(dtype):
        torch_dtype_to_num: Dict[torch.dtype, int] = {
            torch.bool: 15,
            torch.uint8: 11,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 3,
            torch.int64: 4,
            torch.float16: 6,
            torch.float: 7,
            torch.bfloat16: 8
        }
        dtype_num = 0
        if dtype in torch_dtype_to_num:
            dtype_num = torch_dtype_to_num[dtype]

        return dtype_num

    @staticmethod
    def save_params(shape: List[int], out_dtype: torch.dtype, world_size: int, output_dir: Path,
                    in_dtype: torch.dtype = None):
        dtype_num = DistributedTestGolden.get_dtype_num(out_dtype)
        if in_dtype:
            in_dtype_num = DistributedTestGolden.get_dtype_num(in_dtype)
            params = shape + [in_dtype_num, dtype_num, world_size]
        else:
            params = shape + [dtype_num, world_size]
        params = torch.tensor(params, dtype=torch.int64)
        params_path = output_dir / 'params.bin'
        params.numpy().tofile(params_path)

    @staticmethod
    def gen_input_tensor(shape: Tuple[int], dtype: torch.dtype, world_size: int, output_dir: Path, filename: str):
        input_tensor_list = []
        for rank in range(world_size):
            if dtype == torch.int32 or dtype == torch.int16 or dtype == torch.int8:
                input_tensor = torch.randint(-10, 10, shape, dtype=dtype)
            else:
                input_tensor = torch.randn(shape, dtype=dtype)
            DistributedTestGolden.save_tensor(input_tensor, output_dir, f'{filename}_rank_{rank}.bin')
            input_tensor_list.append(input_tensor)
        return input_tensor_list

    @staticmethod
    def gen_all_gather_case1(input_tensor_list, world_size, output_dir: Path, filename):
        logging.error(f"world_size========={world_size}")
        if world_size <= 1:
            logging.error("world_size must be greater than 1")
            return False
        allgather_out_tensor_list = []
        for rank in range(world_size):
            if rank == 0:
                output_tensor = input_tensor_list[rank]
            else:
                output_tensor = torch.cat((output_tensor, input_tensor_list[rank]), dim=0)
        for rank in range(world_size):
            DistributedTestGolden.save_tensor(output_tensor, output_dir, f'{filename}_rank_{rank}.bin')
            allgather_out_tensor_list.append(output_tensor)
        return allgather_out_tensor_list

    @staticmethod
    def gen_tensor_add(
            input_tensor_list: List[torch.Tensor], dtype: torch.dtype, world_size: int, output_dir: Path, filename: str,
        ):
        out_tensor_list = []
        for rank in range(world_size):
            t = input_tensor_list[rank]
            if dtype == torch.float16 or dtype == torch.bfloat16:
                t = t.to(torch.float32)

            res = torch.add(t, t)
            if dtype == torch.float16 or dtype == torch.bfloat16:
                res = res.to(dtype)
            DistributedTestGolden.save_tensor(res, output_dir, f'{filename}_rank_{rank}.bin')
            out_tensor_list.append(res)
        return out_tensor_list

    @staticmethod
    def gen_reduce_scatter_case1(
            input_tensor_list, row, dtype, world_size, output_dir: Path, filename, reduce_type: str = "sum",
        ):
        if dtype == torch.bfloat16:
            for i in range(len(input_tensor_list)):
                input_tensor_list[i] = input_tensor_list[i].to(torch.float32)
        
        # 现阶段只有sum操作
        if reduce_type == "sum":
            output_tensor = torch.sum(torch.stack(input_tensor_list, dim=0), dim=0).to(dtype)
        elif reduce_type == "max":
            output_tensor = torch.max(torch.stack(input_tensor_list, dim=0), dim=0)[0].to(dtype)
        elif reduce_type == "min":
            output_tensor = torch.min(torch.stack(input_tensor_list, dim=0), dim=0)[0].to(dtype)
        else:
            logging.error("ReduceScatter only supports three operations: sum, max and min.")
            return False

        row_out = row // world_size
        out_tensor_list = []
        if dtype == torch.bfloat16:
            output_tensor = output_tensor.to(torch.bfloat16)
        for rank in range(world_size):
            rank_output_tensor = output_tensor[rank * row_out: (rank + 1) * row_out]
            DistributedTestGolden.save_tensor(rank_output_tensor, output_dir, f'{filename}_rank_{rank}.bin')
            out_tensor_list.append(rank_output_tensor)
        return out_tensor_list

    @staticmethod
    def gen_all_gather_case(row, col, world_size, dtype, output_dir: Path):
        if world_size <= 1:
            logging.error("world_size must be greater than 1")
            return False

        DistributedTestGolden.save_params([row, col], dtype, world_size, output_dir)

        input_shape = [row, col]
        input_tensor_list = []
        for rank in range(world_size):
            if dtype == torch.int32 or dtype == torch.int16 or dtype == torch.int8:
                input_tensor = torch.randint(-10, 10, input_shape, dtype=dtype)
            else:
                input_tensor = torch.randn(input_shape, dtype=dtype)
            input_tensor_list.append(input_tensor)
            DistributedTestGolden.save_tensor(input_tensor, output_dir, f'input_rank_{rank}.bin')
            if rank == 0:
                output_tensor = input_tensor
            else:
                output_tensor = torch.cat((output_tensor, input_tensor), dim=0)
        for rank in range(world_size):
            DistributedTestGolden.save_tensor(output_tensor, output_dir, f'output_rank_{rank}.bin')
        return True

    @staticmethod
    def dist_all_gather(case_name: str, output: Path):
        case_info = DistributedTestGolden.split_case_name(case_name, 2)
        row, col = case_info['shape']
        ret = DistributedTestGolden.gen_all_gather_case(row, col, case_info['rank'], case_info['dtype'], output)
        if ret:
            logging.info("Case(%s), Golden generated success.", case_name)
        return ret

    @staticmethod
    def dist_moe_dispatch(case_name: str, output: Path):
        case = MoeDispatchCase(
            batch_size=8,
            hidden_size=7168,
            share_expert_num=1,
            routing_expert_num=3,
            top_k=2,
            rank_size=int(case_name.split("_")[-1]),
            dtype=torch.bfloat16,
            is_dynamic=False,
        )
        DistributedTestGolden.gen_moe_dispatch_case(case, output)
        logging.debug("Case(%s), Golden generated success.", case_name)
        return True
    
    @staticmethod
    def dist_allgather_matmul_reducescatter(case_name: str, output: Path):
        case_info = DistributedTestGolden.split_case_name(case_name, 2)
        row, col = case_info['shape']
        DistributedTestGolden.save_params([row, col], case_info['dtype'], case_info['rank'], output)
        
        input_tensor_list = DistributedTestGolden.gen_input_tensor((row, col), case_info['dtype'], case_info['rank'], 
                output, "input")
        
        # ag
        ag_out_tensor_list = DistributedTestGolden.gen_all_gather_case1(input_tensor_list, int(case_info['rank']),
                                                                        output, "allgather")

        # matmul ag [row, col]   mm[col, col]
        matmul_tensor_list = DistributedTestGolden.gen_input_tensor((col, col), case_info['dtype'], case_info['rank'], 
            output, "matmul")
        
        # ag and matmul ag [row, col]   mm[col, col]
        ag_add_tensor_list = DistributedTestGolden.gen_tensor_add(ag_out_tensor_list,  
                    case_info['dtype'], case_info['rank'], output, "ag_add")

        rs_out_tensor_list = DistributedTestGolden.gen_reduce_scatter_case1(ag_add_tensor_list, row * case_info['rank'],
                    case_info['dtype'], case_info['rank'], output, "rs")

        # ag
        DistributedTestGolden.gen_all_gather_case1(rs_out_tensor_list, int(case_info['rank']), output,
                                                   "double_allgather")        
        return True

    @staticmethod
    def gen_reduce_scatter_case(row, col, world_size, dtype, output_dir: Path, reduce_type: str = "sum"):
        if world_size <= 1:
            logging.error("world_size must be greater than 1")
            return False
        if row % world_size != 0:
            logging.error("The first dimension of the input tensor must be an integer multiple of the world size")
            return False

        DistributedTestGolden.save_params([row, col], dtype, world_size, output_dir)

        input_shape = [row, col]
        input_tensor_list = []
        for rank in range(world_size):
            if dtype == torch.int32 or dtype == torch.int16 or dtype == torch.int8:
                input_tensor = torch.randint(-10, 10, input_shape, dtype=dtype)
            else:
                input_tensor = torch.randn(input_shape, dtype=dtype)
            input_tensor_list.append(input_tensor)
            DistributedTestGolden.save_tensor(input_tensor, output_dir, f'input_rank_{rank}.bin')

        # 现阶段只有sum操作
        if reduce_type == "sum":
            output_tensor = torch.sum(torch.stack(input_tensor_list, dim=0), dim=0).to(dtype)
        elif reduce_type == "max":
            output_tensor = torch.max(torch.stack(input_tensor_list, dim=0), dim=0)[0].to(dtype)
        elif reduce_type == "min":
            output_tensor = torch.min(torch.stack(input_tensor_list, dim=0), dim=0)[0].to(dtype)
        else:
            logging.error("ReduceScatter only supports three operations: sum, max and min.")
            return False

        row_out = row // world_size
        for rank in range(world_size):
            rank_output_tensor = output_tensor[rank * row_out: (rank + 1) * row_out]
            DistributedTestGolden.save_tensor(rank_output_tensor, output_dir, f'output_rank_{rank}.bin')
        return True

    @staticmethod
    def dist_reduce_scatter(case_name: str, output: Path):
        case_info = DistributedTestGolden.split_case_name(case_name, 2)
        row, col = case_info['shape']
        ret = DistributedTestGolden.gen_reduce_scatter_case(row, col, case_info['rank'], case_info['dtype'], output)
        if ret:
            logging.info("Case(%s), Golden generated success.", case_name)
        return ret

    @staticmethod
    @GoldenRegister.reg_golden_func(
        case_names=[
            "DistributedTest.aicpuWaitFlag_single_test_reduce_scatter_int32_32_32_4",
            "DistributedTest.aicpuWaitFlag_single_test_all_gather_bfloat16_256_256_4",
            "DistributedTest.aicpuWaitFlag_multi_test_reduce_scatter_float32_128_256_4",
            "DistributedTest.aicpuWaitFlag_multi_test_all_gather_float16_32_32_4",
            "DistributedTest.aivWaitFlag_single_test_reduce_scatter_int32_128_256_4",
            "DistributedTest.aivWaitFlag_single_test_all_gather_bfloat16_256_128_4",
            "DistributedTest.aivWaitFlag_multi_test_reduce_scatter_float32_128_256_4",
            "DistributedTest.aivWaitFlag_multi_test_all_gather_float16_32_32_4",
            "DistributedTest.aivWaitFlag_single_test_moe_dispatch_bfloat16_rank_size_4",
            "DistributedTest.test_dyn_all_gather_int32_128_256_4",
            "DistributedTest.shmem_reduce_scatter_int32_128_256_4",
            "DistributedTest.dyn_allgather_matmul_reducescatter_int32_128_256_4",
        ]
    )
    def dist_operator_golden_gen_func(case_name: str, output: Path) -> bool:
        ret = False
        if "reduce_scatter" in case_name:
            ret = DistributedTestGolden.dist_reduce_scatter(case_name, output)
        elif "all_gather" in case_name:
            ret = DistributedTestGolden.dist_all_gather(case_name, output)
        elif "moe_dispatch" in case_name:
            ret = DistributedTestGolden.dist_moe_dispatch(case_name, output)
        elif "allgather_matmul_reducescatter" in case_name:
            ret = DistributedTestGolden.dist_allgather_matmul_reducescatter(case_name, output)
        if not ret:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return ret

    @staticmethod
    @GoldenRegister.reg_golden_func("DistributedTest.test_matmul_reducescatter_float16_float32_4_256_256_2")
    def gen_matmul_reducescatter(case_name: str, output: Path) -> bool:
        case_info = DistributedTestGolden.split_case_name(case_name, 3)
        m, k, n = case_info['shape']
        DistributedTestGolden.gen_matmul_reducescatter_case(
            m, k, n, case_info['rank'], case_info['in_dtype'], case_info['dtype'], output
        )
        return True

    @staticmethod
    def gen_matmul_reducescatter_case(m, k, n, world_size, in_dtype, out_dtype, output_dir):
        assert world_size >= 1, "world_size must be greater than or equal to 1"
        assert m % world_size == 0, "m % world_size != 0"

        DistributedTestGolden.save_params([m, k, n], out_dtype, world_size, output_dir, in_dtype)

        input_shape = [m, k]
        weight_shape = [k, n]
        matmul_output_tensor_list = []
        for rank in range(world_size):
            if in_dtype == torch.float32 or in_dtype == torch.float16 or in_dtype == torch.bfloat16:
                input_tensor = torch.randn(input_shape, dtype=in_dtype)
                weight_tensor = torch.randn(weight_shape, dtype=in_dtype)
            else:
                raise "matmul + reduce scatter only support 3 dtype: float32, float16, bfloat16"
            matmul_output_tensor = torch.matmul(input_tensor, weight_tensor).to(out_dtype)
            matmul_output_tensor_list.append(matmul_output_tensor)
            DistributedTestGolden.save_tensor(input_tensor, output_dir, f"input_rank_{rank}.bin")
            DistributedTestGolden.save_tensor(weight_tensor, output_dir, f"weight_rank_{rank}.bin")
            DistributedTestGolden.save_tensor(matmul_output_tensor, output_dir, f"mm_output_rank_{rank}.bin")

        # 现阶段只有sum操作
        output_tensor = torch.sum(torch.stack(matmul_output_tensor_list, dim=0), dim=0).to(out_dtype)

        out_m = m // world_size
        for rank in range(world_size):
            rank_output_tensor = output_tensor[rank * out_m: (rank + 1) * out_m]
            DistributedTestGolden.save_tensor(rank_output_tensor, output_dir, f'rs_output_rank_{rank}.bin')

    @staticmethod
    @GoldenRegister.reg_golden_func("DistributedTest.test_attention_post_reducescatter_bf16_real_batch16")
    def gen_attn_post_reducescatter(case_name: str, output: Path) -> bool:
        if case_name == "DistributedTest.test_deepseek_attention_post_reducescatter_bf16_batch4_ranksize4":
            input_b = 4
            input_s = 1
            input_n = 32
            input_h = 7168
            kv_lora_rank = 512
            v_head_dim = 128
            rank_size = 4
            dtype = torch.bfloat16
            DistributedTestGolden.gen_attn_post_reducescatter_case(input_b, input_s, input_n, input_h, kv_lora_rank,
                                                                   v_head_dim, rank_size, dtype, output)
        elif case_name == "DistributedTest.test_attention_post_fp16_real_rs":
            input_b = 32
            input_s = 1
            input_n = 32
            input_h = 7168
            kv_lora_rank = 512
            v_head_dim = 128
            rank_size = 4
            dtype = torch.bfloat16
            DistributedTestGolden.gen_attn_post_reducescatter_case(input_b, input_s, input_n, input_h, kv_lora_rank,
                                                                   v_head_dim, rank_size, dtype, output)
        elif case_name == "DistributedTest.test_attention_post_reducescatter_bf16_real_batch16":
            input_b = 16
            input_s = 1
            input_n = 32
            input_h = 7168
            kv_lora_rank = 512
            v_head_dim = 128
            rank_size = 4
            dtype = torch.bfloat16
            DistributedTestGolden.gen_attn_post_reducescatter_case(input_b, input_s, input_n, input_h, kv_lora_rank,
                                                                   v_head_dim, rank_size, dtype, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
        return True

    @staticmethod
    def gen_attn_post_reducescatter_case(input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, rank_size, dtype,
                                         output_dir):
        assert rank_size >= 1, "rank_size must be greater than or equal to 1"
        assert (input_b * input_s) % rank_size == 0, "(input_b * input_s) % rank_size != 0"

        DistributedTestGolden.save_params([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim], dtype,
                                          rank_size, output_dir)

        input_list = []
        w_uv_list = []
        w_o_list = []
        attn_output_list = []
        for rank in range(rank_size):
            if dtype == torch.float32 or dtype == torch.float16 or dtype == torch.bfloat16:
                input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)
                w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
                w_o = torch.randn([input_n * v_head_dim, input_h], dtype=dtype)
            else:
                raise "only support 3 dtype: float32, float16, bfloat16"
            input_list.append(input_t)
            w_uv_list.append(w_uv)
            w_o_list.append(w_o)
            DistributedTestGolden.save_tensor(input_t, output_dir / f"rank{rank}", "input.bin")
            DistributedTestGolden.save_tensor(w_uv, output_dir / f"rank{rank}", "w_uv.bin")
            DistributedTestGolden.save_tensor(w_o, output_dir / f"rank{rank}", "w_o.bin")

            t1 = input_t.transpose(1, 2)
            r1 = t1.reshape(input_b * input_s, input_n, kv_lora_rank)
            t2 = r1.transpose(0, 1)
            calc_input = t2
            bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
            if dtype != torch.float32:
                bmm4 = bmm4.to(dtype)
            t3 = bmm4.transpose(0, 1)
            r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
            bmm5_i = r2
            bmm5 = torch.matmul(bmm5_i.to(torch.float32), w_o.to(torch.float32))
            if dtype != torch.float32:
                bmm5 = bmm5.to(dtype)
            attn_output = bmm5.reshape(input_b, input_s, input_h)
            attn_output_list.append(attn_output)
            DistributedTestGolden.save_tensor(attn_output, output_dir / f"rank{rank}", "attn_output.bin")

        # 现阶段只有sum操作
        reduce_sum = torch.sum(torch.stack(attn_output_list, dim=0), dim=0).to(dtype).reshape(-1, input_h)

        out_m = (input_b * input_s) // rank_size
        for rank in range(rank_size):
            rs_output = reduce_sum[rank * out_m: (rank + 1) * out_m]
            DistributedTestGolden.save_tensor(rs_output, output_dir / f"rank{rank}", 'rs_output.bin')

    @staticmethod
    def gen_moe_dispatch(case_name: str, output: Path):
        if case_name == "DistributedTest.test_dispatch_rank_size_4":
            row = 8
            col = 7168
            share_num = 1
            expert_num = 3
            top_k = 2
            rank_size = 4
            dtype = torch.bfloat16
            DistributedTestGolden.gen_moe_dispatch_case(
                row, col, share_num, expert_num, top_k, rank_size, dtype, output,
            )
        elif case_name == "DistributedTest.test_dispatch_rank_size_8":
            row = 8
            col = 7168
            share_num = 2
            expert_num = 6
            top_k = 4
            rank_size = 8
            dtype = torch.bfloat16
            DistributedTestGolden.gen_moe_dispatch_case(
                row, col, share_num, expert_num, top_k, rank_size, dtype, output,
            )
        elif case_name == "DistributedTest.test_dynamic_dispatch_rank_size_8":
            row = 8
            col = 7168
            share_num = 2
            expert_num = 6
            top_k = 4
            rank_size = 8
            dtype = torch.bfloat16
            is_dynamic = True
            DistributedTestGolden.gen_moe_dispatch_case(
                row, col, share_num, expert_num, top_k, rank_size, dtype, output, is_dynamic,
            )

        logging.info("Case(%s), Golden generated success.", case_name)
        return True

    @staticmethod
    @GoldenRegister.reg_golden_func("DistributedTest.aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4")
    def gen_combine(case_name: str, output: Path):
        if case_name == "DistributedTest.aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4":
            row = 8
            col = 7168
            share_num = 1
            expert_num = 3
            top_k = 2
            rank_size = 4
            dtype = torch.bfloat16
            dispatch_case_name = "DistributedTest.test_dispatch_rank_size_4"
            case = MoeDispatchCase(
                batch_size=row,
                hidden_size=col,
                share_expert_num=share_num,
                routing_expert_num=expert_num,
                top_k=top_k,
                rank_size=rank_size,
                dtype=dtype,
                is_dynamic=False,
            )
            dispatch_output = output / dispatch_case_name
            dispatch_output.mkdir(parents=True, exist_ok=True)
            DistributedTestGolden.gen_moe_dispatch_case(case, dispatch_output)
            DistributedTestGolden.gen_combine_case(row, col, share_num, expert_num, top_k, rank_size, dtype, output,
                                                   dispatch_output)
            DistributedTestGolden.save_params([row, col, top_k], dtype, rank_size, output)
        elif case_name == "DistributedTest.test_combine_rank_size_8":
            row = 8
            col = 7168
            share_num = 2
            expert_num = 6
            top_k = 4
            rank_size = 8
            dtype = torch.bfloat16
            dispatch_case_name = "DistributedTest.test_dispatch_rank_size_8"
            DistributedTestGolden.gen_combine_case(row, col, share_num, expert_num, top_k, rank_size, dtype, output,
                                                   dispatch_case_name, case_name)
        elif case_name == "DistributedTest.test_dynamic_combine_rank_size_8":
            row = 8
            col = 7168
            share_num = 2
            expert_num = 6
            top_k = 4
            rank_size = 8
            dtype = torch.bfloat16
            dispatch_case_name = "DistributedTest.test_dynamic_dispatch_rank_size_8"
            DistributedTestGolden.gen_combine_case(row, col, share_num, expert_num, top_k, rank_size, dtype, output,
                                                   dispatch_case_name, case_name)

        logging.info("Case(%s), Golden generated success.", case_name)
        return True

    @staticmethod
    def gen_golden_input_data(case: MoeDispatchCase, output_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x_list = []
        expert_ids_list = []
        for rank in range(case.rank_size):
            x = torch.randn([case.batch_size, case.hidden_size], dtype=case.dtype)
            x_list.append(x)
            DistributedTestGolden.save_tensor(x, output_dir, f"x_rank_{rank}.bin")

            scores = torch.sigmoid(torch.randn([case.batch_size, case.routing_expert_num], dtype=torch.float32))
            _, expert_ids = torch.topk(scores, k=case.top_k)
            expert_ids_list.append(expert_ids)

            scales = scores.gather(1, expert_ids)
            DistributedTestGolden.save_tensor(scales, output_dir, f"scale_rank_{rank}.bin")

            expert_ids_rank = expert_ids + case.share_expert_num
            expert_ids_rank = expert_ids_rank.to(dtype=torch.int32)
            DistributedTestGolden.save_tensor(expert_ids_rank, output_dir, f"expert_ids_rank_{rank}.bin")

        return x_list, expert_ids_list

    @staticmethod
    def gen_golden_data_sended_to_single_share_expert(
            rank_id: int,
            case: MoeDispatchCase,
            x_list: List[torch.Tensor],
            output_dir: Path,
        ) -> None:
        y = torch.zeros([case.batch_size * case.rank_size, case.hidden_size], dtype=case.dtype)
        combine_info = torch.full([case.batch_size * case.rank_size, 3], -1, dtype=torch.int32)

        # 发给共享专家的kOffset暂时固定为topK
        # 本卡共享专家自己的数据
        offset = 0
        y[offset:offset + case.batch_size] = x_list[rank_id]
        for token_id in range(case.batch_size):
            combine_info[offset + token_id] = torch.tensor([rank_id, token_id, case.top_k], dtype=torch.int32)
        offset += case.batch_size

        # 路由专家发送给共享专家，根据 rank_id = expert_id // share_capacity
        for expert_id in range(case.routing_expert_num):
            routing_expert_rank_id = expert_id + case.share_expert_num
            # 一个共享专家负责MoE专家的容量
            share_capacity = case.routing_expert_num // case.share_expert_num if case.share_expert_num > 0 else 0
            if expert_id // share_capacity == rank_id:
                y[offset:offset + case.batch_size] = x_list[routing_expert_rank_id]
                for token_id in range(case.batch_size):
                    combine_info[offset + token_id] = torch.tensor(
                        [routing_expert_rank_id, token_id, case.top_k], dtype=torch.int32,
                    )
                offset += case.batch_size

        # 动态模式下，expand BS 是可变的
        if case.is_dynamic:
            DistributedTestGolden.save_tensor(y[:offset], output_dir, f"y_rank_{rank_id}.bin")
            DistributedTestGolden.save_tensor(combine_info[:offset], output_dir, f"combine_info_rank_{rank_id}.bin")
        else:
            DistributedTestGolden.save_tensor(y, output_dir, f"y_rank_{rank_id}.bin")
            DistributedTestGolden.save_tensor(combine_info, output_dir, f"combine_info_rank_{rank_id}.bin")

        valid_count = torch.zeros([128], dtype=torch.int32)
        valid_count[0] = offset
        DistributedTestGolden.save_tensor(valid_count, output_dir, f"valid_count_rank_{rank_id}.bin")

    @staticmethod
    def gen_golden_data_sended_to_all_share_experts(
            case: MoeDispatchCase,
            x_list: List[torch.Tensor],
            output_dir: Path,
        ) -> None:
        for rank_id in range(case.share_expert_num):
            DistributedTestGolden.gen_golden_data_sended_to_single_share_expert(rank_id, case, x_list, output_dir)

    @staticmethod
    def gen_golden_data_sended_to_single_routing_expert(
            expert_id: int,
            case: MoeDispatchCase,
            x_list: List[torch.Tensor],
            expert_ids_list: List[torch.Tensor],
            output_dir: Path,
        ) -> None:
        y = torch.zeros([case.batch_size * case.rank_size, case.hidden_size], dtype=case.dtype)
        combine_info = torch.full([case.batch_size * case.rank_size, 3], -1, dtype=torch.int32)
        offset = 0
        # 所有卡发送给本卡MoE专家
        for rank_id in range(case.rank_size):
            x = x_list[rank_id]
            expert_ids = expert_ids_list[rank_id]
            token_ids, k_offsets = torch.where(expert_ids == expert_id)
            if len(token_ids) > 0:
                y[offset:offset + len(token_ids)] = x[token_ids]
                for i, (token_id, k_offset) in enumerate(zip(token_ids, k_offsets)):
                    combine_info[offset + i] = torch.tensor(
                        [rank_id, token_id.item(), k_offset.item()], dtype=torch.int32,
                    )
            offset += len(token_ids)
        # MoE专家所在卡的位置
        routing_expert_rank_id = case.share_expert_num + expert_id
        if case.is_dynamic:
            y = y[:offset]
            combine_info = combine_info[:offset]
        DistributedTestGolden.save_tensor(y, output_dir, f"y_rank_{routing_expert_rank_id}.bin")
        DistributedTestGolden.save_tensor(combine_info, output_dir, f"combine_info_rank_{routing_expert_rank_id}.bin")
        # 有效的token数量 = 偏移量 offset
        valid_count = torch.zeros([128], dtype=torch.int32)
        valid_count[0] = offset
        DistributedTestGolden.save_tensor(valid_count, output_dir, f"valid_count_rank_{routing_expert_rank_id}.bin")

    @staticmethod
    def gen_golden_data_sended_to_all_routing_experts(
            case: MoeDispatchCase,
            x_list: List[torch.Tensor],
            expert_ids_list: List[torch.Tensor],
            output_dir: Path,
        ) -> None:
        for expert_id in range(case.routing_expert_num):
            DistributedTestGolden.gen_golden_data_sended_to_single_routing_expert(
                expert_id, case, x_list, expert_ids_list, output_dir,
            )

    @staticmethod
    def gen_moe_dispatch_case(case: MoeDispatchCase, output_dir: Path) -> None:
        DistributedTestGolden.save_params(
            [case.batch_size, case.hidden_size, case.share_expert_num, case.routing_expert_num, case.top_k],
            case.dtype,
            case.rank_size,
            output_dir,
        )
        x_list, expert_ids_list = DistributedTestGolden.gen_golden_input_data(case, output_dir)
        DistributedTestGolden.gen_golden_data_sended_to_all_share_experts(case, x_list, output_dir)
        DistributedTestGolden.gen_golden_data_sended_to_all_routing_experts(case, x_list, expert_ids_list, output_dir)

    @staticmethod
    def gen_combine_case(row: int, col: int, share_num: int, expert_num: int, top_k: int, rank_size: int,
        dtype: torch.dtype, output_dir: Path, dispatch_case_name: str):
        assert (share_num + expert_num) == rank_size, "(share_num + expert_num) != rank_size"

        # 一个共享专家负责MoE专家的容量
        share_capacity = expert_num // share_num if share_num > 0 else 0

        # 获取输入数据，combine_info的信息依赖于Dispatch的输出
        dispatch_dir = output_dir / dispatch_case_name
        x_list = []
        combine_info_list = []
        scale_list = []
        for rank in range(rank_size):
            x = torch.from_numpy(np.fromfile(dispatch_dir / f"y_rank_{rank}.bin"))
            x = x.view(dtype=dtype).view([-1, col])
            x_list.append(x)

            combine_info = torch.from_numpy(np.fromfile(dispatch_dir / f"combine_info_rank_{rank}.bin", dtype=np.int32))
            combine_info = combine_info.view([-1, 3])
            combine_info_list.append(combine_info)

            scale = torch.from_numpy(np.fromfile(dispatch_dir / f"scale_rank_{rank}.bin", dtype=np.float32))
            scale = scale.view([row, -1, 1])
            scale_list.append(scale)

        for rank in range(rank_size):
            # Combine共享专家的数据
            share_y = torch.zeros([row, col], dtype=dtype)
            if share_num > 0:
                # 本卡就是共享专家
                if rank < share_num:
                    rank_share = rank
                else:       # 本卡是MoE专家，根据 rank_share = expert_id // share_capacity 获取专家卡
                    rank_share = (rank - share_num) // share_capacity
                x = x_list[rank_share]
                combine_info = combine_info_list[rank_share]

                mask = combine_info[:, 0] == rank
                x = x[mask]
                assert x.shape == share_y.shape, "combine share data error: x.shape != share_y.shape"
                share_y = x
            DistributedTestGolden.save_tensor(share_y, output_dir, f"share_y_rank_{rank}.bin")

            # Combine MoE专家的数据
            moe_y = torch.zeros([row, top_k, col], dtype=dtype)
            for expert_id in range(expert_num):
                rank_moe = share_num + expert_id
                x = x_list[rank_moe]
                combine_info = combine_info_list[rank_moe]

                mask = combine_info[:, 0] == rank
                x = x[mask]
                combine_info = combine_info[mask]

                for idx in range(len(combine_info)):
                    token_id = combine_info[idx, 1]
                    k_offset = combine_info[idx, 2]
                    moe_y[token_id, k_offset] = x[idx]
            DistributedTestGolden.save_tensor(moe_y, output_dir, f"moe_y_rank_{rank}.bin")

            # 计算的时候转化为fp32
            scale = scale_list[rank]
            moe_y = moe_y.to(dtype=torch.float32)
            moe_y = moe_y * scale
            DistributedTestGolden.save_tensor(moe_y, output_dir, f"scaled_moe_y_rank_{rank}.bin")

            y = torch.sum(moe_y, dim=1) + share_y.to(dtype=torch.float32)
            DistributedTestGolden.save_tensor(y.to(dtype=dtype), output_dir, f"y_rank_{rank}.bin")

    @staticmethod
    @GoldenRegister.reg_golden_func(case_names=[
            "DistributedTest.allgather_attn_post_reducescatter_b64_s1_n32_lora256_dim128_h128_rank4_bf16",
        ]
    )
    def allgather_attnpost_reducescatter_gen_func(case_name: str, output: Path) -> bool:
        if case_name == "DistributedTest.allgather_attn_post_reducescatter_b64_s1_n32_lora256_dim128_h128_rank4_bf16":
            b = 64
            s = 1
            n = 32
            kv_lora_rank = 256
            v_head_dim = 128
            h = 128
            rank_size = 4
            dtype = torch.bfloat16
            DistributedTestGolden.save_params([b, s, n, kv_lora_rank, v_head_dim, h], dtype, rank_size, output)
            DistributedTestGolden.gen_allgather_attnpost_reducescatter_case(
                b, s, n, kv_lora_rank, v_head_dim, h, rank_size, dtype, output
            )
        else:
            logging.info("Don't exist Case(%s).", case_name)
            return Flase

        logging.info("Case(%s), Golden generated success.", case_name)
        return True

    @staticmethod
    def gen_allgather_attnpost_reducescatter_case(b, s, n, kv_lora_rank, v_head_dim, h, rank_size, dtype, output):
        ag_in_list = []
        for rank in range(rank_size):
            ag_in = torch.randn([b * s * n // rank_size, kv_lora_rank], dtype=dtype)
            DistributedTestGolden.save_tensor(ag_in, output, f"ag_in_rank_{rank}.bin")
            ag_in_list.append(ag_in)

        attn_in = torch.cat(ag_in_list, dim=0).reshape([b, n, s, kv_lora_rank])
        attn_in = torch.transpose(attn_in, 1, 2)
        attn_in = torch.reshape(attn_in, [b * s, n, kv_lora_rank])
        attn_in = torch.transpose(attn_in, 0, 1)

        rs_in_list = []
        for rank in range(rank_size):
            w_lora = torch.randn([n, kv_lora_rank, v_head_dim], dtype=dtype)
            DistributedTestGolden.save_tensor(w_lora, output, f"w_lora_rank_{rank}.bin")

            attn_out = torch.bmm(attn_in.to(dtype=torch.float32), w_lora.to(dtype=torch.float32)).to(dtype=dtype)
            attn_out = torch.transpose(attn_out, 0, 1)
            attn_out = torch.reshape(attn_out, [b * s, n * v_head_dim])

            w_out = torch.randn([n * v_head_dim, h], dtype=dtype)
            DistributedTestGolden.save_tensor(w_out, output, f"w_out_rank_{rank}.bin")

            attn_out = torch.matmul(attn_out.to(dtype=torch.float32), w_out.to(dtype=torch.float32)).to(dtype=dtype)
            rs_in_list.append(attn_out)

        rs_out = torch.stack(rs_in_list, dim=0).to(torch.float32)
        rs_out = torch.sum(rs_out, dim=0).to(dtype)
        out_bs = b * s // rank_size
        for rank in range(rank_size):
            rank_rs_out = rs_out[rank * out_bs: (rank + 1) * out_bs]
            DistributedTestGolden.save_tensor(
                rank_rs_out, output, f"rs_out_rank_{rank}.bin"
            )


def main() -> bool:
    """
    单独调试 入口函数
    """

    # 用例名称
    case_name_list: List[str] = [
        "DistributedTest.aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = DistributedTestGolden.gen_combine(case_name=cs, output=output)
    return ret

if __name__ == "__main__":
    exit(0 if main() else 1)