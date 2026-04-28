#!/usr/bin/env python3
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F

TensorList = list[torch.Tensor]

@dataclasses.dataclass(frozen=True)
class MoeCase:
    batch_size: int
    hidden_size: int
    moe_expert_num: int
    topk: int
    data_type: str
    ep_world_size: int

np.random.seed(0)
torch.manual_seed(0)

def generate_random_tensor(shape, dtype):
    float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if dtype in float_dtypes:
        return torch.randn(shape, dtype=dtype)
    elif dtype in int_dtypes:
        return torch.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')

def generate_inputs(moe_case, torch_data_type):
    x_list = [
        generate_random_tensor((moe_case.batch_size, moe_case.hidden_size), torch_data_type)
        for _ in range(moe_case.ep_world_size)
    ]
    moe_expert_ids_list = []
    topk_expert_scales_list = []
    
    for _ in range(moe_case.ep_world_size):
        expert_scores = generate_random_tensor((moe_case.batch_size, moe_case.moe_expert_num), torch.float32)
        topk_expert_scores, moe_expert_ids = expert_scores.topk(k=moe_case.topk)
        topk_expert_scales = topk_expert_scores.softmax(dim=-1)
        topk_expert_scales_list.append(topk_expert_scales)
        moe_expert_ids = moe_expert_ids.to(dtype=torch.int32)
        moe_expert_ids_list.append(moe_expert_ids)
    
    return x_list, moe_expert_ids_list, topk_expert_scales_list

def get_moe_expert_num_per_rank(moe_case):
    return (moe_case.moe_expert_num + moe_case.ep_world_size - 1) // moe_case.ep_world_size

def dispatch_tokens(moe_case, torch_data_type, x_list, moe_expert_ids_list):
    moe_expert_num_per_rank = get_moe_expert_num_per_rank(moe_case)
    expand_x_per_expert = [[[] for _ in range(moe_expert_num_per_rank)] for _ in range(moe_case.ep_world_size)]
    assist_info_for_combine_per_expert = [
        [[] for _ in range(moe_expert_num_per_rank)]
        for _ in range(moe_case.ep_world_size)
    ]
    
    for sending_rank_id, (x, moe_expert_ids) in enumerate(zip(x_list, moe_expert_ids_list)):
        for token_id, (token, topk_moe_expert_ids) in enumerate(zip(x, moe_expert_ids)):
            for k_offset, moe_expert_id in enumerate(topk_moe_expert_ids):
                receiving_expert_id = moe_expert_id.item()
                receiving_rank_id, expert_offset = divmod(receiving_expert_id, moe_expert_num_per_rank)
                expand_x_per_expert[receiving_rank_id][expert_offset].append(token)
                assist_info_for_combine_per_expert[receiving_rank_id][expert_offset].append((sending_rank_id, token_id, k_offset))
    
    row = min(moe_case.topk * moe_case.batch_size * moe_case.ep_world_size, moe_case.batch_size * moe_case.moe_expert_num)
    combine_info_col = 3
    expand_x_per_rank = []
    assist_info_for_combine_per_rank = []
    expert_token_nums_per_rank = []
    recv_counts_per_rank = []
    
    for logical_rank_id in range(moe_case.ep_world_size):
        fixed_shape_expand_x = torch.zeros((row, moe_case.hidden_size), dtype=torch_data_type)
        fixed_shape_assist_info_for_combine = torch.zeros((row, combine_info_col), dtype=torch.int32)
        expert_token_nums = torch.zeros([moe_expert_num_per_rank], dtype=torch.int32)
        offset = 0
        
        for expert_offset in range(moe_expert_num_per_rank):
            tokens = expand_x_per_expert[logical_rank_id][expert_offset]
            if tokens:
                actual_expand_x = torch.stack(tokens, dim=0)
                end = offset + actual_expand_x.size(0)
                fixed_shape_expand_x[offset:end] = actual_expand_x
                actual_assist_info_for_combine = torch.tensor(
                    assist_info_for_combine_per_expert[logical_rank_id][expert_offset],
                    dtype=torch.int32,
                )
                fixed_shape_assist_info_for_combine[offset:end, :] = actual_assist_info_for_combine
                expert_token_nums[expert_offset] = actual_expand_x.size(0)
                offset = end
        
        expand_x_per_rank.append(fixed_shape_expand_x)
        assist_info_for_combine_per_rank.append(fixed_shape_assist_info_for_combine)
        expert_token_nums_per_rank.append(expert_token_nums)
        recv_counts_per_rank.append(expert_token_nums.sum(dtype=torch.int32).unsqueeze(0))
    
    return expand_x_per_rank, assist_info_for_combine_per_rank, expert_token_nums_per_rank, recv_counts_per_rank

def dispatch_tokens_v2(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
    x_list: TensorList,
    expert_ids_list: TensorList,
) -> tuple[TensorList, TensorList, TensorList, TensorList]:
    expert_num_per_rank = get_moe_expert_num_per_rank(moe_case)
    total_send_tasks = moe_case.batch_size * moe_case.topk
    
    sending_rank_cumsum_tables_list = []
    sending_rank_token_counts_list = []
    for expert_ids in expert_ids_list:
        expert_ids_flat = expert_ids.flatten().to(torch.long)
        one_hot_table = F.one_hot(expert_ids_flat, num_classes=moe_case.moe_expert_num)
        cumsum_table = torch.cumsum(one_hot_table, dim=0)
        sending_rank_cumsum_tables_list.append(cumsum_table)
        sending_rank_token_counts_list.append(cumsum_table[-1, :])
    
    receive_rank_token_counts_list = [torch.zeros(moe_case.moe_expert_num + 1, dtype=torch.int32) for _ in range(moe_case.ep_world_size)]
    
    for sending_rank_id, token_count_per_expert in enumerate(sending_rank_token_counts_list):
        for expert_id in range(moe_case.moe_expert_num):
            receiving_rank_id, expert_offset = divmod(expert_id, expert_num_per_rank)
            receive_rank_token_counts_list[receiving_rank_id][expert_offset * moe_case.ep_world_size + sending_rank_id + 1] = token_count_per_expert[expert_id]
    
    token_counts_tensor = torch.stack(receive_rank_token_counts_list)
    cumsum_result = torch.cumsum(token_counts_tensor, dim=1)
    recv_counts_tensor = cumsum_result[:, -1].unsqueeze(1)
    recv_counts_list = [recv_counts_tensor[i] for i in range(moe_case.ep_world_size)]
    
    indices = torch.arange(0, moe_case.moe_expert_num + 1, moe_case.ep_world_size)
    start_indices = indices[:-1]
    end_indices = indices[1:]
    expert_token_nums_tensor = cumsum_result[:, end_indices] - cumsum_result[:, start_indices]
    expert_token_nums_list = [expert_token_nums_tensor[i] for i in range(moe_case.ep_world_size)]
    
    row = min(moe_case.topk * moe_case.batch_size * moe_case.ep_world_size, moe_case.batch_size * moe_case.moe_expert_num)
    expand_x_list = [torch.zeros((row, moe_case.hidden_size), dtype=torch_data_type) for _ in range(moe_case.ep_world_size)]
    assist_info_for_combine_list = [torch.zeros((row, 3), dtype=torch.int32) for _ in range(moe_case.ep_world_size)]
    
    for sending_rank_id, (x, cumsum_table) in enumerate(zip(x_list, sending_rank_cumsum_tables_list)):
        for index in range(moe_case.batch_size * moe_case.topk):
            token_id, k_offset = divmod(index, moe_case.topk)
            expert_id = expert_ids_list[sending_rank_id][token_id, k_offset].item()
            receiving_rank_id, expert_offset = divmod(expert_id, expert_num_per_rank)
            cumsum_offset = 0 if index == 0 else cumsum_table[index - 1, expert_id]
            token_offset = cumsum_result[receiving_rank_id, expert_offset * moe_case.ep_world_size + sending_rank_id] + cumsum_offset
            expand_x_list[receiving_rank_id][token_offset] = x[token_id]
            assist_info_for_combine_list[receiving_rank_id][token_offset] = torch.tensor([sending_rank_id, token_id, k_offset], dtype=torch.int32)
    
    return expand_x_list, assist_info_for_combine_list, expert_token_nums_list, recv_counts_list

moe_case = MoeCase(8, 5120, 160, 8, 'BF16', 4)
torch_data_type = torch.bfloat16

x_list, moe_expert_ids_list, _ = generate_inputs(moe_case, torch_data_type)

expand_x_list_1, assist_info_list_1, expert_token_nums_list_1, recv_counts_list_1 = dispatch_tokens(moe_case, torch_data_type, x_list, moe_expert_ids_list)
expand_x_list_2, assist_info_list_2, expert_token_nums_list_2, recv_counts_list_2 = dispatch_tokens_v2(moe_case, torch_data_type, x_list, moe_expert_ids_list)

for rank_id, (assist_1, assist_2) in enumerate(zip(assist_info_list_1, assist_info_list_2)):
    recv_count = recv_counts_list_1[rank_id].item()
    all_match = torch.all(assist_1[:recv_count] == assist_2[:recv_count]).item()
    print(f"Rank {rank_id}: assist_info_for_combine match = {all_match}")
    if not all_match:
        for i in range(min(recv_count, 5)):
            print(f"  index {i}: v1={assist_1[i].tolist()}, v2={assist_2[i].tolist()}")

for rank_id, (expand_x_1, expand_x_2) in enumerate(zip(expand_x_list_1, expand_x_list_2)):
    recv_count = recv_counts_list_1[rank_id].item()
    all_match = torch.allclose(expand_x_1[:recv_count], expand_x_2[:recv_count])
    print(f"Rank {rank_id}: expand_x match = {all_match}")
    if not all_match:
        diff_count = (expand_x_1[:recv_count] != expand_x_2[:recv_count]).sum().item()
        print(f"  diff_count = {diff_count}")
        for i in range(min(recv_count, 10)):
            print(f"  index {i}: v1={expand_x_1[i, 0].item():.4f}, v2={expand_x_2[i, 0].item():.4f}")

for rank_id, (recv_1, recv_2) in enumerate(zip(recv_counts_list_1, recv_counts_list_2)):
    print(f"Rank {rank_id}: dispatch_tokens recv_count = {recv_1.item()}, dispatch_tokens_v2 recv_count = {recv_2.item()}, match = {recv_1.item() == recv_2.item()}")

for rank_id, (token_nums_1, token_nums_2) in enumerate(zip(expert_token_nums_list_1, expert_token_nums_list_2)):
    all_match = torch.all(token_nums_1 == token_nums_2).item()
    print(f"Rank {rank_id}: expert_token_nums match = {all_match}")
    if not all_match:
        diff = (token_nums_1 != token_nums_2).nonzero()
        for idx in diff[:10]:
            i = idx.item()
            print(f"  expert_id={i}: v1={token_nums_1[i].item()}, v2={token_nums_2[i].item()}")