import torch
import torch.distributed as dist
import torch_npu
import os

def inspect():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("hccl", rank=0, world_size=1)
    torch.npu.set_device(0)
    
    pg = dist.group.WORLD
    print(f"PG type: {type(pg)}")
    print(f"PG dir: {dir(pg)}")
    
    print(f"torch_npu.distributed dir: {dir(torch_npu.distributed)}")
    
    if hasattr(torch_npu.distributed, 'backend'):
        print(f"torch_npu.distributed.backend dir: {dir(torch_npu.distributed.backend)}")

if __name__ == "__main__":
    inspect()



'''
PG type: <class 'torch.distributed.distributed_c10d.ProcessGroup'>
PG dir: ['BackendType', 'CUSTOM', 'GLOO', 'MPI', 'NCCL', 'UCC', 'UNDEFINED', 'XCCL', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_allgather_base', '_backend_id', '_device_types', '_enable_collectives_timing', '_end_coalescing', '_get_backend', '_get_backend_name', '_get_sequence_number_for_group', '_has_hooks', '_id', '_pybind11_conduit_v1_', '_reduce_scatter_base', '_register_backend', '_register_on_completion_hook', '_set_default_backend', '_set_group_desc', '_set_group_name', '_set_sequence_number_for_group', '_start_coalescing', '_wait_for_pending_works', 'abort', 'allgather', 'allgather_coalesced', 'allgather_into_tensor_coalesced', 'allreduce', 'allreduce_coalesced', 'alltoall', 'alltoall_base', 'barrier', 'bound_device_id', 'boxed', 'broadcast', 'gather', 'group_desc', 'group_name', 'monitored_barrier', 'name', 'rank', 'recv', 'recv_anysource', 'reduce', 'reduce_scatter', 'reduce_scatter_tensor_coalesced', 'scatter', 'send', 'shutdown', 'size', 'unbox']
torch_npu.distributed dir: ['ErrCode', 'ParallelStore', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_is_support_hccl_comm_name', '_verify_params_across_processes', 'all_gather_into_tensor_uneven', 'dist_error', 'distributed_c10d', 'fsdp', 'is_available', 'is_hccl_available', 'reduce_scatter_tensor_uneven', 'reinit_process_group', 'rendezvous', 'rpc', 'run', 'tensor', 'torch_npu']
'''