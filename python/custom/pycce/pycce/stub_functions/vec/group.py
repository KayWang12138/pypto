from ...utils import Tensor, Var, Instruction, Position, DT, ReduceModeInst, ReduceMode
from ... import context
from typing import Union


# group functions 
def cmax(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var], mode: ReduceModeInst=ReduceMode.ONLY_VALUE):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CMAX] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCMAX', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride, mode=mode))
    

def cgmax(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGMAX] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGMAX', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))


def cmin(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var], mode: ReduceModeInst=ReduceMode.ONLY_VALUE):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CMIN] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCMIN', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride, mode=mode))
    

def cgmin(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGMIN] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGMIN', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    

def cadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    

def cgadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))


def cpadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCPADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
