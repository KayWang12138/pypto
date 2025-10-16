from ...utils import Tensor, GMTensor, Var, Instruction, Position
from ... import context
from typing import Union


# Data move
def gm_to_ub(dst: Tensor, src: GMTensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call gm_to_ub within vector forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, GMTensor), 'src must be GMTensor on GM'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('GM2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_gm(dst: GMTensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_gm within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, GMTensor), 'dst must be GMTensor on GM'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2GM', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_ub(dst: Tensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))

