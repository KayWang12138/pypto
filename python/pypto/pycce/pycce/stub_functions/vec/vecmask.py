from ...utils import Var, Instruction, DT
from ... import context
from typing import Union


def set_mask(maskHigh: Union[int, Var], maskLow: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    if isinstance(maskHigh, Var):
        assert maskHigh.dtype==DT.uint64
    if isinstance(maskLow, Var):
        assert maskLow.dtype==DT.uint64

    g_vec.append(Instruction('SETMASK', low=maskLow, high=maskHigh))
    

def reset_mask():
    g_vec = context.active_vec
    assert g_vec is not None 

    g_vec.append(Instruction('RESETMASK'))


def set_continuous_mask(count: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    if isinstance(count, Var) and count.dtype != DT.uint64:
        raise TypeError("The dtype of Var count in set_continuous_mask must be uint64!")

    g_vec.append(Instruction('SET_CONTINUOUS_MASK', count = count))
