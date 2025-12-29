from . import pypto_impl
from typing import Union, Sequence, List
from .scalar import Scalar

__all__ = [
    "to_sym",
    "to_syms",
]

def to_sym(value) -> pypto_impl.Scalar:
    if isinstance(value, int):
        return pypto_impl.Scalar(value)
    elif isinstance(value, pypto_impl.Scalar):
        return value
    elif isinstance(value, Scalar):
        return value.base()
    else:
        raise ValueError("Invalid value type")

def to_syms(value: Union[Sequence[int], "Sequence[Scalar]"]) -> List[Scalar]:
    return [to_sym(v) for v in value]