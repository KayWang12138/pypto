"""PyPTO"""
import functools
from . import pypto_impl
from .tensor import Tensor, Scalar
    
def _to_base(arg):
    if isinstance(arg, (Tensor, Scalar)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _to_base(v) for k, v in arg.items()}
    else:
        return arg


def _from_base(out):
    if isinstance(out, pypto_impl.Tensor):
        return Tensor.from_base(out)
    elif isinstance(out, (list, tuple)):
        return [_from_base(a) for a in out]
    elif isinstance(out, dict):
        return {k: _from_base(v) for k, v in out.items()}
    else:
        return out

def op_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = _to_base(args)
        kwargs = _to_base(kwargs)
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"args must be list or tuple, but got {type(args)}.")
        # set_source_location()
        out = func(*args, **kwargs)
        # clear_source_location()
        if out is None:
            return None
        else:
            return _from_base(out)

    return wrapper