from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn.functional as F

import pypto

def _eager_loop(*args, **kwargs):
    n = len(args)
    if n == 1:
        start, stop, step = 0, args[0], 1
    elif n == 2:
        start, stop, step = args[0], args[1], 1
    elif n == 3:
        start, stop, step = args
    else:
        raise TypeError(f"loop() takes 1 to 3 positional arguments but {n} were given")
    start = int(start); stop = int(stop); step = int(step) if step else 1
    return iter(range(start, stop, step))


def _eager_loop_unroll(*args, **kwargs):
    n = len(args)
    if n == 1:
        start, stop, step = 0, args[0], 1
    elif n == 2:
        start, stop, step = args[0], args[1], 1
    elif n >= 3:
        start, stop, step = args[0], args[1], args[2]
    else:
        raise TypeError("loop_unroll() requires at least 1 positional arg")
    start = int(start); stop = int(stop); step = int(step) if step else 1
    for i in range(start, stop, step):
        yield i, 1


def _eager_cond(scalar, *args, **kwargs):
    return bool(int(scalar))


@contextmanager
def _eager_function(name, *args, **kwargs):
    yield None


def _eager_is_loop_begin(scalar):
    return int(scalar) == getattr(scalar, "_loop_begin", 0)


def _eager_is_loop_end(scalar):
    end = getattr(scalar, "_loop_end", None)
    if end is None:
        return False
    return int(scalar) == int(end) - 1


def _eager_view(input, shape=None, offsets=None, *, valid_shape=None, dtype=None):
    if dtype is not None and shape is None and offsets is None:
        th_dtype = _pypto_dtype_to_torch(dtype)
        return input.to(th_dtype) if th_dtype is not None else input
    if shape is None:
        return input
    if offsets is None:
        offsets = [0] * len(shape)
    if valid_shape is None:
        slices = tuple(slice(int(o), int(o) + int(s)) for o, s in zip(offsets, shape))
        return input[slices].clone()
    # valid_shape: output is `shape`-sized; copy `valid_shape` region from offsets, rest zero
    shape_ints = [int(s) for s in shape]
    valid_ints = [int(v) for v in valid_shape]
    out = torch.zeros(shape_ints, dtype=input.dtype, device=input.device)
    src_slices = tuple(slice(int(o), int(o) + v) for o, v in zip(offsets, valid_ints))
    dst_slices = tuple(slice(0, v) for v in valid_ints)
    out[dst_slices] = input[src_slices]
    return out


def _eager_assemble(*args, parallel=False):
    """Emulate pypto.assemble — write src into dst at offsets (in-place on dst)."""
    if isinstance(args[0], (list, tuple)) and args[0] and isinstance(args[0][0], tuple):
        srcs, dst = args
        for src, offsets in srcs:
            slices = tuple(slice(int(o), int(o) + int(s)) for o, s in zip(offsets, src.shape))
            dst[slices] = src
        return None
    src, offsets, dst = args
    slices = tuple(slice(int(o), int(o) + int(s)) for o, s in zip(offsets, src.shape))
    dst[slices] = src
    return None


def _pypto_dtype_to_torch(dtype):
    """Map PyPTO datatype to torch.dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    name = getattr(dtype, "name", str(dtype)).replace("DataType.", "")
    return {
        "DT_FP32": torch.float32,
        "DT_FP16": torch.float16,
        "DT_BF16": torch.bfloat16,
        "DT_FP8":  torch.float16,
        "DT_INT4": torch.int8,
        "DT_INT8": torch.int8,
        "DT_INT16": torch.int16,
        "DT_INT32": torch.int32,
        "DT_INT64": torch.int64,
        "DT_UINT8": torch.uint8,
        "DT_UINT16": torch.int16,
        "DT_UINT32": torch.int32,
        "DT_UINT64": torch.int64,
        "DT_BOOL": torch.bool,
    }.get(name, None)


def _tensor_move(self, other):
    """pypto.Tensor.move shim: copy other's data into self (in-place write)."""
    if isinstance(other, torch.Tensor):
        if self.shape != other.shape:
            other = other.expand_as(self) if other.numel() == 1 or all(
                o in (1, s) for o, s in zip(other.shape, self.shape)
            ) else other.reshape(self.shape)
        return self.copy_(other)
    return self.fill_(other)


def _eager_tensor_factory(shape=None, dtype=None, name="", format=None,
                          data_ptr=None, device=None, ori_shape=None):
    """Eager-mode replacement for pypto.tensor(shape, dtype)."""
    th_dtype = _pypto_dtype_to_torch(dtype) or torch.float32
    shape_list = [int(s) for s in (shape or [])]
    return torch.zeros(shape_list, dtype=th_dtype)


def _eager_symbolic_scalar(value=0):
    return int(value)


def _to_plain_scalar(v):
    if isinstance(v, (int, float, bool)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _eager_full(size, fill_value, dtype=None, *, valid_shape=None):
    th_dtype = _pypto_dtype_to_torch(dtype)
    fv = _to_plain_scalar(fill_value)
    if th_dtype is not None:
        return torch.full(list(size), fv, dtype=th_dtype)
    return torch.full(list(size), fv)


def _eager_zeros(*size, dtype=None):
    th_dtype = _pypto_dtype_to_torch(dtype)
    if th_dtype is not None:
        return torch.zeros(*size, dtype=th_dtype)
    return torch.zeros(*size)


def _eager_ones(*size, dtype=None):
    th_dtype = _pypto_dtype_to_torch(dtype)
    if th_dtype is not None:
        return torch.ones(*size, dtype=th_dtype)
    return torch.ones(*size)


def _eager_cast(input, dtype, mode=None, satmode=None):
    th_dtype = _pypto_dtype_to_torch(dtype)
    if th_dtype is None:
        return input
    return input.to(th_dtype)


def _eager_matmul(input, mat2, out_dtype=None, *, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=None):
    if a_trans:
        input = input.transpose(-1, -2)
    if b_trans:
        mat2 = mat2.transpose(-1, -2)
    if input.dtype != mat2.dtype:
        common = torch.promote_types(input.dtype, mat2.dtype)
        input = input.to(common)
        mat2 = mat2.to(common)
    out = torch.matmul(input, mat2)
    th_out_dtype = _pypto_dtype_to_torch(out_dtype)
    if th_out_dtype is not None and out.dtype != th_out_dtype:
        out = out.to(th_out_dtype)
    return out


def _cbrt(x):
    return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


def _ceil_div(a, b):
    if isinstance(a, torch.Tensor) and a.is_floating_point():
        return torch.ceil(torch.div(a, b))
    if isinstance(b, (int, float)):
        b_t = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    else:
        b_t = b
    return torch.div(a + b_t - 1, b_t, rounding_mode="trunc")


def _scatter(inp, dim, index, src, *, reduce=None):
    index = index.long()
    if reduce is None:
        return torch.scatter(inp, dim, index, src)
    return torch.scatter(inp, dim, index, src, reduce=reduce)


def _scatter_(inp, dim, index, src, *, reduce=None):
    index = index.long()
    if reduce is None:
        return inp.scatter_(dim, index, src)
    return inp.scatter_(dim, index, src, reduce=reduce)


def _topk(x, k, dim=None, largest=True):
    return torch.topk(x, k, dim=(-1 if dim is None else dim), largest=largest)


def _argsort(x, dim=None, descending=False):
    return torch.argsort(x, dim=(-1 if dim is None else dim), descending=descending)


_OP_TABLE = {
    "amax":   lambda input, dim, keepdim=False: torch.amax(input, dim=dim, keepdim=keepdim),
    "amin":   lambda input, dim, keepdim=False: torch.amin(input, dim=dim, keepdim=keepdim),
    "sum":    lambda input, dim, keepdim=False: torch.sum(input, dim=dim, keepdim=keepdim),
    "mean":   lambda input, dim, keepdim=False: torch.mean(input, dim=dim, keepdim=keepdim),
    "prod":   lambda self, dim, keepdim=False: torch.prod(self, dim=dim, keepdim=keepdim),
    "argmax": lambda input, dim=-1, keepdim=False: torch.argmax(input, dim=dim, keepdim=keepdim),
    "argmin": lambda input, dim=-1, keepdim=False: torch.argmin(input, dim=dim, keepdim=keepdim),
    "cumsum":  lambda input, dim: torch.cumsum(input, dim=dim),
    "cumprod": lambda input, dim: torch.cumprod(input, dim=dim),
    "var": lambda input, dim=None, correction=1, keepdim=False: torch.var(input, dim=dim, correction=correction, keepdim=keepdim),

    "maximum": lambda input, other: torch.maximum(input, other),
    "minimum": lambda input, other: torch.minimum(input, other),

    "add": lambda input, other, *, alpha=1: torch.add(input, other, alpha=alpha),
    "sub": lambda input, other, *, alpha=1: torch.sub(input, other, alpha=alpha),
    "mul": lambda input, other: torch.mul(input, other),
    "div": lambda input, other, precision_type=None: torch.div(input, other),
    "pow": lambda input, other: torch.pow(input, other),
    "neg": lambda a: torch.neg(a),
    "abs": lambda a: torch.abs(a),
    "sign":      lambda a: torch.sign(a),
    "signbit":   lambda a: torch.signbit(a),
    "reciprocal": lambda a, precision_type=None: torch.reciprocal(a),
    "hypot":     lambda self, other: torch.hypot(self, other),
    "fmod":      lambda input, other: torch.fmod(input, other),
    "remainder": lambda input, other: torch.remainder(input, other),
    "copysign":  lambda input, other: torch.copysign(input, other),
    "isfinite":  lambda self: torch.isfinite(self),
    "cbrt":      lambda self: _cbrt(self),
    "gcd":       lambda input, other: torch.gcd(input, other),
    "ceil_div":  lambda self, other: _ceil_div(self, other),
    "floor_div": lambda input, other: torch.div(input, other, rounding_mode="floor"),
    "round":     lambda input, decimals=0: torch.round(input, decimals=decimals),
    "ceil":      lambda input: torch.ceil(input),
    "floor":     lambda input: torch.floor(input),
    "trunc":     lambda input: torch.trunc(input),
    "clip":      lambda input, min=None, max=None: torch.clamp(input, min=min, max=max),

    "exp":   lambda input, precision_type=None: torch.exp(input),
    "exp2":  lambda input: torch.exp2(input),
    "expm1": lambda input: torch.expm1(input),
    "log":   lambda input, precision_type=None: torch.log(input),
    "log2":  lambda input, precision_type=None: torch.log2(input),
    "log10": lambda input, precision_type=None: torch.log10(input),
    "log1p": lambda input: torch.log1p(input),
    "sqrt":  lambda input, precision_type=None: torch.sqrt(input),
    "rsqrt": lambda input, precision_type=None: torch.rsqrt(input),

    "sin": lambda x: torch.sin(x),
    "cos": lambda x: torch.cos(x),
    "tan": lambda x: torch.tan(x),
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "relu": lambda a: torch.relu(a),
    "lrelu": lambda other, negative_slope=0.01: F.leaky_relu(other, negative_slope=negative_slope),
    "prelu": lambda self, weight: F.prelu(self, weight),

    "logical_not": lambda input: torch.logical_not(input),
    "logical_and": lambda input, other: torch.logical_and(input, other),
    "bitwise_and": lambda self, other: torch.bitwise_and(self, other),
    "bitwise_or":  lambda input1, input2: torch.bitwise_or(input1, input2),
    "bitwise_xor": lambda first, second: torch.bitwise_xor(first, second),
    "bitwise_not": lambda self: torch.bitwise_not(self),
    "bitwise_left_shift":  lambda input, other: torch.bitwise_left_shift(input, other),
    "bitwise_right_shift": lambda input, other: torch.bitwise_right_shift(input, other),

    "greater": lambda input, other: torch.gt(input, other),
    "gt": lambda input, other: torch.gt(input, other),
    "ge": lambda input, other: torch.ge(input, other),
    "eq": lambda input, other: torch.eq(input, other),
    "ne": lambda input, other: torch.ne(input, other),
    "lt": lambda input, other: torch.lt(input, other),
    "le": lambda input, other: torch.le(input, other),
    "topk":    _topk,
    "argsort": _argsort,

    "arange": lambda *args: torch.arange(*args),
    "full":   _eager_full,
    "zeros":  _eager_zeros,
    "ones":   _eager_ones,

    "concat": lambda tensors, dim=0: torch.cat(list(tensors), dim=dim),

    "permute":   lambda input_tensor, dims: input_tensor.permute(list(dims)).contiguous(),
    "transpose": lambda input, dim0, dim1: torch.transpose(input, dim0, dim1).contiguous(),
    "expand_clone": lambda input, shape, *, valid_shape=None: input.expand(list(shape)).clone(),
    "triu": lambda input, diagonal=0: torch.triu(input, diagonal=diagonal),
    "tril": lambda input, diagonal=0: torch.tril(input, diagonal=diagonal),

    "gather":         lambda input, dim, index: torch.gather(input, dim, index.long()),
    "index_select":   lambda input, dim, index: torch.index_select(input, dim, index.long()),
    "scatter":        _scatter,
    "scatter_":       _scatter_,
    "scatter_update": lambda input, dim, index, src: torch.scatter(input, dim, index, src),
    "index_add":  lambda input, dim, index, source, *, alpha=1: torch.index_add(input, dim, index.long(), source, alpha=alpha),
    "index_add_": lambda input, dim, index, source, *, alpha=1: input.index_add_(dim, index.long(), source, alpha=alpha),
    "index_put_": lambda input, indices, values, accumulate=False: input.index_put_(list(indices), values, accumulate=accumulate),

    "where":   lambda condition, input, other: torch.where(condition, input, other),
    "pad":     lambda input, pad, mode="constant", value=0.0: F.pad(input, list(pad), mode=mode, value=value),
    "fillpad": lambda input, mode="constant", value=0.0: input,  # no valid_shape in eager tensors
    "one_hot": lambda input, num_classes: F.one_hot(input, num_classes=num_classes),
    "softmax": lambda x, dim=-1: torch.softmax(x, dim=dim),
    "expand_exp_dif": lambda input, other: torch.exp(input - other),

    "matmul": _eager_matmul,
    "cast":   _eager_cast,

    "view":      _eager_view,
    "assemble":  _eager_assemble,
    "reshape":   lambda input, shape, *, valid_shape=None, inplace=False: input.reshape(list(shape)),
    "clone":     lambda input: input.clone(),
    "unsqueeze": lambda input, dim: torch.unsqueeze(input, dim),
    "min":       lambda a, b: min(int(a), int(b)),
    "max":       lambda a, b: max(int(a), int(b)),
    "tensor":    _eager_tensor_factory,
    "symbolic_scalar": _eager_symbolic_scalar,

    # set tile shapes are no ops in eager mode
    "set_vec_tile_shapes":  lambda *a, **kw: None,
    "set_cube_tile_shapes": lambda *a, **kw: None,
    "set_conv_tile_shapes": lambda *a, **kw: None,
    "set_matrix_size":      lambda *a, **kw: None,
    "set_build_static":     lambda *a, **kw: None,
    "set_scope":            lambda *a, **kw: None,

    "loop":           _eager_loop,
    "loop_unroll":    _eager_loop_unroll,
    "cond":           _eager_cond,
    "function":       _eager_function,
    "is_loop_begin":  _eager_is_loop_begin,
    "is_loop_end":    _eager_is_loop_end,
}


@contextmanager
def eager_mode(extra_patches: Iterable[tuple] = ()):
    """
    Use torch equivalents for PyPTO ops during eager-mode execution.
    """
    saved = {}
    patches = dict(_OP_TABLE)
    patches.update(dict(extra_patches))

    for name, fn in patches.items():
        if hasattr(pypto, name):
            saved[name] = getattr(pypto, name)
            setattr(pypto, name, fn)

    had_move = hasattr(torch.Tensor, "move")
    prev_move = getattr(torch.Tensor, "move", None)
    torch.Tensor.move = _tensor_move

    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(pypto, name, fn)
        if had_move:
            torch.Tensor.move = prev_move
        else:
            try:
                del torch.Tensor.move
            except AttributeError:
                pass
