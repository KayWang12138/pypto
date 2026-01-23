from __future__ import annotations

import abc
import builtins
import functools
import io
import math
import operator
from numbers import Real
from typing import Any, Callable, Iterable, Optional, Tuple, Type, TypeVar, Union, List, overload
from typing_extensions import Self, TypeAlias

import numpy as np
import pypto
import torch

from ..log import get_logger
from .compound import Arange, CompoundMask, CompoundSentinel, TensorPointer, TensorWithOffset
from .errors import NonAffineLayoutError
from . import dtypes, pypto_wrap

T = TypeVar("T")
IntArrayLike: TypeAlias = Union[int, Tuple[int, ...], np.ndarray]

logger = get_logger("triton_pypto.language", "TRITON_PYPTO")


class Context:
    program_id = (0, 0, 0)
    num_programs = (1, 1, 1)
    dynamic = False


def expand_impl(tensor: Union[TensorWrapper, pypto.tensor], target_shape: Iterable[int]) -> TensorWrapper:
    target_shape = np.array(target_shape, dtype=np.int32)
    shape = np.array(tensor.shape, dtype=np.int32)
    prev_shape = shape.copy()
    shape_it = np.nditer([shape, target_shape], flags=["f_index"])
    for dim, target_dim in shape_it:
        if dim != target_dim:
            prev_shape[shape_it.index] = target_dim
            tensor = TensorWrapper(pypto_wrap.expand_clone(tensor, prev_shape.tolist()))
    return tensor


def common_broadcast(a: Union[TensorWrapper, Real], b: Union[TensorWrapper, Real],
                     keep_scalar: bool = False) -> Tuple[Union[TensorWrapper, Real], Union[TensorWrapper, Real]]:
    if isinstance(a, TensorElementWrapper):
        a = TensorWrapper(a.unwrap())
    if isinstance(b, TensorElementWrapper):
        b = TensorWrapper(b.unwrap())
    if keep_scalar and (not isinstance(a, TensorWrapper) or not isinstance(b, TensorWrapper)):
        return a, b
    a_shape = np.array(a.shape, dtype=np.int32)
    b_shape = np.array(b.shape, dtype=np.int32)
    if np.array_equal(a_shape, b_shape):
        pypto_wrap.auto_vec_tile(a_shape, a.dtype)
        return a, b
    # Prepend ones if needed
    if a_shape.size != b_shape.size:
        common_size = builtins.max(a_shape.size, b_shape.size)
        if common_size != a_shape.size:
            a_shape = np.insert(a_shape, 0, [1] * (common_size - a_shape.size))
            pypto_wrap.auto_vec_tile(a_shape, a.dtype)
            a = TensorWrapper(pypto_wrap.reshape(a, a_shape.tolist()))
        if common_size != b_shape.size:
            b_shape = np.insert(b_shape, 0, [1] * (common_size - b_shape.size))
            pypto_wrap.auto_vec_tile(b_shape, b.dtype)
            b = TensorWrapper(pypto_wrap.reshape(b, b_shape.tolist()))
    a_nonones = a_shape != 1
    b_nonones = b_shape != 1
    # Differ only with ones -> pypto.reshape
    if np.array_equal(a_nonones, b_nonones):
        a.auto_vec_tile()
        b = TensorWrapper(pypto_wrap.reshape(b, a.shape))
        return a, b
    # Broadcastable into each other -> pypto.expand_clone
    if np.any(a_nonones & b_nonones & (a_shape != b_shape)):
        raise RuntimeError(f"Unable to common broadcast {a!r} to {b!r}")
    common_shape = np.where(a_nonones, a_shape, b_shape)
    common_type = dtypes.common_type(a.dtype, b.dtype)
    pypto_wrap.auto_vec_tile(common_shape, common_type)
    return expand_impl(a, common_shape), expand_impl(b, common_shape)


def pad_shape(shape: List[int], required_rank: int) -> List[int]:
    rank = len(shape)
    if rank == required_rank:
        return shape
    if rank < required_rank:
        return [1] * (required_rank - rank) + shape
    shift = rank - required_rank
    if not all(s == 1 for s in shape[:shift]):
        raise RuntimeError(f"Unable to pad shape {shape} to rank {required_rank}")
    return shape[shift:]


def reshape_impl(tensor: Union[TensorWrapper, pypto.tensor], shape: List[int]) -> TensorWrapper:
    src_shape = tensor.shape
    if src_shape == shape:
        return TensorWrapper(pypto_wrap.unwrap(tensor))
    if len(shape) > len(src_shape):
        pypto_wrap.auto_vec_tile(shape, tensor.dtype)
    else:
        pypto_wrap.auto_vec_tile(src_shape, tensor.dtype)
    return TensorWrapper(pypto_wrap.reshape(tensor, shape))


def tensor_binary_op(
    lhs: Union[TensorWrapper, Real, pypto.symbolic_scalar],
    rhs: Union[TensorWrapper, Real, pypto.symbolic_scalar],
    op: Callable[..., pypto.tensor],
    name: str,
    keep_scalar: bool = True,
) -> TensorWrapper:
    logger.debug("Tensor %s %s %s", name, lhs, rhs)
    lhs_tensor = isinstance(lhs, TensorWrapper)
    rhs_tensor = isinstance(rhs, TensorWrapper)
    if not lhs_tensor and not rhs_tensor:
        raise RuntimeError(f"Unexpected operands in binary {name}: {lhs!r}, {rhs!r}")
    if not lhs_tensor:
        lhs = rhs.full_like(lhs)
    if not rhs_tensor and not keep_scalar:
        rhs = lhs.full_like(rhs)
    if lhs_tensor and rhs_tensor:
        common_type = dtypes.common_type(lhs.dtype, rhs.dtype)
        lhs = cast(lhs, common_type)
        rhs = cast(rhs, common_type)
    lhs, rhs = common_broadcast(lhs, rhs, keep_scalar)
    result = op(lhs, rhs)
    return TensorWrapper(result)


def log_call(fn: T) -> T:

    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        log_str = io.StringIO()
        print(fn.__name__, *args, *(f"{name}={arg}" for name, arg in kwds.items()), end="", file=log_str)
        logger.debug(log_str.getvalue())
        return fn(*args, **kwds)

    return wrapper


def first_or_all(*values: Union[T, Iterable[T]], name: str = "values") -> Tuple[T]:
    if not values:
        raise RuntimeError(f"{name} must be provided")
    if isinstance(values[0], Iterable):
        values = tuple(values[0])
    return values


class TensorWrapper(pypto_wrap.BaseWrapper[pypto.tensor]):

    def __init__(self, tensor: Optional[pypto.tensor] = None, name: Optional[str] = None):
        super().__init__(tensor)
        if self.base is not None and name is not None:
            self.base.name = name

    def __repr__(self) -> str:
        return f"TensorWrapper(tensor={self.base!r})"

    def __str__(self) -> str:
        return repr(self)

    @property
    def shape(self) -> List[int]:
        return self.base.shape

    @property
    def dtype(self) -> dtypes.DataTypeInfo:
        return dtypes.to_info(self.base.dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def T(self) -> Self:
        return trans(self)

    def __add__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.add, "add")

    def __radd__(self, other) -> Self:
        return tensor_binary_op(other, self, pypto_wrap.add, "radd")

    def __sub__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.sub, "sub")

    def __rsub__(self, other) -> Self:
        return tensor_binary_op(other, self, pypto_wrap.sub, "rsub")

    def __mul__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.mul, "mul")

    def __rmul__(self, other) -> Self:
        return tensor_binary_op(other, self, pypto_wrap.mul, "rmul")

    def __truediv__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.div, "truediv")

    def __rtruediv__(self, other) -> Self:
        return tensor_binary_op(other, self, pypto_wrap.div, "rtruediv")

    def __and__(self, other) -> Self:
        if self.dtype != pypto.DT_BOOL:
            raise RuntimeError("Bitwise AND is supported only for DT_BOOL")
        return tensor_binary_op(self, other, pypto_wrap.logical_and, "and", keep_scalar=False)

    def __rand__(self, other) -> Self:
        if self.dtype != pypto.DT_BOOL:
            raise RuntimeError("Bitwise AND is supported only for DT_BOOL")
        return tensor_binary_op(other, self, pypto_wrap.logical_and, "rand", keep_scalar=False)

    def __or__(self, other) -> Self:
        return ~(~self & ~other)

    def __ror__(self, other) -> Self:
        return self.__or__(other)

    def __xor__(self, other) -> Self:
        return (self & ~other) | (~self & other)

    def __rxor__(self, other) -> Self:
        return self.__xor__(other)

    def __le__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.le, "le", keep_scalar=False)

    def __lt__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.lt, "lt", keep_scalar=False)

    def __eq__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.eq, "eq", keep_scalar=False)

    def __ne__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.ne, "ne", keep_scalar=False)

    def __gt__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.gt, "gt", keep_scalar=False)

    def __ge__(self, other) -> Self:
        return tensor_binary_op(self, other, pypto_wrap.ge, "ge", keep_scalar=False)

    def __neg__(self) -> Self:
        info = dtypes.to_info(self.dtype)
        if info.is_float():
            return self * (-1.0)
        if info.is_int():
            return self * (-1)
        raise RuntimeError(f"Negation is not supported for dtype {self.dtype}")

    def __pos__(self) -> Self:
        return self

    @log_call
    def __invert__(self) -> Self:
        if self.dtype != pypto.DT_BOOL:
            raise RuntimeError("Bitwise NOT is supported only for DT_BOOL")
        self.auto_vec_tile()
        return TensorWrapper(pypto_wrap.logical_not(self))

    def __getitem__(self, slices) -> Self:
        if not isinstance(slices, tuple):
            slices = (slices, )
        if not all(s is None or isinstance(s, slice) for s in slices):
            raise TypeError(f"All in {slices!r} must be either slice or None")
        axes = tuple(i for i, k in enumerate(slices) if k is None)
        return expand_dims(self, axes)

    def to(self, dtype: dtypes.AnyDataType) -> Self:
        return cast(self, dtype)

    def full_like(self, value: Real) -> Self:
        self.auto_vec_tile()
        return TensorWrapper(pypto_wrap.full(self.shape, value, dtypes.to_pypto(self.dtype)))

    def auto_vec_tile(self, buf_num: int = 2) -> None:
        pypto_wrap.auto_vec_tile(self.shape, self.dtype, buf_num=buf_num)


@overload
def bind_tensor_method(fn: T) -> T:
    ...


@overload
def bind_tensor_method(*, name: Optional[str] = None) -> Callable[[T], T]:
    ...


def bind_tensor_method(fn: Optional[T] = None, *, name: Optional[str] = None):

    def decorator(fn: T) -> T:
        method_name = name or fn.__name__
        setattr(TensorWrapper, method_name, fn)
        return fn

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


class HostTensorWrapper(TensorWrapper):

    def __init__(self, tensor: pypto.tensor, storage: torch.Tensor):
        super().__init__(tensor)
        self.storage = storage
        self.original_shape = storage.shape

    def __repr__(self) -> str:
        return f"HostTensorWrapper(tensor={self.base!r}, storage=<torch.Tensor>)"


class ScalarWrapper(pypto_wrap.BaseWrapper[T], abc.ABC):

    @abc.abstractmethod
    def __int__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __float__(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, dtype: dtypes.AnyDataType) -> Self:
        raise NotImplementedError


class StaticScalarWrapper(ScalarWrapper[Real]):

    def __init__(self, base: Real) -> None:
        super().__init__(base)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base!r})"

    def __int__(self) -> int:
        return int(self.base)

    def __float__(self) -> float:
        return float(self.base)

    def to(self, dtype: dtypes.AnyDataType) -> Self:
        return self


class TensorElementWrapper(ScalarWrapper[pypto.tensor]):

    def __init__(self, tensor: HostTensorWrapper, index: int, dtype: Optional[dtypes.AnyDataType] = None) -> None:
        if not isinstance(tensor, HostTensorWrapper) or not isinstance(index, int):
            raise TypeError("TensorElementWrapper requires (HostTensorWrapper, int, ...)")
        super().__init__(None)
        self.tensor = tensor
        self.index = index
        self.dtype = dtype or tensor.dtype
        self._item: Optional[Real] = None
        self._unwrap: Optional[pypto.tensor] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tensor!r}, {self.index})"

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    @staticmethod
    def _binary_scalar_op(left, right, op: Callable[[Real, Real], Any]) -> Any:
        lval = left.item() if isinstance(left, TensorElementWrapper) else left
        rval = right.item() if isinstance(right, TensorElementWrapper) else right
        logger.debug(f" ↳ {op.__name__}({lval}, {rval})")
        if isinstance(lval, AffineTensorLayout) or isinstance(rval, AffineTensorLayout):
            return op(lval, rval)
        if not isinstance(lval, Real) or not isinstance(rval, Real):
            raise RuntimeError(f"Non-scalar values: {lval!r}, {rval!r}")
        return op(lval, rval)

    def __add__(self, other):
        return self._binary_scalar_op(self, other, operator.add)

    def __radd__(self, other):
        return self._binary_scalar_op(other, self, operator.add)

    def __sub__(self, other):
        return self._binary_scalar_op(self, other, operator.sub)

    def __rsub__(self, other):
        return self._binary_scalar_op(other, self, operator.sub)

    def __mul__(self, other):
        return self._binary_scalar_op(self, other, operator.mul)

    def __rmul__(self, other):
        return self._binary_scalar_op(other, self, operator.mul)

    def __truediv__(self, other):
        return self._binary_scalar_op(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return self._binary_scalar_op(other, self, operator.truediv)

    def __floordiv__(self, other):
        return self._binary_scalar_op(self, other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._binary_scalar_op(other, self, operator.floordiv)

    def __mod__(self, other):
        return self._binary_scalar_op(self, other, operator.mod)

    def __rmod__(self, other):
        return self._binary_scalar_op(other, self, operator.mod)

    def __eq__(self, other):
        return self._binary_scalar_op(self, other, operator.eq)

    def __ne__(self, other):
        return self._binary_scalar_op(self, other, operator.ne)

    def __lt__(self, other):
        return self._binary_scalar_op(self, other, operator.lt)

    def __le__(self, other):
        return self._binary_scalar_op(self, other, operator.le)

    def __gt__(self, other):
        return self._binary_scalar_op(self, other, operator.gt)

    def __ge__(self, other):
        return self._binary_scalar_op(self, other, operator.ge)

    def to(self, dtype: dtypes.AnyDataType) -> Self:
        return self.__class__(self.tensor, self.index, dtype)

    def item(self) -> Real:
        if self._item is not None:
            return self._item
        storage = self.tensor.storage.flatten()
        self._item = storage[self.index].to(dtypes.to_info(self.dtype).as_torch).item()
        logger.debug("! Static scalar %s -> %s", self.index, self._item)
        return self._item

    def unwrap(self) -> pypto.tensor:
        if self._unwrap is not None:
            return self._unwrap
        multidim_offset = delinearize_offset(self.index, self.tensor.original_shape)
        view_shape = [1] * self.tensor.rank
        pypto_wrap.auto_vec_tile(view_shape, self.tensor.dtype)
        tensor = pypto_wrap.view(self.tensor, view_shape, multidim_offset)
        self._unwrap = reshape_impl(tensor, [1]).unwrap()
        return self._unwrap


class StaticDynamic(abc.ABC):

    @abc.abstractmethod
    def to_static(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_dynamic(self):
        raise NotImplementedError


class BaseTensorLayout:
    pass


class SingleTensorLayout(BaseTensorLayout):

    def __init__(self, base: Optional[TensorWrapper] = None, itype: Optional[dtypes.AnyDataType] = None) -> None:
        self.base = base
        if itype is not None:
            itype = dtypes.to_info(itype).as_numpy
        self.itype: Type[np.integer] = itype if itype is not None else np.int32

    @property
    def dtype(self) -> dtypes.DataTypeInfo:
        if self.base is None:
            return dtypes.to_info(self.itype)
        return dtypes.to_info(self.base.dtype)

    @property
    def type(self) -> dtypes.DataTypeInfo:
        if self.base is None:
            raise RuntimeError("Base tensor is None")
        return self.dtype

    def to(self, itype: dtypes.AnyDataType) -> Self:
        raise NotImplementedError

    def is_contiguous(self) -> bool:
        return False


class StaticTensorLayout(SingleTensorLayout):

    def __init__(self, base: Optional[TensorWrapper], indices: np.ndarray) -> None:
        super().__init__(base)
        self.indices = indices

    @property
    def offset(self) -> int:
        return int(self.indices.flat[0])

    def clone(self, indices: np.ndarray) -> Self:
        return StaticTensorLayout(self.base, indices)

    def is_contiguous(self) -> bool:
        if self.indices.size < 2:
            return True
        return np.all(np.diff(self.indices.ravel()) == 1)

    def __getitem__(self, slices):
        item = self.indices.__getitem__(slices)
        if isinstance(item, np.ndarray):
            return self.clone(self.indices.__getitem__(slices))
        return item

    def __add__(self, other):
        return self.clone(self.indices + other)

    def __radd__(self, other):
        return self.clone(other + self.indices)

    def __mul__(self, other):
        return self.clone(self.indices * other)

    def __rmul__(self, other):
        return self.clone(other * self.indices)

    def __floordiv__(self, other):
        return self.clone(self.indices // other)

    def __ge__(self, other) -> StaticMaskLayout:
        return StaticMaskLayout(self.indices >= other)

    def __gt__(self, other) -> StaticMaskLayout:
        return StaticMaskLayout(self.indices > other)

    def __le__(self, other) -> StaticMaskLayout:
        return StaticMaskLayout(self.indices <= other)

    def __lt__(self, other) -> StaticMaskLayout:
        return StaticMaskLayout(self.indices < other)


class AffineTensorLayout(SingleTensorLayout, StaticDynamic):

    def __init__(self, *, base: Optional[TensorWrapper] = None, offset: int = 0, sizes: IntArrayLike,
                 strides: IntArrayLike, order: Optional[IntArrayLike] = None,
                 itype: Optional[dtypes.AnyDataType] = None) -> None:
        super().__init__(base, itype)
        self.offset = offset
        self.sizes = self.as_array(sizes)
        self.strides = self.as_array(strides)
        self.order = self.as_array(order) if order is not None else np.arange(self.sizes.size, dtype=itype)

    @overload
    def clone(self, *, base: Optional[TensorWrapper] = None, offset: Optional[int] = None,
              sizes: Optional[IntArrayLike] = None, strides: Optional[IntArrayLike] = None,
              order: Optional[IntArrayLike] = None, itype: Optional[dtypes.AnyDataType] = None) -> Self:
        ...

    def clone(self, **kwds):
        attrs = ["base", "offset", "sizes", "strides", "order", "itype"]
        return self.__class__(**{k: kwds.get(k, getattr(self, k)) for k in attrs})

    @staticmethod
    def broadcast_sizes(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if np.any((lhs != 1) & (rhs != 1)):
            raise NonAffineLayoutError
        return np.where(lhs != 1, lhs, rhs)

    @staticmethod
    def broadcast_strides(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if np.any((lhs != 0) & (rhs != 0)):
            raise NonAffineLayoutError
        return np.where(lhs != 0, lhs, rhs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base={self.base!r}, offset={self.offset}, sizes={self.sizes}, " \
               f"strides={self.strides}, itype={self.itype.__name__})"

    def __str__(self) -> str:
        return repr(self)

    def __add__(self, other: Union[int, pypto.symbolic_scalar, Self]) -> Self:
        if isinstance(other, (int, pypto.symbolic_scalar)):
            return self.clone(offset=self.offset + other)
        if isinstance(other, self.__class__):
            return self.clone(
                base=self.select_base(other),
                offset=self.offset + other.offset,
                sizes=self.broadcast_sizes(self.sizes, other.sizes),
                strides=self.broadcast_strides(self.strides, other.strides),
            )
        return NotImplemented

    def __radd__(self, other) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Union[int, pypto.symbolic_scalar, Self]) -> Self:
        if isinstance(other, (int, pypto.symbolic_scalar)):
            return self.clone(offset=self.offset - other)
        if isinstance(other, self.__class__):
            return self.clone(
                base=self.select_base(other),
                offset=self.offset - other.offset,
                sizes=self.broadcast_sizes(self.sizes, other.sizes),
                strides=self.broadcast_strides(self.strides, other.strides),
            )
        return NotImplemented

    def __mul__(self, factor: int) -> Self:
        if not isinstance(factor, int):
            return NotImplemented
        return self.clone(offset=self.offset * factor, strides=self.strides * factor)

    def __rmul__(self, factor: int) -> Self:
        return self.__mul__(factor)

    def __truediv__(self, c: int) -> Self:
        if not isinstance(c, int):
            return NotImplemented
        if c <= 0:
            raise ZeroDivisionError("Undefined layout")
        if self.offset % c != 0:
            raise NonAffineLayoutError
        if np.any(self.strides % c != 0):
            return self.try_implicit_broadcast(self.to_static() // c)
        return self.clone(offset=self.offset // c, strides=self.strides // c)

    def __floordiv__(self, c: int) -> Self:
        return self.__truediv__(c)

    def __mod__(self, c: int) -> Self:
        if not isinstance(c, int):
            return NotImplemented
        if c <= 0:
            raise ZeroDivisionError("Undefined layout")
        if c == 1:
            return self.clone(offset=0, strides=np.zeros_like(self.strides))
        if np.count_nonzero(self.strides) == 0:
            return self.clone(offset=self.offset % c, strides=np.zeros_like(self.strides))
        extent = self.sizes - 1
        contrib = extent * self.strides
        contrib_min = np.minimum(0, contrib)
        contrib_max = np.maximum(0, contrib)
        min_val = self.offset + contrib_min.sum()
        max_val = self.offset + contrib_max.sum()
        if min_val >= 0 and max_val < c:
            return self
        raise NonAffineLayoutError

    def __gt__(self, other):
        return CompoundMask(operator.gt, self, other)

    def __ge__(self, other):
        return CompoundMask(operator.ge, self, other)

    def __lt__(self, other):
        return CompoundMask(operator.lt, self, other)

    def __le__(self, other):
        return CompoundMask(operator.le, self, other)

    def __getitem__(self, slices) -> Self:
        if not isinstance(slices, tuple):
            slices = (slices, )
        axes = tuple(i for i, k in enumerate(slices) if k is None)
        return self.clone(sizes=np.insert(self.sizes, axes, 1), strides=np.insert(self.strides, axes, 0))

    def inverse_order_permutation(self) -> Optional[List[int]]:
        if np.array_equal(self.order, np.arange(self.order.size)):
            return None
        inv_order = [0] * self.order.size
        for i, dim in enumerate(self.order.flat):
            inv_order[dim.item()] = i
        return inv_order

    def is_contiguous(self) -> bool:
        cont_strides = np.ones_like(self.sizes)
        cont_strides[:-1] = np.cumprod(self.sizes[:0:-1])[::-1]
        return np.array_equal(self.strides, cont_strides)

    def to_static(self) -> StaticTensorLayout:
        grids = np.meshgrid(*(np.arange(s, dtype=self.itype) for s in self.sizes), indexing="ij")
        indices = self.offset + builtins.sum(g * st for g, st in zip(grids, self.strides))
        return StaticTensorLayout(self.base, indices)

    def to_dynamic(self) -> TensorWrapper:
        shape = self.sizes.tolist()
        pypto_wrap.auto_vec_tile(shape, self.dtype)
        indices = pypto_wrap.arange(self.sizes.prod()).reshape(shape).add(self.offset)
        return TensorWrapper(indices)

    def select_base(self, other: Self) -> Optional[np.ndarray]:
        return other.base if self.base is None else self.base

    def try_implicit_broadcast(self, layout: StaticTensorLayout) -> BaseTensorLayout:
        """
        Computes the Run-Length Encoding (RLE) of a 1D NumPy array in O(N) time.

        Identifies consecutive sequences of identical values and returns the
        values and their corresponding lengths. This implementation preserves
        the order of appearance, meaning non-monotonic sequences like
        [4, 4, 5, 4, 4] are treated as three distinct groups.
        """
        offsets = layout.indices.ravel()
        switch_points = np.flatnonzero(offsets[1:] != offsets[:-1]) + 1
        boundaries = np.concatenate(([0], switch_points, [offsets.size]))
        values = offsets[boundaries[:-1]]
        if values.size > offsets.size // 2:
            raise NonAffineLayoutError
        counts = np.diff(boundaries)
        logger.debug("! Implicit broadcast %s %s", values, counts)
        layouts = []
        for offset, count in zip(values, counts):
            layout = AffineTensorLayout(base=self.base, offset=offset, sizes=count, strides=0)
            layouts.append(layout)
        if len(layouts) == 1:
            return layouts[0]
        return MultiTensorLayout(layouts)

    def as_array(self, val: IntArrayLike) -> np.ndarray:
        if isinstance(val, (int, np.generic)):
            return np.array([val], dtype=self.itype)
        return np.array(val, dtype=self.itype)

    def to(self, itype: dtypes.AnyDataType) -> Self:
        return self.clone(itype=itype)


class MultiTensorLayout(BaseTensorLayout):

    def __init__(self, layouts: Optional[Iterable[SingleTensorLayout]] = None):
        self.layouts = list(layouts) if layouts is not None else []

    def __repr__(self) -> str:
        return f"MultiTensorLayout({self.layouts!r})"

    def __str__(self) -> str:
        return repr(self)

    def append(self, layout: SingleTensorLayout) -> None:
        self.layouts.append(layout)

    def apply(self, op: Callable[[SingleTensorLayout], SingleTensorLayout]) -> MultiTensorLayout:
        layouts = [op(layout) for layout in self.layouts]
        return MultiTensorLayout(layouts)

    def __add__(self, other) -> MultiTensorLayout:
        return self.apply(lambda layout: layout + other)

    def __radd__(self, other) -> MultiTensorLayout:
        return self.__add__(other)

    def __mul__(self, other) -> MultiTensorLayout:
        return self.apply(lambda layout: layout * other)

    def __rmul__(self, other) -> MultiTensorLayout:
        return self.__mul__(other)

    def __iter__(self):
        return iter(self.layouts)

    def __len__(self) -> int:
        return len(self.layouts)


class BaseMaskLayout:
    pass


class StaticMaskLayout(BaseMaskLayout):

    def __init__(self, bits: Optional[np.ndarray] = None):
        self.bits = bits

    def __repr__(self) -> str:
        if self.bits is None:
            return "StaticMaskLayout()"
        return f"StaticMaskLayout(bits=<numpy.ndarray>)"

    def __str__(self) -> str:
        return repr(self)

    def __and__(self, other) -> Self:
        if self.bits is None:
            return StaticMaskLayout()
        rhs = other.bits if isinstance(other, StaticMaskLayout) else other
        return StaticMaskLayout(self.bits & rhs)

    def __or__(self, other) -> Self:
        if self.bits is None:
            return StaticMaskLayout()
        rhs = other.bits if isinstance(other, StaticMaskLayout) else other
        return StaticMaskLayout(self.bits | rhs)

    def __getitem__(self, slices) -> Self:
        if self.bits is None:
            return StaticMaskLayout()
        if not isinstance(slices, tuple):
            slices = (slices, )
        axes = tuple(i for i, k in enumerate(slices) if k is None)
        return StaticMaskLayout(np.expand_dims(self.bits, axes))

    def broadcast_to(self, shape: Iterable[int]) -> Self:
        return StaticMaskLayout(np.broadcast_to(self.bits, shape))

    def valid_shape(self) -> Optional[Tuple[int, ...]]:
        if self.bits is None:
            return None
        assert self.bits.dtype == np.bool_
        ndim = self.bits.ndim
        block_shape = []
        # Step 1: find block extent along each axis
        for axis in builtins.range(ndim):
            other_axes = tuple(i for i in builtins.range(ndim) if i != axis)
            proj = self.bits.any(axis=other_axes)
            # find first False after True prefix
            false_idx = np.flatnonzero(~proj)
            if false_idx.size == 0:
                k = proj.size
            else:
                k = false_idx[0]
                if proj[k:].any():
                    return None  # True appears after a False → invalid
            block_shape.append(int(k))
        block_shape = tuple(block_shape)
        # Step 2: verify exact block
        slices = tuple(slice(0, k) for k in block_shape)
        if not self.bits[slices].all():
            return None
        tmp = self.bits.copy()
        tmp[slices] = False
        if tmp.any():
            return None
        return block_shape


class ArangeConcrete(Arange):

    def to_affine(self):
        return AffineTensorLayout(sizes=self.end - self.start, strides=1)

    def to_static(self):
        return StaticTensorLayout(None, np.arange(self.start, self.end, dtype=np.int32))

    def to_dynamic(self):
        # PyPTO crashes when compare INT32 tensors, need to force FP32 here
        pypto_wrap.auto_vec_tile([self.end - self.start], pypto.DT_FP32)
        return TensorWrapper(pypto_wrap.arange(float(self.start), float(self.end)))


def program_id(axis: int) -> int:
    assert 0 <= axis <= 2
    return Context.program_id[axis]


def num_programs(axis: int) -> int:
    assert 0 <= axis <= 2
    return Context.num_programs[axis]


def arange(start: int, end: int) -> AffineTensorLayout:
    return ArangeConcrete(start, end)


def delinearize_offset(linear_offset: int, shape: Tuple[int, ...]) -> List[int]:
    offsets = []
    for dim in reversed(shape):
        offsets.append(linear_offset % dim)
        linear_offset //= dim
    return list(reversed(offsets))


def compute_valid_shape(mask: Optional[BaseMaskLayout], target_shape: Iterable[int]) -> Optional[Tuple[int]]:
    if Context.dynamic or mask is None:
        return None
    if isinstance(mask, CompoundMask):
        mask = mask.to_static()
    if not isinstance(mask, StaticMaskLayout):
        raise RuntimeError(f"valid shape cannot be computed from mask: {mask!r}")
    valid_shape = mask.broadcast_to(target_shape).valid_shape()
    if all(v == t for v, t in zip(valid_shape, target_shape)):
        return None
    return valid_shape


def tensor_to_affine(two: TensorWithOffset) -> BaseTensorLayout:
    offset = two.offset
    if isinstance(offset, CompoundSentinel):
        layout = offset.to_affine()
    else:
        layout = AffineTensorLayout(offset=offset, sizes=1, strides=1)
    layout.base = two.base
    return layout


@log_call
def load(pointer: Any, mask: Optional[BaseMaskLayout] = None, other: Optional[Any] = None,
         **kwds) -> Union[TensorWrapper, TensorElementWrapper]:
    layout = pointer
    if other is not None and (not isinstance(other, Real) or other != 0):
        raise RuntimeError(f"only scalar zero is supported as 'other', got {other!r}")
    if isinstance(layout, MultiTensorLayout):
        results = [load(l, mask, other) for l in layout]
        if len(results) == 1:
            return results[0]
        return TensorWrapper(pypto_wrap.concat([result.tensor for result in results]))
    if isinstance(layout, TensorWithOffset):
        layout = tensor_to_affine(pointer)
    if not isinstance(layout, AffineTensorLayout):
        raise TypeError(f"{layout.__class__.__name__} is not supported in single tensor load")
    if not isinstance(layout.base, HostTensorWrapper):
        raise NotImplementedError("Only HostTensorWrapper supported")
    static_scalar = layout.sizes.prod() == 1 and isinstance(layout.offset, int)
    if static_scalar:
        return TensorElementWrapper(layout.base, layout.offset)
    target_shape = layout.sizes.tolist()
    padded_shape = pad_shape(target_shape, layout.base.rank)
    multidim_offset = delinearize_offset(layout.offset, layout.base.original_shape)
    requires_expand = layout.sizes.prod() != 1 and np.count_nonzero(layout.strides) == 0
    inv_order = layout.inverse_order_permutation()
    if inv_order is not None:
        order = layout.order.tolist()
        target_shape = [target_shape[i] for i in order]
        multidim_offset = [multidim_offset[i] for i in order]
    if requires_expand:
        src = pypto_wrap.view(layout.base, [1] * len(padded_shape), multidim_offset)
        pypto_wrap.auto_vec_tile(padded_shape, src.dtype)
        result = expand_impl(src, padded_shape)
    else:
        valid_shape = compute_valid_shape(mask, padded_shape)
        result = pypto_wrap.view(layout.base, padded_shape, multidim_offset, valid_shape=valid_shape)
    result = reshape_impl(result, target_shape)
    if inv_order is not None:
        result = permute(result, inv_order)
    return result


@log_call
def store(pointer: Any, value: TensorWrapper, mask: Optional[BaseMaskLayout] = None, **kwds) -> None:
    if isinstance(pointer, TensorWithOffset):
        layout = tensor_to_affine(pointer)
    else:
        layout = pointer
    if not isinstance(layout, AffineTensorLayout):
        raise TypeError(f"store requires {AffineTensorLayout.__name__}, got {layout.__class__.__name__}")
    if not isinstance(layout.base, HostTensorWrapper):
        raise NotImplementedError("Only HostTensorWrapper supported")
    multidim_offset = delinearize_offset(layout.offset, layout.base.original_shape)
    src_shape = value.shape
    target_shape = pad_shape(src_shape, len(layout.base.shape))
    value = reshape_impl(value, target_shape)
    if valid_shape := compute_valid_shape(mask, target_shape):
        value = pypto_wrap.view(value, src_shape, [0] * len(src_shape), valid_shape=valid_shape)
    if layout.inverse_order_permutation() is not None:
        order = layout.order
        value = permute(value, order)
        target_shape = [target_shape[i] for i in order]
        multidim_offset = [multidim_offset[i] for i in order]
    pypto_wrap.assemble(value, multidim_offset, layout.base)


@log_call
def make_block_ptr(base: Any, shape: Tuple[int, ...], strides: Tuple[int, ...], offsets: Tuple[int, ...],
                   block_shape: Tuple[int, ...], order: Tuple[int, ...]) -> AffineTensorLayout:
    if isinstance(base, TensorWithOffset):
        layout = tensor_to_affine(base)
    elif isinstance(base, TensorPointer):
        layout = AffineTensorLayout(base=base.base, offset=0, sizes=1, strides=0)
    else:
        layout = base
    if not isinstance(layout, AffineTensorLayout):
        raise RuntimeError(f"base must be AffineTensorLayout, got {layout.__class__.__name__}")
    rank = len(shape)
    if not all(len(t) == rank for t in (strides, offsets, block_shape, order)):
        raise RuntimeError(f"rank {rank} is not consistent among descriptors")
    offset = builtins.sum(s * o for s, o in zip(shape, offsets))
    order = None  # TODO: respect the order
    return AffineTensorLayout(base=layout.base, offset=offset, sizes=block_shape, strides=strides, order=order)


# TODO: implement tensor method
def cdiv(x: int, div: int) -> int:
    if div == 0:
        raise ZeroDivisionError
    q = int(x / div)  # truncate toward zero
    return q + int(q * div != x)


def scalarize(arg: Union[pypto.SymbolicScalar, int, TensorWrapper, ScalarWrapper]) -> pypto.SymbolicScalar:
    if isinstance(arg, (pypto.SymbolicScalar, int)):
        return arg
    if isinstance(arg, TensorWrapper):
        if not all([dim == 1 for dim in arg.shape]):
            raise ValueError(f"Incorrect tensor shape: {arg.shape}")
        idxs = [0] * len(arg.shape)
        return arg.base[tuple(idxs)]
    if isinstance(arg, ScalarWrapper):
        return int(arg)
    raise ValueError(f"Unsupported type to scalarize: {type(arg)}")


@log_call
def range(*args: int, **kwds):
    if all(isinstance(arg, int) for arg in args):
        return builtins.range(*args)
    mapped_args = [scalarize(arg) for arg in args]
    return pypto_wrap.loop(*mapped_args, name="tl_range_loop", idx_name="tl_range_loop_indvar")


@log_call
def static_range(*args: int, **kwds):
    if all(isinstance(arg, int) for arg in args):
        return builtins.range(*args)
    raise RuntimeError(f"dynamic range is not allowed in static_range: {args!r}")


@log_call
def zeros(shape: Tuple[int, ...], dtype: dtypes.AnyDataType) -> TensorWrapper:
    pypto_wrap.auto_vec_tile(shape, dtype)
    tensor = pypto_wrap.full(shape, 0.0, dtypes.to_pypto(dtype))
    return TensorWrapper(tensor)


@log_call
def dot(input: TensorWrapper, other: TensorWrapper, acc: Optional[TensorWrapper] = None,
        out_dtype: dtypes.AnyDataType = pypto.DataType.DT_FP32) -> TensorWrapper:
    if input.dtype != other.dtype:
        raise RuntimeError("input matrices must have the same dtype")
    a_shape = input.shape
    b_shape = other.shape
    matmul_compatible = len(a_shape) == 2 and len(b_shape) == 2 and a_shape[1] == b_shape[0]
    if not matmul_compatible:
        raise RuntimeError(f"Tensor shapes are not compatible for matmul: {a_shape} vs. {b_shape}")
    m = a_shape[0]
    k = a_shape[1]
    n = b_shape[1]
    pypto_wrap.auto_cube_tile([m, m], [k, k], [n, n], input.dtype)
    result = pypto_wrap.matmul(input, other, dtypes.to_pypto(out_dtype))
    if acc is not None:
        acc.auto_vec_tile()
        result = pypto_wrap.add(acc, result)
    return TensorWrapper(result)


@bind_tensor_method
@log_call
def abs(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.abs(x))


@log_call
def clamp(x: TensorWrapper, min: Union[TensorWrapper, Real], max: Union[TensorWrapper, Real]) -> TensorWrapper:
    if not isinstance(min, TensorWrapper):
        min = x.full_like(min)
    if not isinstance(max, TensorWrapper):
        max = x.full_like(max)
    x.auto_vec_tile()
    # PTO issue: assertion in pypto.clip fails
    maxmin = pypto_wrap.maximum(x, min)
    minmax = pypto_wrap.minimum(TensorWrapper(maxmin), max)
    return TensorWrapper(minmax)


@bind_tensor_method
@log_call
def exp(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.exp(x))


@bind_tensor_method
@log_call
def max(input: TensorWrapper, axis: Optional[int] = None, keep_dims: bool = False) -> TensorWrapper:
    axis = axis if axis is not None else -1
    keep_dims = keep_dims or input.rank == 1
    input.auto_vec_tile()
    return TensorWrapper(pypto_wrap.amax(input, axis, keep_dims))


@log_call
def maximum(x: TensorWrapper, y: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.maximum(x, y))


@log_call
def minimum(x: TensorWrapper, y: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.minimum(x, y))


@bind_tensor_method
@log_call
def sum(input: TensorWrapper, axis: Optional[int] = None, keep_dims: bool = False) -> TensorWrapper:
    axis = axis if axis is not None else -1
    keep_dims = keep_dims or input.rank == 1
    input.auto_vec_tile(buf_num=4)
    return TensorWrapper(pypto_wrap.sum(input, axis, keep_dims))


@bind_tensor_method
@log_call
def permute(input: TensorWrapper, *dims: Union[int, Iterable[int]]) -> TensorWrapper:
    dims = first_or_all(*dims, name="dims")
    rank = len(input.shape)
    if len(dims) != rank:
        raise RuntimeError(f"dims permutation must have {rank} items: {dims}")
    canon_dims = tuple(builtins.range(rank))
    pto_dims = []
    for dim, canon_dim in zip(dims, canon_dims):
        if dim not in canon_dims:
            raise RuntimeError(f"there is no dimension {dim} in {canon_dims}")
        if dim != canon_dim:
            pto_dims.append(dim)
    if len(pto_dims) == 0:
        return input
    if len(pto_dims) != 2:
        raise RuntimeError(f"more than two dimensions are different: {dims} vs. {canon_dims}")
    pto_dims.sort()
    input.auto_vec_tile(buf_num=4)
    input = pypto_wrap.transpose(input, *pto_dims)
    return TensorWrapper(input)


@bind_tensor_method
@log_call
def trans(input: TensorWrapper, *dims: Union[int, Iterable[int]]) -> TensorWrapper:
    if not dims:
        rank = len(input.shape)
        dims = (*builtins.range(rank - 2), rank - 1, rank - 2)
    return permute(input, *dims)


@log_call
def where(condition: Union[CompoundMask, TensorWrapper], x: TensorWrapper, y: TensorWrapper) -> TensorWrapper:
    if isinstance(condition, CompoundMask):
        condition = condition.to_dynamic()
    condition.auto_vec_tile()
    return TensorWrapper(pypto_wrap.where(condition, x, y))


@bind_tensor_method
@log_call
def log(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.log(x))


@bind_tensor_method
@log_call
def sigmoid(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.sigmoid(x))


@bind_tensor_method
@log_call
def broadcast_to(input: TensorWrapper, *shape: Union[int, List[int]]) -> TensorWrapper:
    shape = first_or_all(*shape, name="shape")
    pypto_wrap.auto_vec_tile(shape, input.dtype)
    return expand_impl(input, shape)


@bind_tensor_method
@log_call
def cast(input: TensorWrapper, dtype: dtypes.AnyDataType) -> TensorWrapper:
    if input.dtype == dtypes.to_pypto(dtype):
        return input
    input.auto_vec_tile()
    tensor = pypto_wrap.cast(input, dtypes.to_pypto(dtype))
    return TensorWrapper(tensor)


@bind_tensor_method
@log_call
def reshape(input: TensorWrapper, *shape: Union[int, Iterable[int]]) -> TensorWrapper:
    shape = first_or_all(*shape, name="shape")
    return reshape_impl(input, shape)


@bind_tensor_method
def ravel(x: TensorWrapper) -> TensorWrapper:
    return reshape(x, x.size)


@bind_tensor_method
@log_call
def sqrt(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.sqrt(x))


@bind_tensor_method
@log_call
def rsqrt(x: TensorWrapper) -> TensorWrapper:
    x.auto_vec_tile()
    return TensorWrapper(pypto_wrap.rsqrt(x))


@bind_tensor_method
@log_call
def expand_dims(input: TensorWrapper, axis: Union[int, Iterable[int]]) -> TensorWrapper:
    axis = first_or_all(axis)
    new_shape = np.insert(input.shape, axis, 1).tolist()
    if len(axis) == 1 and input.dtype in (pypto.DT_FP32, pypto.DT_FP16, pypto.DT_BF16):
        pypto_wrap.auto_vec_tile(new_shape, input.dtype)
        return TensorWrapper(pypto_wrap.unsqueeze(input, axis[0]))
    return reshape_impl(input, new_shape)
