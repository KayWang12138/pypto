from enum import Enum
import functools
from typing import Any, Dict, Optional, Type, TypeVar, Union, overload
from typing_extensions import Literal, TypeAlias

import numpy
import pypto
import torch
import triton.language as tl

T = TypeVar("T")


class DataTypeKind(Enum):
    Any = ""
    Int = "int"
    Float = "float"


class DataTypeInfo(tl.dtype):

    def __init__(self, name: str, kind: DataTypeKind, bitwidth: int, as_numpy: Optional[Type[numpy.generic]],
                 as_pypto: Optional[pypto.DataType], as_torch: Optional[torch.dtype], as_triton: Optional[tl.dtype]):
        self.name = name
        self.kind = kind
        self.bitwidth = bitwidth
        self.as_numpy = as_numpy
        self.as_pypto = as_pypto
        self.as_torch = as_torch
        self.as_triton = as_triton
        self.element_ty = self.as_triton  # Comply with triton.language.dtype

    def __hash__(self) -> int:
        return hash(f"DataTypeInfo-{self.name}")

    def __eq__(self, other: Any) -> bool:
        return self.name == to_str(other, required=False)

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def is_float(self) -> bool:
        return self.kind == DataTypeKind.Float

    def is_int(self) -> bool:
        return self.kind == DataTypeKind.Int


AnyDataType: TypeAlias = Union[
    DataTypeInfo,
    str,
    Type[numpy.generic],
    numpy.dtype,
    pypto.DataType,
    torch.dtype,
    tl.dtype,
]

supported_types = (
    DataTypeInfo("bool", DataTypeKind.Int, 8, numpy.bool_, pypto.DataType.DT_BOOL, torch.bool, None),
    DataTypeInfo("int8", DataTypeKind.Int, 8, numpy.int8, pypto.DataType.DT_INT8, torch.int8, tl.int8),
    DataTypeInfo("int16", DataTypeKind.Int, 16, numpy.int16, pypto.DataType.DT_INT16, torch.int16, tl.int16),
    DataTypeInfo("int32", DataTypeKind.Int, 32, numpy.int32, pypto.DataType.DT_INT32, torch.int32, tl.int32),
    DataTypeInfo("int64", DataTypeKind.Int, 64, numpy.int64, pypto.DataType.DT_INT64, torch.int64, tl.int64),
    DataTypeInfo("float8", DataTypeKind.Float, 8, None, pypto.DataType.DT_FP8, None, None),
    DataTypeInfo("float16", DataTypeKind.Float, 16, numpy.float16, pypto.DataType.DT_FP16, torch.float16, tl.float16),
    DataTypeInfo("float32", DataTypeKind.Float, 32, numpy.float32, pypto.DataType.DT_FP32, torch.float32, tl.float32),
    DataTypeInfo("float64", DataTypeKind.Float, 64, numpy.float64, pypto.DataType.DT_DOUBLE, torch.float64, tl.float64),
    DataTypeInfo("bfloat16", DataTypeKind.Float, 16, None, pypto.DataType.DT_BF16, torch.bfloat16, None),
)

name_to_info: Dict[str, DataTypeInfo] = {info.name: info for info in supported_types}
pypto_to_name: Dict[pypto.DataType, str] = {info.as_pypto: info.name for info in supported_types}


def cache(fn: T) -> T:
    return functools.lru_cache(typed=True)(fn)


@overload
def to_str(any_dtype: AnyDataType, required: Literal[True] = True) -> str:
    ...


@overload
def to_str(any_dtype: AnyDataType, required: bool = True) -> Optional[str]:
    ...


@cache
def to_str(any_dtype: AnyDataType, required: bool = True):
    if isinstance(any_dtype, str):
        return any_dtype
    if isinstance(any_dtype, DataTypeInfo):
        return any_dtype.name
    if isinstance(any_dtype, pypto.DataType):
        str_dtype = pypto_to_name.get(any_dtype, None)
        if required and str_dtype is None:
            raise RuntimeError(f"pypto.DataType is not supported: {any_dtype!r}")
        return str_dtype
    if isinstance(any_dtype, numpy.dtype):
        return str(any_dtype)
    if isinstance(any_dtype, type) and issubclass(any_dtype, numpy.generic):
        return to_str(numpy.dtype(any_dtype))
    if isinstance(any_dtype, torch.dtype):
        return str(any_dtype)[len("torch."):]
    if isinstance(any_dtype, tl.dtype):
        name = any_dtype.name
        remap = {
            "fp16": "float16",
            "fp32": "float32",
        }
        return remap.get(name, name)
    if required:
        raise RuntimeError(f"Unsupported dtype: no str for {any_dtype!r}")
    return None


@overload
def to_info(any_dtype: AnyDataType, required: Literal[True] = True) -> DataTypeInfo:
    ...


@overload
def to_info(any_dtype: AnyDataType, required: bool = True) -> Optional[DataTypeInfo]:
    ...


@cache
def to_info(any_dtype: AnyDataType, required: bool = True):
    if isinstance(any_dtype, DataTypeInfo):
        return any_dtype
    str_dtype = to_str(any_dtype, required)
    info = name_to_info.get(str_dtype, None)
    if required and info is None:
        raise RuntimeError(f"Unsupported dtype: no DataTypeInfo for {any_dtype!r}")
    return info


@overload
def to_pypto(any_dtype: AnyDataType, required: Literal[True] = True) -> pypto.DataType:
    ...


@overload
def to_pypto(any_dtype: AnyDataType, required: bool = True) -> Optional[pypto.DataType]:
    ...


@cache
def to_pypto(any_dtype: AnyDataType, required: bool = True):
    if isinstance(any_dtype, pypto.DataType):
        return any_dtype
    info = to_info(any_dtype, required)
    if required and info.as_pypto is None:
        raise RuntimeError(f"Unsupported dtype: no pypto.DataType for {any_dtype!r}")
    return info.as_pypto


@cache
def common_type(*dtypes: AnyDataType) -> DataTypeInfo:
    if not dtypes:
        raise ValueError("At least one dtype must be provided")
    infos = {to_info(dtype) for dtype in dtypes}
    if len(infos) == 1:
        return infos.pop()
    kinds = {info.kind for info in infos}

    if DataTypeKind.Any in kinds:
        raise RuntimeError("bool type is not supported in common_type")
    if len(kinds) != 1:
        raise RuntimeError("Cannot compute common type between integer and floating-point types")

    common_info = max(infos, key=lambda i: i.bitwidth)
    if common_info not in infos:
        raise RuntimeError("Unexpected dtype encountered")

    return common_info
