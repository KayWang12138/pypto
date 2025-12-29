from . import pypto_impl

def _enum_repr(self):
    return f"{self.__class__.__name__}.{self.name}"  # remove pypto_impl. prefix

DataType = pypto_impl.DataType
DataType.__repr__ = _enum_repr

CastMode = pypto_impl.CastMode
CastMode.__repr__ = _enum_repr

ReLuType = pypto_impl.ReLuType
ReLuType.__repr__ = _enum_repr

DT_INT4 = pypto_impl.DataType.DT_INT4
DT_INT8 = pypto_impl.DataType.DT_INT8
DT_INT16 = pypto_impl.DataType.DT_INT16
DT_INT32 = pypto_impl.DataType.DT_INT32
DT_INT64 = pypto_impl.DataType.DT_INT64
DT_FP8 = pypto_impl.DataType.DT_FP8
DT_FP16 = pypto_impl.DataType.DT_FP16
DT_FP32 = pypto_impl.DataType.DT_FP32
DT_BF16 = pypto_impl.DataType.DT_BF16
DT_HF4 = pypto_impl.DataType.DT_HF4
DT_HF8 = pypto_impl.DataType.DT_HF8
DT_UINT8 = pypto_impl.DataType.DT_UINT8
DT_UINT16 = pypto_impl.DataType.DT_UINT16
DT_UINT32 = pypto_impl.DataType.DT_UINT32
DT_UINT64 = pypto_impl.DataType.DT_UINT64
DT_BOOL = pypto_impl.DataType.DT_BOOL
DT_DOUBLE = pypto_impl.DataType.DT_DOUBLE
DT_BOTTOM = pypto_impl.DataType.DT_BOTTOM