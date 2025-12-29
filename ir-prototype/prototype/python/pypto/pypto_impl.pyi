import enum
from typing import List, overload

class DataType(enum.Enum):
    DT_INT4: "DataType"
    DT_INT8: "DataType"
    DT_INT16: "DataType"
    DT_INT32: "DataType"
    DT_INT64: "DataType"
    DT_FP8: "DataType"
    DT_FP16: "DataType"
    DT_FP32: "DataType"
    DT_BF16: "DataType"
    DT_HF4: "DataType"
    DT_HF8: "DataType"
    DT_UINT8: "DataType"
    DT_UINT16: "DataType"
    DT_UINT32: "DataType"
    DT_UINT64: "DataType"
    DT_BOOL: "DataType"
    DT_DOUBLE: "DataType"
    DT_BOTTOM: "DataType"


class ScalarValueKind(enum.Enum):
    Constant: "ScalarValueKind"
    Symbolic: "ScalarValueKind"


class TileOpFormat(enum.Enum):
    TILEOP_ND: "TileOpFormat"
    TILEOP_NZ: "TileOpFormat"


class ReLuType(enum.Enum):
    NoReLu: "ReLuType"
    ReLu: "ReLuType"


class Scalar:
    def __init__(
        self,
        value: int | bool | float | DataType | str | None = None,
        name: str = "",
        valueKind: ScalarValueKind = ScalarValueKind.Symbolic,
    ) -> None: ...

    def get_value_kind(self) -> ScalarValueKind: ...
    def has_constant_value(self) -> bool: ...
    def to_int(self) -> int: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...


class Tensor:
    @overload
    def __init__(self, dtype: DataType, shape: List[int],
                 name: str = "", format: TileOpFormat = TileOpFormat.TILEOP_ND): ...

    @overload
    def __init__(self, dtype: DataType, shape: List[Scalar],
                 name: str = "", format: TileOpFormat = TileOpFormat.TILEOP_ND): ...

    def GetDataType(self) -> DataType: ...
    def GetShape(self) -> List[int]: ...
    def Dim(self) -> int: ...
    def Id(self) -> int: ...
    def Format(self) -> TileOpFormat: ...
    def GetName(self) -> str: ...
    def SetName(self, name: str) -> None: ...
    def SetCachePolicy(self, policy: int, value: bool) -> None: ...
    def GetCachePolicy(self, policy: int) -> bool: ...
    def Move(self, other: "Tensor") -> None: ...


def SetTensorData(value: Scalar, indices: List[int], tensor: Tensor) -> None: ...


def GetTensorData(tensor: Tensor, indices: List[int]) -> Scalar: ...


def GetInputShape(tensor: Tensor, dim: int) -> Scalar: ...


class MatmulExtendParam:
    bias_tensor: Tensor
    scale_tensor: Tensor
    relu_type: ReLuType
    scale: float
    
    def __init__(self) -> None: ...
