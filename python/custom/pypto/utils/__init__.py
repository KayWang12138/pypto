# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from .aggrvec import AggregationVec
from .codehelper import CodeHelper
from .configmap import ConfigMap
from .custstruct import CustStruct
from .enums import DATATYPE, ReduceMode, CastMode, ScatterMode
from .instruction import Instruction
from .shape import Shape
from .std_tuple import Tuple
from .std_vec import Vector
from .tensor import Tensor
from .tensormap import TensorMap
from .utilfuncs import get_obj_dtype, get_vartype_str
from .var import Var
