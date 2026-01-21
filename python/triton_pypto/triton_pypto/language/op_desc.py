from __future__ import annotations

import operator
from typing import Callable, Optional
from typing_extensions import Self


class OpDescriptor:

    def __init__(self, name: str, static_op: Callable, dynamic_op: Optional[Callable] = None) -> None:
        self.name = name
        self.static = static_op
        self.dynamic = dynamic_op if dynamic_op is not None else static_op

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_operator(cls, op: Callable) -> Self:
        return cls(op.__name__, op)


gt = OpDescriptor("gt", operator.gt)
ge = OpDescriptor("ge", operator.ge)
lt = OpDescriptor("lt", operator.lt)
le = OpDescriptor("le", operator.le)
and_ = OpDescriptor("and", operator.and_)
or_ = OpDescriptor("or", operator.or_)
getitem = OpDescriptor("getitem", operator.getitem)
