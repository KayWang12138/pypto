import abc
import operator
from typing import List

import pypto


class CompoundNode(abc.ABC):

    @abc.abstractmethod
    def to_affine(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_static(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_dynamic(self):
        raise NotImplementedError


class CompoundCombiner(CompoundNode):

    def __init__(self, op, *args) -> None:
        self.op = op
        self.args = args

    def to_impl(self, method):
        args = [method(arg) if isinstance(arg, CompoundNode) else arg for arg in self.args]
        return self.op(*args)

    def to_affine(self):
        return self.to_impl(lambda arg: arg.to_affine())

    def to_static(self):
        return self.to_impl(lambda arg: arg.to_static())

    def to_dynamic(self):
        return self.to_impl(lambda arg: arg.to_dynamic())


class CompoundOffset(CompoundCombiner):

    def __repr__(self) -> str:
        return f"CompoundOffset@{self.op.__name__}{self.args}"

    @classmethod
    def apply_arith_op(cls, op, lhs, rhs):
        allowed_types = (CompoundOffset, int, pypto.symbolic_scalar)
        if not isinstance(lhs, allowed_types) or not isinstance(rhs, allowed_types):
            return NotImplemented
        return CompoundOffset(op, lhs, rhs)

    @classmethod
    def apply_compare_op(cls, op, lhs, rhs):
        allowed_types = (CompoundOffset, int, pypto.symbolic_scalar)
        if not isinstance(lhs, allowed_types) or not isinstance(rhs, allowed_types):
            return NotImplemented
        return CompoundMask(op, lhs, rhs)

    def __add__(self, other):
        return self.apply_arith_op(operator.add, self, other)

    def __radd__(self, other):
        return self.apply_arith_op(operator.add, other, self)

    def __sub__(self, other):
        return self.apply_arith_op(operator.sub, self, other)

    def __rsub__(self, other):
        return self.apply_arith_op(operator.sub, other, self)

    def __mul__(self, other):
        return self.apply_arith_op(operator.mul, self, other)

    def __rmul__(self, other):
        return self.apply_arith_op(operator.mul, other, self)

    def __floordiv__(self, other):
        return self.apply_arith_op(operator.floordiv, self, other)

    def __mod__(self, other):
        return self.apply_arith_op(operator.mod, self, other)

    def __gt__(self, other):
        return self.apply_compare_op(operator.gt, self, other)

    def __ge__(self, other):
        return self.apply_compare_op(operator.ge, self, other)

    def __lt__(self, other):
        return self.apply_compare_op(operator.lt, self, other)

    def __le__(self, other):
        return self.apply_compare_op(operator.le, self, other)

    def __getitem__(self, slices):
        return CompoundOffset(operator.getitem, self, slices)


class CompoundMask(CompoundCombiner):

    def __repr__(self) -> str:
        return f"CompoundMask@{self.op.__name__}{self.args}"

    @classmethod
    def apply_logical_op(cls, op, lhs, rhs):
        allowed_types = (CompoundMask, bool)
        if not isinstance(lhs, allowed_types) or not isinstance(rhs, allowed_types):
            return NotImplemented
        return CompoundMask(op, lhs, rhs)

    def __and__(self, other):
        return self.apply_logical_op(operator.and_, self, other)

    def __rand__(self, other):
        return self.apply_logical_op(operator.and_, other, self)

    def __or__(self, other):
        return self.apply_logical_op(operator.or_, self, other)

    def __ror__(self, other):
        return self.apply_logical_op(operator.or_, other, self)

    def __xor__(self, other):
        return self.apply_logical_op(operator.xor, self, other)

    def __rxor__(self, other):
        return self.apply_logical_op(operator.xor, other, self)

    def __invert__(self):
        return CompoundMask(operator.invert, self)

    def __getitem__(self, slices):
        return CompoundMask(operator.getitem, self, slices)


class BaseArange(CompoundNode):

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start={self.start}, end={self.end})"

    @property
    def shape(self) -> List[int]:
        return [self.end - self.start]

    def __add__(self, other):
        return CompoundOffset(operator.add, self, other)

    def __radd__(self, other):
        return CompoundOffset(operator.add, other, self)

    def __sub__(self, other):
        return CompoundOffset(operator.sub, self, other)

    def __rsub__(self, other):
        return CompoundOffset(operator.sub, other, self)

    def __mul__(self, other):
        return CompoundOffset(operator.mul, self, other)

    def __rmul__(self, other):
        return CompoundOffset(operator.mul, other, self)

    def __floordiv__(self, other):
        return CompoundOffset(operator.floordiv, self, other)

    def __mod__(self, other):
        return CompoundOffset(operator.mod, self, other)

    def __gt__(self, other):
        return CompoundMask(operator.gt, self, other)

    def __ge__(self, other):
        return CompoundMask(operator.ge, self, other)

    def __lt__(self, other):
        return CompoundMask(operator.lt, self, other)

    def __le__(self, other):
        return CompoundMask(operator.le, self, other)

    def __getitem__(self, slices):
        return CompoundOffset(operator.getitem, self, slices)


class TensorPointer:

    def __init__(self, base) -> None:
        self.base = base

    def __repr__(self) -> str:
        return f"TensorPointer({self.base})"

    def __add__(self, other):
        return TensorWithOffset(self.base, other)

    def __radd__(self, other):
        return TensorWithOffset(self.base, other)

    @property
    def dtype(self):
        return self.base.dtype


class TensorWithOffset:

    def __init__(self, base, offset) -> None:
        self.base = base
        self.offset = offset

    def __repr__(self) -> str:
        return f"TensorWithOffset(base={self.base}, offset={self.offset})"

    def clone(self, offset):
        return TensorWithOffset(self.base, offset)

    def __add__(self, other):
        return self.clone(self.offset + other)

    def __radd__(self, other):
        return self.clone(other + self.offset)
