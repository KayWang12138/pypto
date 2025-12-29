from typing import Union
from . import pypto_impl

import sympy

class Scalar:

    def __init__(self, arg0: Union[int, str, 'Scalar'],
                 arg1: Union[int, str, None] = None,
                 arg2 = None):
        """
        Construct a SymbolicScalar.

        Args:
            arg0 Union[int, str, Scalar]: The value, type name, or another Scalar
            arg1 Union[int, str, None]: If arg0 is str, this can be the name (str) or None.
                                       If arg0 is int, this is ignored.
            arg2: Optional third argument (e.g., ScalarValueKind). Defaults to None.

        Examples:
            >>> b = Scalar(10)
            >>> c = Scalar("int32", "x")  # typeName, name
            >>> d = Scalar("int32")  # typeName only
            >>> e = Scalar("int32", "x", ScalarValueKind.Symbolic)  # typeName, name, valueKind
        """
        if isinstance(arg0, int):
            # Scalar(int value, name="")
            if arg1 is not None and isinstance(arg1, str):
                self._base = pypto_impl.Scalar(arg0, arg1)
            else:
                self._base = pypto_impl.Scalar(arg0)
        elif isinstance(arg0, str):
            # Scalar(std::string typeName, std::string name="", ScalarValueKind valueKind=...)
            if arg2 is not None:
                # Three arguments: typeName, name, valueKind
                if isinstance(arg1, str):
                    self._base = pypto_impl.Scalar(arg0, arg1, arg2)
                else:
                    raise ValueError(f"Invalid arguments: when arg2 is provided, arg1 must be str, got {type(arg1)}")
            elif isinstance(arg1, str):
                # typeName and name provided
                self._base = pypto_impl.Scalar(arg0, arg1)
            elif arg1 is None:
                # Only typeName provided, name will be empty
                self._base = pypto_impl.Scalar(arg0)
            else:
                # arg1 is int - this is not supported, treat as name by converting to string
                self._base = pypto_impl.Scalar(arg0, str(arg1))
        elif isinstance(arg0, Scalar):
            self._base = arg0._base
        else:
            raise ValueError(f"Invalid arguments: arg0={arg0}, arg1={arg1}, arg2={arg2}")

    def __str__(self) -> str:
        return self._base.Dump()

    def __repr__(self) -> str:
        return f"Scalar({self._base.Dump()})"

    def __eq__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__eq__', lambda a, b: a == b, lambda a, b: a.Eq(b))

    def __ne__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__ne__', lambda a, b: a != b, lambda a, b: a.Ne(b))

    def __lt__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__lt__', lambda a, b: a < b, lambda a, b: a.Lt(b))

    def __le__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__le__', lambda a, b: a <= b, lambda a, b: a.Le(b))

    def __gt__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__gt__', lambda a, b: a > b, lambda a, b: a.Gt(b))

    def __ge__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__ge__', lambda a, b: a >= b, lambda a, b: a.Ge(b))

    def __add__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__add__', lambda a, b: a + b, lambda a, b: a.Add(b))

    def __radd__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__radd__', lambda a, b: b + a, lambda a, b: a.RAdd(b))

    def __sub__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__sub__', lambda a, b: a - b, lambda a, b: a.Sub(b))

    def __rsub__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__rsub__', lambda a, b: b - a, lambda a, b: a.RSub(b))

    def __mul__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__mul__', lambda a, b: a * b, lambda a, b: a.Mul(b))

    def __rmul__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__rmul__', lambda a, b: b * a, lambda a, b: a.RMul(b))

    def __truediv__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__truediv__', lambda a, b: a // b, lambda a, b: a.Div(b))

    def __rtruediv__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__rtruediv__', lambda a, b: b // a, lambda a, b: a.RDiv(b))

    def __mod__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__mod__', lambda a, b: a % b, lambda a, b: a.Mod(b))

    def __rmod__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__rmod__', lambda a, b: b % a, lambda a, b: a.RMod(b))

    def __floordiv__(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__floordiv__', lambda a, b: a // b, lambda a, b: a.Div(b))

    def __rfloordiv__(self, other: int) -> 'Scalar':
        return self._binary_ops(other, '__rfloordiv__', lambda a, b: b // a, lambda a, b: a.RDiv(b))

    def __neg__(self) -> 'Scalar':
        return self._unary_ops(lambda a: -a, lambda a: a.Neg())

    def __pos__(self) -> 'Scalar':
        return self._unary_ops(lambda a: +a, lambda a: a.Pos())

    def __invert__(self) -> 'Scalar':
        return self._unary_ops(lambda a: not a, lambda a: a.Not())

    def __int__(self) -> int:
        return self.concrete()

    def __bool__(self) -> bool:
        return bool(self.concrete())

    @classmethod
    def from_base(cls, base: pypto_impl.Scalar) -> 'Scalar':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def is_immediate(self) -> bool:
        return self._base.IsImmediate()

    def is_symbol(self) -> bool:
        return self._base.IsSymbol()

    def is_expression(self) -> bool:
        return self._base.IsExpression()

    def is_concrete(self) -> bool:
        return self._base.ConcreteValid()

    def concrete(self) -> int:
        if self.is_concrete():
            return self._base.Concrete()
        else:
            raise ValueError("Not concrete value")

    def as_variable(self) -> None:
        self._base.AsIntermediateVariable()

    def base(self) -> pypto_impl.Scalar:
        return self._base

    def min(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__min__', lambda a, b: min(a, b), lambda a, b: a.Min(b))

    def max(self, other: 'Scalar | int') -> 'Scalar':
        return self._binary_ops(other, '__max__', lambda a, b: max(a, b), lambda a, b: a.Max(b))

    def _binary_ops(self, other, name: str, bop, sym_bop):
        if isinstance(other, int):
            if self.is_concrete():
                out = Scalar(bop(self.concrete(), other))
            else:
                out = self.from_base(sym_bop(self._base, other))
        else:
            if self.is_concrete() and other.is_concrete():
                out = Scalar(bop(self.concrete(), other.concrete()))
            else:
                out = self.from_base(sym_bop(self._base, other._base))

        if not out.is_concrete():
            expr = str(out)
            try:
                expr = sympy.simplify(expr)
                if isinstance(expr, sympy.Integer):
                    out = Scalar(int(expr))
                elif expr == sympy.true:
                    out = Scalar(1)
                elif expr == sympy.false:
                    out = Scalar(0)
            except Exception:
                pass
        return out

    def _unary_ops(self, uop, sym_uop):
        if self.is_concrete():
            return Scalar(uop(self.concrete()))
        else:
            return self.from_base(sym_uop(self._base))


SymInt = Union[int, Scalar]