#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
IR BNF:

<DataType>        ::= I8 | S8 | FP16 | ...

<Var>               ::= <ScalarVar>
                     |  <TokenVar>
                     |  <TensorVar>
<ScalarVar>         ::= % <Symbol>
<TokenVar>          ::= % <Symbol>
<TensorVar>         ::= % <Symbol> <DataType> <RawTensorVar> <ConstInt>*, <ScalarVar>*, <ConstInt>*, <ScalarVar>* # raw-tensor, static-shape, dyn-valid-shape, static-offset, dyn-offset
<RawTensorVar>      ::= % <Symbol> <DataType> <ConstInt>*, <ScalarVar>* # static-shape, dyn-valid-shape
<Attr>              ::= # <Symbol> = <ScalarExpr>
                     |  # <Symbol> = { <ScalarExpr>* }

<ScalarExpr>        ::= <ConstInt>
                     |  <ConstFloat>
                     |  <ScalarVar>

<Stmt>              ::= <OpStmt>
                     |  <IfStmt>
                     |  <ForStmt>
                     |  <YieldStmt>
                     |  <ReturnStmt>
                     |  <SeqStmt>
<OpStmt>            ::= <AssignStmt>+
<IfStmt>            ::= <Var>* = IF <CoreAttr>? <ScalarExpr> THEN <Stmt>* ELSE <Stmt>*
<ForIterExpr>       ::= <ScalarExpr> | <TensorVar> | <TokenVar>
<ForStmt>           ::= <Var>* = FOR <CoreAttr>? ITERARG <Var>*, <ForIterExpr>*
                     |    LOOP <ScalarVar> : <ScalarExpr> , <ScalarExpr> , <ScalarExpr>
                     |    BODY <Stmt>*
<YieldStmt>         ::= YIELD <Var>*
<ReturnStmt>        ::= RETURN <Var>*
<SeqStmt>           ::= <Stmt>+

<AssignStmt>        ::= <ScalarExprOpAssignStmt>
                     |  <ScalarOpAssignStmt>
                     |  <TensorOpAssignStmt>
                     |  <CallOpAssignStmt>
<ScalarExprOpAssignStmt>  ::= <ScalarVar>*, <TokenVar> = <ScalarExprOp> <ScalarExpr>*
<ScalarOpAssignStmt>::= <ScalarVar>*, <TokenVar> = OP_GET_TENSOR_DATA <TensorVar> , <TokenVar>* , <CoreAttr>? , <Attr>*
<TensorOpAssignStmt>::= <TensorVar>*, <TokenVar> = <TensorOp> <TensorVar>* , <TokenVar>* , <CoreAttr>? , <Attr>*
<CallOpAssignStmt>  ::= <TensorVar>*, <ScalarVar>*, <TokenVar>* = <Var> <TensorVar>*, <ScalarVar>*, <TokenVar>*, <CoreAttr>?, <Attr>*

<Function>          ::= @ <Symbol> <Var>* , <Var>* , <Attr>* , <SeqStmt>
<Config>            ::= <Attr>*
<Module>            ::= <Function>* <Config>*
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScalarDType(Enum):
    """Scalar element data types supported in the IR."""
    I8   = "i8"
    S8   = "s8"
    U8   = "u8"
    I16  = "i16"
    U16  = "u16"
    I32  = "i32"
    U32  = "u32"
    I64  = "i64"
    U64  = "u64"
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    BF16 = "bf16"


class UnaryOp(Enum):
    NEG = "neg"
    NOT = "not"
    ABS = "abs"


class BinaryOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    AND = "and"
    OR  = "or"
    EQ  = "eq"
    NE  = "ne"
    LT  = "lt"
    LE  = "le"
    GT  = "gt"
    GE  = "ge"


class RuntimeOp(Enum):
    """Built-in runtime scalar operations."""
    CAST     = "cast"
    SIZEOF   = "sizeof"
    CEIL_DIV = "ceil_div"
    MIN      = "min"
    MAX      = "max"


# Union of all scalar-expression opcodes used in ScalarExprOpAssignStmt.
ScalarExprOp = Union["UnaryOp", "BinaryOp", "RuntimeOp"]

# <ForIterExpr> ::= <ScalarExpr> | <TensorVar> | <TokenVar>
ForIterExpr = Union["ScalarExpr", "TensorVar", "TokenVar"]



# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------

Symbol = str  # % or @ prefixed identifiers are represented as plain strings


class ScalarExpr(ABC):
    """Abstract base for scalar expressions (ConstInt | ConstFloat | ScalarVar)."""


class Var(ABC):
    """Abstract base for IR variables (%symbol)."""


class Stmt(ABC):
    """Abstract base for IR statements."""


StmtList = list[Stmt]


class AssignStmt(Stmt, ABC):
    """Abstract base for assignment statements."""


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

@dataclass
class CoreAttr:
    """Optional core-binding annotation that may appear in control-flow ops."""
    core_id: ScalarExpr


@dataclass
class Attr:
    """# name = expr  |  # name = { expr* }"""
    name:  Symbol
    value: Union[ScalarExpr, List[ScalarExpr]]


# ---------------------------------------------------------------------------
# Scalar Expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstInt(ScalarExpr):
    """Integer constant."""
    value: int


@dataclass(frozen=True)
class ConstFloat(ScalarExpr):
    """Floating-point constant."""
    value: float


# ---------------------------------------------------------------------------
# Variables  (%symbol)
# Note: ScalarVar inherits from both Var and ScalarExpr because the BNF
#       lists <ScalarVar> as a valid <ScalarExpr> alternative.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarVar(Var, ScalarExpr):
    """A scalar SSA variable; also usable as a scalar expression."""
    symbol: Symbol


@dataclass(frozen=True)
class TokenVar(Var):
    """A dependency-token SSA variable."""
    symbol: Symbol


@dataclass
class RawTensorVar(Var):
    """The underlying physical tensor buffer: % <Symbol> <DataType> <ConstInt>*, <ScalarVar>*
    Corresponds to <RawTensorVar> in the BNF.
    """
    symbol:       Symbol
    dtype:        ScalarDType
    static_shape: list[int]
    dyn_valid_shape: list[ScalarVar]


@dataclass
class TensorVar(Var):
    """A tensor-buffer SSA variable.
    % <Symbol> <DataType> <RawTensorVar> <ConstInt>*, <ScalarVar>*, <ConstInt>*, <ScalarVar>*
    Fields: raw_tensor, static_shape, dyn_valid_shape, static_offset, dyn_offset
    """
    symbol:          Symbol
    dtype:           ScalarDType
    raw_tensor:      RawTensorVar
    static_shape:    list[int]
    dyn_valid_shape: list[ScalarVar]
    static_offset:   list[int]
    dyn_offset:      list[ScalarVar]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Assignment Statements
# ---------------------------------------------------------------------------

@dataclass
class ScalarExprOpAssignStmt(AssignStmt):
    """<ScalarVar>*, <TokenVar> = <ScalarExprOp> <ScalarExpr>*"""
    scalar_results: List[ScalarVar]
    token_result:   TokenVar
    op:             ScalarExprOp
    args:           List[ScalarExpr]


@dataclass
class ScalarOpAssignStmt(AssignStmt):
    """<ScalarVar>*, <TokenVar> = OP_GET_TENSOR_DATA <TensorVar> , <TokenVar>* , <CoreAttr>? , <Attr>*"""
    scalar_results: List[ScalarVar]
    token_result:   TokenVar
    tensor:         TensorVar
    tokens:         List[TokenVar]       = field(default_factory=list)
    core_attr:      Optional[CoreAttr]  = None
    attrs:          List[Attr]           = field(default_factory=list)


@dataclass
class TensorOpAssignStmt(AssignStmt):
    """<TensorVar>*, <TokenVar> = <TensorOp> <TensorVar>* , <TokenVar>* , <CoreAttr>? , <Attr>*"""
    tensor_results: List[TensorVar]
    token_result:   TokenVar
    op:             str
    tensors:        List[TensorVar]      = field(default_factory=list)
    tokens:         List[TokenVar]       = field(default_factory=list)
    core_attr:      Optional[CoreAttr]  = None
    attrs:          List[Attr]           = field(default_factory=list)


@dataclass
class CallOpAssignStmt(AssignStmt):
    """<TensorVar>*, <ScalarVar>*, <TokenVar>* = <Var> <TensorVar>*, <ScalarVar>*, <TokenVar>*, <CoreAttr>?, <Attr>*"""
    tensor_results: List[TensorVar]
    scalar_results: List[ScalarVar]
    token_results:  List[TokenVar]
    callee:         Var
    tensor_args:    List[TensorVar]     = field(default_factory=list)
    scalar_args:    List[ScalarVar]     = field(default_factory=list)
    token_args:     List[TokenVar]      = field(default_factory=list)
    core_attr:      Optional[CoreAttr]  = None
    attrs:          List[Attr]          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composite / Control-Flow Statements
# ---------------------------------------------------------------------------

@dataclass
class OpStmt(Stmt):
    """One or more assignment statements forming a single operator step."""
    assigns: List[AssignStmt]


@dataclass
class IfStmt(Stmt):
    """result* = IF core_attr? cond THEN stmt* ELSE stmt*"""
    results:   List[Var]
    cond:      ScalarExpr
    then_body: List[Stmt]
    else_body: List[Stmt]
    core_attr: Optional[CoreAttr] = None


@dataclass
class ForStmt(Stmt):
    """result* = FOR core_attr? loop_var : start , stop , step
                 ITERARG var*, iter_expr*
                 BODY stmt*
    """
    results:        List[Var]
    iter_arg_vars:  List[Var]
    iter_arg_inits: List[ForIterExpr]
    loop_var:       ScalarVar
    start:          ScalarExpr
    stop:           ScalarExpr
    step:           ScalarExpr
    body:           List[Stmt]
    core_attr:      Optional[CoreAttr] = None


@dataclass
class WhileStmt(Stmt):
    """result* = WHILE core_attr? cond BODY stmt"""
    results:   List[Var]
    cond:      ScalarExpr
    body:      Stmt
    core_attr: Optional[CoreAttr] = None


@dataclass
class YieldStmt(Stmt):
    """YIELD var*  – carries loop-iteration values to the next iteration."""
    vars: List[Var]


@dataclass
class ReturnStmt(Stmt):
    """RETURN var*  – returns values from a function."""
    vars: List[Var]


@dataclass
class SeqStmt(Stmt):
    """An ordered sequence of statements."""
    stmts: List[Stmt]


# ---------------------------------------------------------------------------
# Top-level IR nodes
# ---------------------------------------------------------------------------

@dataclass
class Function:
    """@ symbol inputs , outputs , attr* , body"""
    symbol:  Symbol
    inputs:  List[Var]
    outputs: List[Var]
    attrs:   List[Attr]
    body:    SeqStmt


@dataclass
class Config:
    """Module-level configuration block: attr*"""
    attrs: List[Attr]


@dataclass
class Module:
    """A complete IR module: function* config*"""
    functions: List[Function] = field(default_factory=list)
    configs:   List[Config]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# IRBuilder
# ---------------------------------------------------------------------------

class IRBuilder:
    """Factory for constructing IR nodes.

    Each ``create_XXX`` method corresponds to one non-terminal in the BNF and
    returns an instance of the matching IR class.
    """

    # ------------------------------------------------------------------
    # Variables  <Var> / <ScalarVar> / <TokenVar> / <TensorVar>
    # ------------------------------------------------------------------

    def create_scalar_var(self, symbol: Symbol) -> ScalarVar:
        """<ScalarVar> ::= % <Symbol>"""
        return ScalarVar(symbol=symbol)

    def create_token_var(self, symbol: Symbol) -> TokenVar:
        """<TokenVar> ::= % <Symbol>"""
        return TokenVar(symbol=symbol)

    def create_raw_tensor_var(
        self,
        symbol: Symbol,
        dtype: ScalarDType,
        static_shape: List[int],
        dyn_valid_shape: List[ScalarVar],
    ) -> RawTensorVar:
        """<RawTensorVar> ::= % <Symbol> <DataType> <ConstInt>*, <ScalarVar>*"""
        return RawTensorVar(
            symbol=symbol,
            dtype=dtype,
            static_shape=static_shape,
            dyn_valid_shape=dyn_valid_shape,
        )

    def create_tensor_var(
        self,
        symbol: Symbol,
        dtype: ScalarDType,
        raw_tensor: RawTensorVar,
        static_shape: List[int],
        dyn_valid_shape: List[ScalarVar],
        static_offset: List[int],
        dyn_offset: List[ScalarVar],
    ) -> TensorVar:
        """<TensorVar> ::= % <Symbol> <DataType> <RawTensorVar> <ConstInt>*, <ScalarVar>*, <ConstInt>*, <ScalarVar>*"""
        return TensorVar(
            symbol=symbol,
            dtype=dtype,
            raw_tensor=raw_tensor,
            static_shape=static_shape,
            dyn_valid_shape=dyn_valid_shape,
            static_offset=static_offset,
            dyn_offset=dyn_offset,
        )

    # ------------------------------------------------------------------
    # Attributes  <Attr> / <CoreAttr>
    # ------------------------------------------------------------------

    def create_attr(
        self,
        name: Symbol,
        value: Union[ScalarExpr, List[ScalarExpr]],
    ) -> Attr:
        """<Attr> ::= # <Symbol> = <ScalarExpr>
                    | # <Symbol> = { <ScalarExpr>* }
        """
        return Attr(name=name, value=value)

    def create_core_attr(self, core_id: ScalarExpr) -> CoreAttr:
        """<CoreAttr>"""
        return CoreAttr(core_id=core_id)

    # ------------------------------------------------------------------
    # Scalar Expressions  <ScalarExpr>
    # ------------------------------------------------------------------

    def create_const_int(self, value: int) -> ConstInt:
        """<ScalarExpr> ::= <ConstInt>"""
        return ConstInt(value=value)

    def create_const_float(self, value: float) -> ConstFloat:
        """<ScalarExpr> ::= <ConstFloat>"""
        return ConstFloat(value=value)

    # ------------------------------------------------------------------
    # Assignment Statements  <AssignStmt>
    # ------------------------------------------------------------------

    def create_scalar_expr_op_assign_stmt(
        self,
        scalar_results: List[ScalarVar],
        token_result: TokenVar,
        op: ScalarExprOp,
        args: List[ScalarExpr],
    ) -> ScalarExprOpAssignStmt:
        """<ScalarExprOpAssignStmt> ::= <ScalarVar>*, <TokenVar> = <ScalarExprOp> <ScalarExpr>*"""
        stmt = ScalarExprOpAssignStmt(
            scalar_results=scalar_results,
            token_result=token_result,
            op=op,
            args=args,
        )
        self.capture_stmt(stmt)
        return stmt

    def create_scalar_op_assign_stmt(
        self,
        scalar_results: List[ScalarVar],
        token_result: TokenVar,
        tensor: TensorVar,
        tokens: Optional[List[TokenVar]] = None,
        core_attr: Optional[CoreAttr] = None,
        attrs: Optional[List[Attr]] = None,
    ) -> ScalarOpAssignStmt:
        """<ScalarOpAssignStmt> ::= <ScalarVar>*, <TokenVar> = OP_GET_TENSOR_DATA <TensorVar> , <TokenVar>* , <CoreAttr>? , <Attr>*"""
        stmt = ScalarOpAssignStmt(
            scalar_results=scalar_results,
            token_result=token_result,
            tensor=tensor,
            tokens=tokens or [],
            core_attr=core_attr,
            attrs=attrs or [],
        )
        self.capture_stmt(stmt)
        return stmt

    def create_tensor_op_assign_stmt(
        self,
        tensor_results: List[TensorVar],
        token_result: TokenVar,
        op: str,
        tensors: Optional[List[TensorVar]] = None,
        tokens: Optional[List[TokenVar]] = None,
        core_attr: Optional[CoreAttr] = None,
        attrs: Optional[List[Attr]] = None,
    ) -> TensorOpAssignStmt:
        """<TensorOpAssignStmt> ::= <TensorVar>*, <TokenVar> = <TensorOp> <TensorVar>* , <TokenVar>* , <CoreAttr>? , <Attr>*"""
        stmt = TensorOpAssignStmt(
            tensor_results=tensor_results,
            token_result=token_result,
            op=op,
            tensors=tensors or [],
            tokens=tokens or [],
            core_attr=core_attr,
            attrs=attrs or [],
        )
        self.capture_stmt(stmt)
        return stmt

    def create_call_op_assign_stmt(
        self,
        tensor_results: List[TensorVar],
        scalar_results: List[ScalarVar],
        token_results: List[TokenVar],
        callee: Var,
        tensor_args: Optional[List[TensorVar]] = None,
        scalar_args: Optional[List[ScalarVar]] = None,
        token_args: Optional[List[TokenVar]] = None,
        core_attr: Optional[CoreAttr] = None,
        attrs: Optional[List[Attr]] = None,
    ) -> CallOpAssignStmt:
        """<CallOpAssignStmt> ::= <TensorVar>*, <ScalarVar>*, <TokenVar>* = <Var> <TensorVar>*, <ScalarVar>*, <TokenVar>*, <CoreAttr>?, <Attr>*"""
        stmt = CallOpAssignStmt(
            tensor_results=tensor_results,
            scalar_results=scalar_results,
            token_results=token_results,
            callee=callee,
            tensor_args=tensor_args or [],
            scalar_args=scalar_args or [],
            token_args=token_args or [],
            core_attr=core_attr,
            attrs=attrs or [],
        )
        self.capture_stmt(stmt)
        return stmt

    # ------------------------------------------------------------------
    # Statements  <Stmt>
    # ------------------------------------------------------------------

    def create_op_stmt(self, assigns: List[AssignStmt]) -> OpStmt:
        """<OpStmt> ::= <AssignStmt>+"""
        stmt = OpStmt(assigns=assigns)
        self.capture_stmt(stmt)
        return stmt

    def create_if_stmt(
        self,
        results: List[Var],
        cond: ScalarExpr,
        then_body: List[Stmt],
        else_body: List[Stmt],
        core_attr: Optional[CoreAttr] = None,
    ) -> IfStmt:
        """<IfStmt> ::= <Var>* = IF <CoreAttr>? <ScalarExpr> THEN <Stmt>* ELSE <Stmt>*"""
        stmt = IfStmt(
            results=results,
            cond=cond,
            then_body=then_body,
            else_body=else_body,
            core_attr=core_attr,
        )
        self.capture_stmt(stmt)
        return stmt

    def create_for_stmt(
        self,
        results: List[Var],
        iter_arg_vars: List[Var],
        iter_arg_inits: List[ForIterExpr],
        loop_var: ScalarVar,
        start: ScalarExpr,
        stop: ScalarExpr,
        step: ScalarExpr,
        body: List[Stmt],
        core_attr: Optional[CoreAttr] = None,
    ) -> ForStmt:
        """<ForStmt> ::= <Var>* = FOR <CoreAttr>? <ScalarVar> : <ScalarExpr> , <ScalarExpr> , <ScalarExpr>
                          ITERARG <Var>*, <ForIterExpr>*
                          BODY <Stmt>*
        """
        stmt = ForStmt(
            results=results,
            loop_var=loop_var,
            iter_arg_vars=iter_arg_vars,
            iter_arg_inits=iter_arg_inits,
            start=start,
            stop=stop,
            step=step,
            body=body,
            core_attr=core_attr,
        )
        self.capture_stmt(stmt)
        return stmt

    def create_while_stmt(
        self,
        results: List[Var],
        cond: ScalarExpr,
        body: Stmt,
        core_attr: Optional[CoreAttr] = None,
    ) -> WhileStmt:
        """<WhileStmt>"""
        stmt = WhileStmt(
            results=results,
            cond=cond,
            body=body,
            core_attr=core_attr,
        )
        self.capture_stmt(stmt)
        return stmt

    def create_yield_stmt(self, vars: List[Var]) -> YieldStmt:
        """<YieldStmt> ::= YIELD <Var>*"""
        stmt = YieldStmt(vars=vars)
        self.capture_stmt(stmt)
        return stmt

    def create_return_stmt(self, vars: List[Var]) -> ReturnStmt:
        """<ReturnStmt> ::= RETURN <Var>*"""
        stmt = ReturnStmt(vars=vars)
        self.capture_stmt(stmt)
        return stmt

    def create_seq_stmt(self, stmts: List[Stmt]) -> SeqStmt:
        """<SeqStmt> ::= <Stmt>+"""
        stmt = SeqStmt(stmts=stmts)
        self.capture_stmt(stmt)
        return stmt

    # ------------------------------------------------------------------
    # Top-level nodes  <Function> / <Config> / <Module>
    # ------------------------------------------------------------------

    def create_function(
        self,
        symbol: Symbol,
        inputs: List[Var],
        outputs: List[Var],
        attrs: List[Attr],
        body: SeqStmt,
    ) -> Function:
        """<Function> ::= @ <Symbol> <Var>* , <Var>* , <Attr>* , <SeqStmt>"""
        return Function(
            symbol=symbol,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            body=body,
        )

    def create_config(self, attrs: List[Attr]) -> Config:
        """<Config> ::= <Attr>*"""
        return Config(attrs=attrs)

    def create_module(
        self,
        functions: Optional[List[Function]] = None,
        configs: Optional[List[Config]] = None,
    ) -> Module:
        """<Module> ::= <Function>* <Config>*"""
        return Module(
            functions=functions or [],
            configs=configs or [],
        )

    builder = None

    def __init__(self):
        self._capture_list = []
        self._temp_count = 0
        self.set_builder()

    @classmethod
    def get_builder(cls):
        return cls.builder

    def set_builder(self):
        assert IRBuilder.builder is None
        IRBuilder.builder = self

    def temp_name(self) -> str:
        name = f'_tmp_{self._temp_count}'
        self._temp_count += 1
        return name

    def reset(self):
        self._capture_list.clear()
        self._temp_count = 0

    def capture_begin(self):
        self._capture_list.append([])
        pass
    def capture_end(self) -> StmtList:
        last = self._capture_list.pop()
        return last

    def capture_stmt(self, stmt):
        if self._capture_list:
            self._capture_list[-1].append(stmt)

g_builder = IRBuilder()
