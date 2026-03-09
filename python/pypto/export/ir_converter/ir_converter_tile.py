from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, Optional

from pypto_ir import DataType, ir
from pypto_ir.ir import IRBuilder
import pypto

# -----------------------------
# Helper Functions
# -----------------------------

def flatten_ABC(A: int, B: int, C: int) -> int:
    return A * B * C

def check_divisible(n: int, d: int, what: str):
    if n % d != 0:
        raise ValueError(f"{what} not divisible: {n} % {d} != 0")

def _as_expr(x) -> ir.Expr:
    if isinstance(x, ir.Expr):
        return x

    if isinstance(x, ir.MakeTuple):
        return x # type: ignore[return-value]

    # Python int -> ConstInt Expr
    if isinstance(x, int):
        return ir.ConstInt(x, DataType.INT32, ir.Span.unknown())

    raise TypeError(f"mt(): unsupported element type: {type(x)}")

def mt(xs) -> ir.MakeTuple:
    if isinstance(xs, ir.MakeTuple):
        return xs
    return ir.MakeTuple([_as_expr(v) for v in xs], ir.Span.unknown())


# -----------------------------
# IR tracing context + proxies
# -----------------------------

@dataclass
class TraceCtx:
    ib: IRBuilder
    tmp_id: int = 0

    def fresh(self, prefix: str) -> str:
        self.tmp_id += 1
        return f"{prefix}_{self.tmp_id}"

class TileProxy:
    """Tile SSA value -> lowered to ir.op.block.*"""

    def __init__(self, ctx: TraceCtx, value: Any, tile_rows: int, tile_cols: int, dtype: DataType):
        self.ctx = ctx
        self.value = value
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.dtype = dtype

    def __add__(self, other: "TileProxy") -> "TileProxy":
        return emit_binop("tile.add", self, other)

    def __sub__(self, other: "TileProxy") -> "TileProxy":
        return emit_binop("tile.sub", self, other)

    def __mul__(self, other: "TileProxy") -> "TileProxy":
        return emit_binop("tile.mul", self, other)

    def divs(self, scalar_f: float) -> "TileProxy":
        return emit_unary("tile.divs", self, scalar_f)

    def __repr__(self):
        return f"TileProxy({self.tile_rows}x{self.tile_cols}, {self.dtype})"

# -----------------------------
# Lowering registry
# -----------------------------

LOWER: Dict[str, Callable[..., Any]] = {}

def lower(name: str):
    def deco(fn):
        LOWER[name] = fn
        return fn
    return deco

def emit_binop(opname: str, a: TileProxy, b: TileProxy) -> TileProxy:
    if opname not in LOWER:
        raise NotImplementedError(f"No lowering registered for {opname}")
    if (a.tile_rows, a.tile_cols, a.dtype) != (b.tile_rows, b.tile_cols, b.dtype):
        raise ValueError(f"Tile mismatch: {a} vs {b}")
    return LOWER[opname](a, b)

def emit_unary(opname: str, a: TileProxy, *args) -> TileProxy:
    if opname not in LOWER:
        raise NotImplementedError(f"No lowering registered for {opname}")
    return LOWER[opname](a, *args)

# ---- tile binops ----
@lower("tile.add")
def _lower_tile_add(a: TileProxy, b: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("add"), ir.op.block.add(a.value, b.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

@lower("tile.sub")
def _lower_tile_sub(a: TileProxy, b: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("sub"), ir.op.block.sub(a.value, b.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

@lower("tile.mul")
def _lower_tile_mul(a: TileProxy, b: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("mul"), ir.op.block.mul(a.value, b.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

@lower("tile.divs")
def _lower_tile_divs(a: TileProxy, scalar_f: float) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("divs"), ir.op.block.divs(a.value, scalar_f))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

# ---- tile binops ----
@lower("tile.exp")
def _lower_tile_exp(a: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("exp"), ir.op.block.exp(a.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

@lower("tile.log")
def _lower_tile_log(a: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("log"), ir.op.block.exp(a.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

@lower("tile.neg")
def _lower_tile_neg(a: TileProxy) -> TileProxy:
    v = a.ctx.ib.let(a.ctx.fresh("neg"), ir.op.block.exp(a.value))
    return TileProxy(a.ctx, v, a.tile_rows, a.tile_cols, a.dtype)

# -----------------------------
# Converter Function
# -----------------------------

def convert_kernel_to_tile_ir(
    kernel_fn: Callable[..., Any],
    *,
    program_name: str,
    func_name: str,
    input_shapes: Tuple[Tuple[int, ...], ...],
    vec_tile_shapes: Optional[Tuple[int, int, int, int]] = None,
    cube_tile_shapes: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
    dtype: DataType = DataType.FP32,
) -> ir.Program:
    ib = IRBuilder()
    ctx = TraceCtx(ib=ib)

    with ib.function(func_name) as f:
        params = []
        for i, shp in enumerate(input_shapes):
            params.append(f.param(f"arg{i}", ir.TensorType(list(shp), dtype)))

        if vec_tile_shapes is None:
            raise ValueError("requires vec_tile_shapes=(a,b,c,d)")
        if len(input_shapes) not in (1, 2):
            raise ValueError("supports 1 or 2 inputs")
        if len(input_shapes[0]) != 4:
            raise ValueError("expects 4D tensor shapes")
        if len(input_shapes) == 2 and input_shapes[0] != input_shapes[1]:
            raise ValueError("requires both inputs to have same 4D shape")

        A, B, C, D = input_shapes[0]
        ta, tb, tc, td = vec_tile_shapes

        M = flatten_ABC(A, B, C)
        N = D
        TR = flatten_ABC(ta, tb, tc)
        TC = td

        check_divisible(M, TR, "Flattened row dim (A*B*C)")
        check_divisible(N, TC, "Last dim D")
        tiles_m = M // TR
        tiles_n = N // TC

        out_param = f.param("out", ir.TensorType([A, B, C, D], dtype))
        f.return_type(ir.TensorType([A, B, C, D], dtype))

        tiles_m_c = ib.let("tiles_m", ir.ConstInt(tiles_m, DataType.INT32, ir.Span.unknown()))
        tiles_n_c = ib.let("tiles_n", ir.ConstInt(tiles_n, DataType.INT32, ir.Span.unknown()))
        ti = ib.var("ti", ir.ScalarType(DataType.INT32))
        tj = ib.var("tj", ir.ScalarType(DataType.INT32))

        with ib.for_loop(ti, 0, tiles_m_c, 1) as loop_i:
            out_i = loop_i.iter_arg("out_i", out_param)
            loop_i.return_var("out_i_updated")

            with ib.for_loop(tj, 0, tiles_n_c, 1) as loop_j:
                out_ij = loop_j.iter_arg("out_ij", out_i)
                loop_j.return_var("out_ij_updated")

                base_row = ib.let(
                    ctx.fresh("base_row"),
                    ti * ir.ConstInt(TR, DataType.INT32, ir.Span.unknown()),
                )
                base_col = ib.let(
                    ctx.fresh("base_col"),
                    tj * ir.ConstInt(TC, DataType.INT32, ir.Span.unknown()),
                )

                x_tile_v = ib.let(
                    ctx.fresh("x_tile"),
                    ir.op.block.load(
                        params[0],
                        mt([base_row, base_col]),
                        mt([TR, TC]),
                        ir.MemorySpace.UB,
                    ),
                )
                X = TileProxy(ctx, x_tile_v, TR, TC, dtype)

                if len(input_shapes) == 2:
                    y_tile_v = ib.let(
                        ctx.fresh("y_tile"),
                        ir.op.block.load(
                            params[1],
                            mt([base_row, base_col]),
                            mt([TR, TC]),
                            ir.MemorySpace.UB,
                        ),
                    )
                    Y = TileProxy(ctx, y_tile_v, TR, TC, dtype)

                # with patch_pypto_for_trace():
                out_tile = kernel_fn(X) if len(input_shapes) == 1 else kernel_fn(X, Y)

                if not isinstance(out_tile, TileProxy):
                    raise TypeError("vec tile lowering expects TileProxy output")

                _ = ib.let(
                    ctx.fresh("out_ij_stored"),
                    ir.op.block.store(
                        out_tile.value,                 # TileType
                        mt([base_row, base_col]),       # offsets
                        mt([TR, TC]),                   # sizes
                        out_param,                      # destination (loop-carried)
                    ),
                )

                ib.emit(ir.YieldStmt([out_ij], ir.Span.unknown()))

            out_i_updated = loop_j.output()
            ib.emit(ir.YieldStmt([out_i_updated], ir.Span.unknown()))

        out_final = loop_i.output()
        ib.return_stmt(out_final)

    func = f.get_result()
    return ir.Program([func], program_name, ir.Span.unknown())
