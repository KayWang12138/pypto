"""VF API module - Python API naming follows AscendC style.

Provides direct VF instruction control, bypassing the PTO-ISA intermediate layer.
API naming references AscendC for easy migration. Backend directly emits VF
instructions (vlds, vmax, vadd, etc.) without a C++ API middle layer.

Example::

    import pypto_block.language as pl
    import pypto_block.language.op.vf_api as vf

    with vf.vf_scope(name="max_reduce"):
        max0 = vf.RegTensor(pl.FP32)
        src0 = vf.RegTensor(pl.FP32)
        preg = vf.CreateMask(pl.ALL, dtype=pl.UINT16)
        vf.Duplicate(-1e9, max0)
        vf.LoadAlign(src0, src_ub, offset=0)
        vf.Max(max0, src0, max0, preg)
        vf.StoreAlign(dst_ub, max0, preg)
"""

from pypto_block.pypto_core import DataType
from pypto_block.pypto_core import ir as _ir_core
from pypto_block.pypto_core.ir import Expr, Span

ALL = "ALL"


def _span() -> Span:
    return Span.unknown()


class RegTensor:
    """Vector register (references AscendC RegTensor<T>).

    Must be declared inside a vf.vf_scope() context.

    Example::

        with vf.vf_scope(name="kernel"):
            max0 = vf.RegTensor(pl.FP32)
    """

    def __init__(self, dtype: DataType):
        self._dtype = dtype
        self._expr = None
        if VFScope.get_current() is None:
            raise RuntimeError(
                "RegTensor must be declared inside vf.vf_scope(). "
                "Use: with vf.vf_scope(name='...'): reg = vf.RegTensor(dtype)"
            )
        kwargs = {"dtype": dtype}
        self._expr = _ir_core.create_op_call("vf.RegTensor", [], kwargs, _span())

    def unwrap(self):
        return self._expr


class VFScope:
    """VF API scope context manager. Marks VF API code boundaries."""

    _current_scope = None

    def __init__(self, name: str = None):
        self._name = name or "vf_scope"

    def __enter__(self):
        if VFScope._current_scope is not None:
            raise RuntimeError(
                f"vf.vf_scope() does not support nesting. "
                f"Already inside scope '{VFScope._current_scope._name}'"
            )
        VFScope._current_scope = self
        _ir_core.create_op_call(
            "vf.vf_scope_enter", [], {"name": self._name}, _span()
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ir_core.create_op_call(
            "vf.vf_scope_exit", [], {"name": self._name}, _span()
        )
        VFScope._current_scope = None
        return False

    @classmethod
    def get_current(cls):
        return cls._current_scope


def vf_scope(name: str = None) -> VFScope:
    """Create a VF API scope context manager.

    Args:
        name: Scope name for debugging and code markers.
    """
    return VFScope(name)


def CreateMask(pattern: str, dtype: DataType):
    """Declare and initialize a mask register (references AscendC CreateMask).

    Backend emits: MaskReg preg; plt_b32(preg, ALL);

    Example::

        preg = vf.CreateMask(pl.ALL, dtype=pl.UINT16)
    """
    kwargs = {"pattern": pattern, "dtype": dtype}
    return _ir_core.create_op_call("vf.CreateMask", [], kwargs, _span())


def Duplicate(dst: RegTensor, scalar, mask=None) -> None:
    """Scalar broadcast to register (references AscendC Duplicate).

    Args:
        dst: Destination register
        scalar: Scalar value to broadcast
        mask: Mask register (optional, if omitted uses vbr)

    Backend emits:
        With mask:    vdup(dstReg, scalar, preg, MODE_ZEROING)
        Without mask: vbr(dstReg, scalar)

    Example::

        vf.Duplicate(max0, -1e9)
        vf.Duplicate(max0, -1e9, preg)
    """
    # All go as positional args: [dst, scalar_const, (optional) mask]
    args = [dst.unwrap()]
    # scalar as ConstFloat IR node
    if isinstance(scalar, Expr):
        args.append(scalar)
    else:
        args.append(_ir_core.ConstFloat(float(scalar), DataType.FP32, _span()))
    if mask is not None:
        args.append(mask)
    dst._expr = _ir_core.create_op_call("vf.Duplicate", args, {}, _span())


def LoadAlign(dst, ub_ptr, offset=0, dist: str = "NORM") -> None:
    """Load aligned data from UB to register (references AscendC LoadAlign).

    Args:
        dst: Destination register (RegTensor or MaskReg result)
        ub_ptr: Source UB pointer (Tile or Expr)
        offset: Load offset (int or IR expression)
        dist: Distribution mode ("NORM", "E2B_B32", "E2B_B16", "DS", "BLK")

    Backend emits:
        vlds(dstReg, ub_ptr, offset, dist)       — for RegTensor
        plds(mask, ub_ptr, offset, dist)         — for MaskReg (dist=DS)

    Example::

        vf.LoadAlign(src0, src_ub, iter_m * m * 4)
        vf.LoadAlign(preg, mask_ub, iter_m * m, dist="DS")
        vf.LoadAlign(broadcast_reg, ori_ub, i * 8, dist="E2B_B32")
    """
    dst_val = dst.unwrap() if hasattr(dst, 'unwrap') else dst
    args = [dst_val]
    if isinstance(ub_ptr, Expr):
        args.append(ub_ptr)
    # offset as positional arg to support dynamic IR expressions
    if isinstance(offset, Expr):
        args.append(offset)
    else:
        args.append(_ir_core.ConstInt(int(offset), DataType.INDEX, _span()))
    kwargs = {"dist": dist}
    result = _ir_core.create_op_call("vf.LoadAlign", args, kwargs, _span())
    if hasattr(dst, '_expr'):
        dst._expr = result
    return result


def StoreAlign(dst_ptr, src, mask, block_stride=None, repeat_stride=None,
               dist: str = "NORM_B32", post_update: bool = False,
               data_copy_mode: str = "NORM") -> None:
    """Store aligned data from register to UB (references AscendC StoreAlign).

    Args:
        dst_ptr: Destination UB pointer (Tile or Expr)
        src: Source register
        mask: Mask register
        block_stride: Block stride (for DATA_BLOCK_COPY mode)
        repeat_stride: Repeat stride (for DATA_BLOCK_COPY + POST_UPDATE mode)
        dist: Distribution mode ("NORM_B32", "NORM_B16", "NORM_B8")
        post_update: Enable pointer auto-increment (POST_MODE_UPDATE)
        data_copy_mode: "NORM" or "DATA_BLOCK_COPY"

    Backend emits:
        Basic:          vsts(src, dst, 0, dist, mask)
        PostUpdate:     vsts(src, dst_ref, stride, dist, mask, POST_UPDATE)
        DataBlockCopy:  vsstb(src, dst_ref, (blockStride<<16)|repeatStride, mask, POST_UPDATE)

    Example::

        vf.StoreAlign(dst_ub, max0, preg)
        vf.StoreAlign(dst_ub, reg, preg, dist="NORM_B16")
        vf.StoreAlign(dst_ptr, reg, preg, dist="NORM_B32", post_update=True)
        vf.StoreAlign(x_exp, reg, preg, block_stride=bs, repeat_stride=rs,
                      data_copy_mode="DATA_BLOCK_COPY", post_update=True)
    """
    src_val = src.unwrap() if hasattr(src, 'unwrap') else src
    dst_val = dst_ptr.unwrap() if hasattr(dst_ptr, 'unwrap') else dst_ptr
    args = [dst_val, src_val, mask]
    # block_stride and repeat_stride as positional args
    if block_stride is not None:
        if isinstance(block_stride, Expr):
            args.append(block_stride)
        else:
            args.append(_ir_core.ConstInt(int(block_stride), DataType.INDEX, _span()))
    if repeat_stride is not None:
        if isinstance(repeat_stride, Expr):
            args.append(repeat_stride)
        else:
            args.append(_ir_core.ConstInt(int(repeat_stride), DataType.INDEX, _span()))
    kwargs = {"dist": dist, "post_update": post_update, "data_copy_mode": data_copy_mode}
    _ir_core.create_op_call("vf.StoreAlign", args, kwargs, _span())


def MemBar(mode: str = "VST_VLD") -> None:
    """Memory barrier (references mem_bar).

    Backend emits: mem_bar(VST_VLD)

    Example::

        vf.MemBar()
        vf.MemBar("VST_VLD")
    """
    kwargs = {"mode": mode}
    _ir_core.create_op_call("vf.MemBar", [], kwargs, _span())


def Max(dst: RegTensor, src0: RegTensor, src1: RegTensor, mask) -> None:
    """Vector maximum (references AscendC Max).

    Backend emits: vmax(dstReg, src0Reg, src1Reg, preg, ZEROING)

    Example::

        vf.Max(max0, src0, max0, preg)
    """
    args = [src0.unwrap(), src1.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Max", args, {}, _span())


def Add(dst: RegTensor, src0: RegTensor, src1: RegTensor, mask) -> None:
    """Vector addition (references AscendC Add).

    Backend emits: vadd(dstReg, src0Reg, src1Reg, preg, MODE_ZEROING)
    """
    args = [src0.unwrap(), src1.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Add", args, {}, _span())


def Sub(dst: RegTensor, src0: RegTensor, src1: RegTensor, mask) -> None:
    """Vector subtraction (references AscendC Sub).

    Backend emits: vsub(dstReg, src0Reg, src1Reg, preg, MODE_ZEROING)
    """
    args = [src0.unwrap(), src1.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Sub", args, {}, _span())


def Mul(dst: RegTensor, src0: RegTensor, src1: RegTensor, mask) -> None:
    """Vector multiplication (references AscendC Mul).

    Backend emits: vmul(dstReg, src0Reg, src1Reg, preg, MODE_ZEROING)
    """
    args = [src0.unwrap(), src1.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Mul", args, {}, _span())


def Muls(dst: RegTensor, src: RegTensor, scalar, mask) -> None:
    """Scalar multiplication (references AscendC Muls).

    Backend emits: vmuls(dstReg, srcReg, scalar, preg)
    """
    args = [src.unwrap(), dst.unwrap(), mask]
    kwargs = {"scalar": scalar}
    dst._expr = _ir_core.create_op_call("vf.Muls", args, kwargs, _span())


def Ln(dst: RegTensor, src: RegTensor, mask) -> None:
    """Natural logarithm (references AscendC Ln).

    Backend emits: vln(dstReg, srcReg, preg, MODE_ZEROING)
    """
    args = [src.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Ln", args, {}, _span())


def FusedExpSub(dst: RegTensor, src: RegTensor, max_reg: RegTensor, mask) -> None:
    """Fused exp(src - max) (references AscendC FusedExpSub).

    Backend emits: vexpdif(dstReg, srcReg, maxReg, preg, PART_ZERO)
    """
    args = [src.unwrap(), max_reg.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.FusedExpSub", args, {}, _span())


def Cast(dst: RegTensor, src: RegTensor, mask,
         layout: str = "ZERO", round_mode: str = "CAST_ROUND") -> None:
    """Type conversion (references AscendC Cast).

    Args:
        dst: Destination register
        src: Source register
        mask: Mask register
        layout: Register layout ("ZERO", "ONE", "TWO", "THREE")
                ZERO=PART_EVEN, ONE=PART_ODD, TWO/THREE for multi-reg
        round_mode: Rounding mode ("CAST_ROUND" or "CAST_RINT")
                    CAST_ROUND → RS_DISABLE, CAST_RINT → RS_ENABLE

    Backend emits: vcvt(dst, src, mask, rs_mode, part_mode)

    Example::

        vf.Cast(dst_bf16, src_f32, preg, layout="ZERO")
        vf.Cast(dst_bf16, src_f32, preg, layout="ONE", round_mode="CAST_RINT")
    """
    args = [src.unwrap(), dst.unwrap(), mask]
    kwargs = {"layout": layout, "round_mode": round_mode}
    dst._expr = _ir_core.create_op_call("vf.Cast", args, kwargs, _span())


def DeInterleave(dst0: RegTensor, dst1: RegTensor,
                 src0: RegTensor, src1: RegTensor) -> None:
    """De-interleave (references AscendC DeInterleave).

    Backend emits: vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg)
    """
    args = [src0.unwrap(), src1.unwrap(), dst0.unwrap(), dst1.unwrap()]
    result = _ir_core.create_op_call("vf.DeInterleave", args, {}, _span())
    dst0._expr = result
    dst1._expr = result


def Select(dst: RegTensor, src_true: RegTensor, src_false: RegTensor, mask) -> None:
    """Conditional select (references AscendC Select).

    Backend emits: vsel(dstReg, srcTrueReg, srcFalseReg, mask)
    """
    args = [src_true.unwrap(), src_false.unwrap(), dst.unwrap(), mask]
    dst._expr = _ir_core.create_op_call("vf.Select", args, {}, _span())


def UpdateMask(scalar, dtype: DataType = None):
    """Update mask with scalar value (references AscendC UpdateMask).

    Backend emits: plt_b32(scalar, POST_UPDATE) or plt_b16(...)
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if isinstance(scalar, Expr):
        return _ir_core.create_op_call("vf.UpdateMask", [scalar], kwargs, _span())
    else:
        args = [_ir_core.ConstInt(int(scalar), DataType.INDEX, _span())]
        return _ir_core.create_op_call("vf.UpdateMask", args, kwargs, _span())


__all__ = [
    "ALL",
    "RegTensor",
    "VFScope",
    "vf_scope",
    "CreateMask",
    "UpdateMask",
    "Duplicate",
    "LoadAlign",
    "StoreAlign",
    "MemBar",
    "Max",
    "Add",
    "Sub",
    "Mul",
    "Muls",
    "Ln",
    "FusedExpSub",
    "Cast",
    "DeInterleave",
    "Select",
]
