from __future__ import annotations


def calc_workspace_fixed(
    x0_shape: tuple[int, int],
    x0_dtype_size: int,
    x1_shape: tuple[int, int],
    x1_dtype_size: int,
) -> int:
    """Workspace estimate from shape products and per-input element sizes."""
    a = x0_shape[0] * x0_shape[1] * x0_dtype_size
    b = x1_shape[0] * x1_shape[1] * x1_dtype_size
    return int(a + b)


def calc_workspace_variadic(
    x_shape: tuple[int, ...],
    x_dtype_size: int,
) -> int:
    """Workspace scales with rank and dtype element size (dynamic rank)."""
    return int(len(x_shape) * x_dtype_size)
