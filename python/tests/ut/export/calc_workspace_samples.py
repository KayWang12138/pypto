from __future__ import annotations


def calc_workspace_fixed(
    x0_shape: tuple[int, int],
    x1_shape: tuple[int, int],
) -> int:
    """Workspace based on product of first input dims."""
    return int(x0_shape[0] * x0_shape[1])


def calc_workspace_variadic(
    x_shape: tuple[int, ...],
) -> int:
    """Workspace based on rank only (supports dynamic rank)."""
    return int(len(x_shape))

