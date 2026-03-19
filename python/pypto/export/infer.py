from .pypto_op import _FUNC_NAME__INFER_SHAPE, _FUNC_NAME__CALC_WORKSPACE

__all__ = ("register_infer_shape_fn", "register_calc_workspace_fn")

_infer_funcs = {}


def register_infer_shape_fn(source: str, namespace=""):
    """Execute source in a namespace and return the infer_shape function."""
    funcs = _infer_funcs.setdefault(namespace, {})
    exec(source, funcs)
    return funcs[_FUNC_NAME__INFER_SHAPE]


def register_calc_workspace_fn(source: str, namespace=""):
    """Execute source in a namespace and return the calc_workspace function."""
    funcs = _infer_funcs.setdefault(namespace, {})
    exec(source, funcs)
    return funcs[_FUNC_NAME__CALC_WORKSPACE]
