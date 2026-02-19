from .pypto_op import _FUNC_NAME__INFER_SHAPE, _FUNC_NAME__CALC_WORKSPACE

_infer_funcs = {}

def register_infer_shape_fn(source: str, namespace=""):
    funcs = _infer_funcs.setdefault(namespace, {})
    exec(source, funcs)
    return funcs[_FUNC_NAME__INFER_SHAPE]

def register_calc_workspace_fn(source: str, namespace=""):
    funcs = _infer_funcs.setdefault(namespace, {})
    exec(source, funcs)
    return funcs[_FUNC_NAME__CALC_WORKSPACE]
