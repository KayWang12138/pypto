import inspect

def _unwrap_decorated_func_source(source: str) -> str:
    return source[source.find("def "):] # a bit ugly, check if there're better options

def _unwrap_decorated_func_name(name: str) -> str:
    return name.split()[0]

def _get_renamed_func_source(func, new_func_name: str) -> str:
    source = _unwrap_decorated_func_source(inspect.getsource(func))
    orig_func_name = _unwrap_decorated_func_name(func.__name__)
    return source.replace(orig_func_name, new_func_name, 1)
