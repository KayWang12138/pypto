import inspect
import re

__all__: tuple[str, ...] = ()


def _unwrap_decorated_func_source(source: str) -> str:
    """Return the def ... body of a function, stripping decorator lines."""
    return source[source.find("def "):]  # a bit ugly, check if there're better options


def _unwrap_decorated_func_name(name: str) -> str:
    """Return the actual function name when __name__ may include decorator info."""
    return name.split()[0]


def _get_renamed_func_source(func, new_func_name: str) -> str:
    """Return function source with the function name replaced by new_func_name."""
    source = _unwrap_decorated_func_source(inspect.getsource(func))
    orig_func_name = _unwrap_decorated_func_name(func.__name__)
    return source.replace(orig_func_name, new_func_name, 1)


def _snake_case_to_camel_case(name: str) -> str:
    """Convert a snake_case string to a camelCase string."""
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _camel_case_to_snake_case(name: str) -> str:
    """Convert PascalCase or camelCase (e.g. ``Add``, ``AddPyptoCustomOp``) to snake_case for paths."""
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.lower()