import pto
from contextlib import contextmanager
from typing import List
import inspect


@contextmanager
def dyn_function(
    name: str, in_tensors: List[pto.tensor], out_tensors: List[pto.tensor]
) -> pto.record_func:
    record_func = pto.record_func(
        name, pto.function_type.DYNAMIC, in_tensors, out_tensors, []
    )
    print(f"Entering DYNAMIC function: {name}")
    try:
        yield record_func
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        del record_func
        print(f"Exiting DYNAMIC function: {name}")


@contextmanager
def loop_function(
    name: str, loop_name: str, loop_range: pto.loop_range
) -> pto.record_loop_func:
    # TODO(anastasios): make it an input parameter
    rlf = None
    empty_set_ints = set()
    print(f"Entering LOOP function: {name}")
    try:
        rlf = pto.record_loop_func(
            name,
            pto.function_type.DYNAMIC_LOOP,
            loop_name,
            loop_range,
            empty_set_ints,
            False,
        )
        yield rlf
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        del rlf
        print(f"Exiting LOOP function: {name}")


# TODO: move common utils into `pto` Python package
@contextmanager
def pto_function(
    name: str, graph_type: pto.graph_type, func_type: pto.function_type, *args
):
    print(f"Entering context: {name}")
    try:
        yield pto.begin_function(name, graph_type, func_type, *args)
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        # TODO(anastasios): make false input param.
        pto.end_function(name, False)
        print(f"Exiting context: {name}")


def record_if_branch(scalar: pto.symbolic_scalar):
    frame = inspect.currentframe().f_back
    return pto.record_if_branch(scalar, frame.f_code.co_filename, frame.f_lineno)
