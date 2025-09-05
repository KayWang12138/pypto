import pto
from contextlib import contextmanager


# TODO: move common utils into `pto` Python package
@contextmanager
def pto_function(name: str, *args):
    print(f"Entering context: {name}")
    try:
        yield pto.begin_function(name, *args)
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        # TODO(anastasios): make false input param.
        pto.end_function(name, False)
        print(f"Exiting context: {name}")
