import pto
import pytest

from contextlib import contextmanager


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


@pytest.mark.skip(
    reason="RuntimeError: ASSERTION FAILED: currentFunctionPtr_->IsGraphType(GraphType::LEAF_GRAPH)"
)
def test_pybind_context_manager():
    dtype = pto.DataType.DT_FP16
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_a")

    c = None
    with pto_function("fnc_name", a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = pto.add(a, b)

    assert isinstance(c, pto.tensor)
