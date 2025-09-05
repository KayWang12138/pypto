import pto
import pytest


def test_begin_add_end_function():
    dtype = pto.DataType.DT_FP16
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = None

    fnc_name = "add_fnc"
    pto.begin_function(fnc_name, a, b)
    pto.set_vec_tile_shapes(8, 8)
    c = pto.add(a, b)
    pto.end_function(fnc_name, False)

    print(pto.dump())
    # Replace True with False to see graph
    assert isinstance(c, pto.tensor)


@pytest.mark.skip(reason="add inplace fails in pytests")
def test_begin_inplaceadd_end_function():
    dtype = pto.DataType.DT_FP16
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = None

    fnc_name = "add_inplace_fnc"
    pto.begin_function(fnc_name, a, b)
    pto.set_vec_tile_shapes(8, 8)
    c = a + b
    pto.end_function(fnc_name, False)

    print(pto.dump())
    assert isinstance(c, pto.tensor)


def test_empty_begin_end_function():
    dtype = pto.DataType.DT_FP16
    a = pto.tensor(dtype, (8, 8), "tensor_a")
    fnc_name = "name"
    pto.begin_function(fnc_name, a)
    pto.set_vec_tile_shapes(8, 8)
    pto.end_function(fnc_name, False)

    assert True
