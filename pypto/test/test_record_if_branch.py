import pto
from utils import dyn_function, loop_function, record_if_branch

# FIXME(anastasios): Fix after pypackage.Monkey patching for now
pto.dyn_function = dyn_function
pto.loop_function = loop_function
pto.cond = record_if_branch

def test_record_if_branch():
    dtype = pto.DataType.DT_FP16
    shape = (32, 32)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = pto.tensor(dtype, shape, "tensor_c")

    with pto.dyn_function("ADD_IF", [a, b], [c]):
        pto.set_vec_tile_shapes(8, 8)
        loop_range = pto.loop_range(2)
        with pto.loop_function(
            "LOOP",
            "k",
            loop_range,
        ) as rlf:
            for k in rlf:
                if pto.cond(k < 10):
                    c = pto.add(a, b)

    assert isinstance(c, pto.tensor)
