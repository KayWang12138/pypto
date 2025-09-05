"""
Run:
    GLOBAL_LOG_LEVEL=1 python example/matrix_op.py | tee run_vadd_py.log

Confirm same output as `vector_add` in `cpp_reference`
"""

import pto
from utils import pto_function

def matrix_matmul():
    dtype = pto.DataType.DT_FP32
    a = pto.tensor(dtype, (32, 64), "A")
    b = pto.tensor(dtype, (64, 32), "B")
    c = None

    with pto_function("MATMUL", a, b):
        pto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        c = pto.matmul(dtype, a, b)
        d = pto.matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    print(c.get_shape())
    assert isinstance(d, pto.tensor)
    print(d.get_shape())

if __name__ == "__main__":
    matrix_matmul()
