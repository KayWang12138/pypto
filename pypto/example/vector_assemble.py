"""
Run:
    GLOBAL_LOG_LEVEL=1 python example/vector_assemble.py | tee run_vassemble_py.log

Confirm same output as `vector_assemble` in `cpp_reference`
"""

import pto
from utils import pto_function


def vector_assemble():
    dtype = pto.DataType.DT_FP32
    shape = (128, 128)
    offsets = (0, 0)
    tensor = pto.tensor(dtype, shape, "tensor")
    c = None

    assemble_input = [(tensor, offsets)]

    with pto_function("ASSEMBLE", tensor):
        pto.set_vec_tile_shapes(128, 128)
        c = pto.assemble(assemble_input)

    print(pto.dump())
    assert isinstance(c, pto.tensor)


if __name__ == "__main__":
    vector_assemble()
