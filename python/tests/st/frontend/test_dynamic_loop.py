# pylint: disable=missing-docstring
import pypto

@pypto.frontend.jit()
def dynamic_loop(
    a: pypto.Tensor((128, 128), pypto.DT_FP32),
) -> pypto.Tensor((128, 128), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(64, 64)
    b = pypto.tensor((128, 128), pypto.DT_FP32)
    for k in range(10):
        b[:] = pypto.add(a, a)
        if k < 2:
            b[:] = pypto.add(b, a)
        else:
            b[:] = pypto.sub(b, a)

        if k < 5:
            b[:] = pypto.mul(b, a)
        else:
            b[:] = pypto.div(b, a)
        b[:] = pypto.sub(b, a)
    return b


print(pypto.dump())
