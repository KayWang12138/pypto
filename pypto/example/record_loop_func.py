import pto
from utils import dyn_function, loop_function


def init_tensors():
    dtype = pto.DataType.DT_FP32
    shape = (128, 128)
    a = pto.tensor(dtype, shape, "a")
    b = pto.tensor(dtype, shape, "b")
    c = pto.tensor(dtype, shape, "c")
    return a, b, c


def main():
    a, b, c = init_tensors()
    with dyn_function("main", [a, b], [c]):

        pto.set_vec_tile_shapes(16, 16)
        c.move(pto.add(a, b))

        loop_range = pto.loop_range(10)

        with loop_function(
            "Dynamic",
            "k",
            loop_range,
        ) as rlf:
            for _ in rlf:
                c.move(pto.add(a, a))


if __name__ == "__main__":
    main()
