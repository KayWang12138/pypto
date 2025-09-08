import pto

from utils import dyn_function, loop_function, record_if_branch


# FIXME(anastasios): Fix after pypackage.Monkey patching for now
pto.dyn_function = dyn_function
pto.loop_function = loop_function
pto.cond = record_if_branch


def main():
    dtype = pto.DataType.DT_FP16
    shape = (128, 128)
    a = pto.tensor(dtype, shape, "PTO_TENSOR_a")
    b = pto.tensor(dtype, shape, "PTO_TENSOR_b")

    with pto.dyn_function("MAIN", [a], [b]):
        pto.set_vec_tile_shapes(64, 64)
        with pto.loop_function("Dynamic", "k", pto.loop_range(10)) as rlf:
            for k in rlf:
                b.move(pto.add(a, a))

                if pto.cond(k < 2):
                    b.move(pto.add(b, a))
                else:
                    b.move(pto.sub(b, a))

                if pto.cond(k < 5):
                    b.move(pto.mul(b, a))
                else:
                    b.move(pto.div(b, a))
                b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


if __name__ == "__main__":
    main()
