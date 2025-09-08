import pto


def init_tensors():
    dtype = pto.DataType.DT_FP32
    shape = (128, 128)
    a = pto.tensor(dtype, shape, "a")
    b = pto.tensor(dtype, shape, "b")
    c = None
    return a, b, c


def main():
    a, b, c = init_tensors()

    recorder = pto.record_func("main", pto.function_type.STATIC, [a, b])
    pto.set_vec_tile_shapes(16, 16)
    c = pto.add(a, b)
    del recorder

    print(pto.dump())


if __name__ == "__main__":
    main()
