import pto


def test_function_type():
    # Make sure all data types are defined
    assert isinstance(pto.function_type.EAGER, pto.function_type)
    assert isinstance(pto.function_type.STATIC, pto.function_type)
    assert isinstance(pto.function_type.DYNAMIC, pto.function_type)
    assert isinstance(pto.function_type.DYNAMIC_LOOP, pto.function_type)
    assert isinstance(pto.function_type.DYNAMIC_LOOP_PATH, pto.function_type)
    assert isinstance(pto.function_type.INVALID, pto.function_type)
    assert isinstance(pto.function_type.MAX, pto.function_type)
