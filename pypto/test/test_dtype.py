import pto


def test_mem_type():
    # Make sure all data types are defined
    assert isinstance(pto.DataType.DT_INT4, pto.DataType)
    assert isinstance(pto.DataType.DT_INT8, pto.DataType)
    assert isinstance(pto.DataType.DT_INT16, pto.DataType)
    assert isinstance(pto.DataType.DT_INT32, pto.DataType)
    assert isinstance(pto.DataType.DT_INT64, pto.DataType)
    assert isinstance(pto.DataType.DT_FP8, pto.DataType)
    assert isinstance(pto.DataType.DT_FP16, pto.DataType)
    assert isinstance(pto.DataType.DT_FP32, pto.DataType)
    assert isinstance(pto.DataType.DT_BF16, pto.DataType)
    assert isinstance(pto.DataType.DT_HF4, pto.DataType)
    assert isinstance(pto.DataType.DT_HF8, pto.DataType)
    assert isinstance(pto.DataType.DT_UINT8, pto.DataType)
    assert isinstance(pto.DataType.DT_UINT16, pto.DataType)
    assert isinstance(pto.DataType.DT_UINT32, pto.DataType)
    assert isinstance(pto.DataType.DT_UINT64, pto.DataType)
    assert isinstance(pto.DataType.DT_BOOL, pto.DataType)
    assert isinstance(pto.DataType.DT_DOUBLE, pto.DataType)
    assert isinstance(pto.DataType.DT_BOTTOM, pto.DataType)
