import pto


def test_tile_shape_set_vec_tiles_shape_2d():
    expected = (8, 16)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_3d():
    expected = (1, 2, 3)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_4d():
    expected = (1, 2, 3, 8)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected
