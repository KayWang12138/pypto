import pto


def test_tile_shape_reset():
    tile_shape = pto.tile_shape()
    tile_shape.reset()

    assert tile_shape is not None


def test_tile_shape_dump():
    tile_shape = pto.tile_shape()
    tile_shape.dump(False)
    tile_shape.dump(True)

    assert tile_shape is not None


def test_tile_shape_specify_static_rank_id():
    tile_shape = pto.tile_shape()
    rank_id = 1
    tile_shape.specify_static_rank_id(rank_id)

    assert tile_shape is not None


def test_tile_shape_set_vec_tiles_shape_2d():
    tile_shape = pto.tile_shape()
    expected = (8, 16)
    tile_shape.set_vec_tile_shapes(*expected)
    actual = tile_shape.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_3d():
    tile_shape = pto.tile_shape()
    expected = (1, 2, 3)
    tile_shape.set_vec_tile_shapes(*expected)
    actual = tile_shape.get_vec_tile_shapes()
    assert tuple(actual) == expected
