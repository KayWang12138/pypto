import pto


def test_graph_type():
    # Make sure all graph types are defined
    assert isinstance(pto.graph_type.TENSOR_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.TILE_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.ROOT_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.LEAF_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.LEAF_VF_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.INVALID, pto.graph_type)
