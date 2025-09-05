import pto


def test_node_types():
    # Make sure all node types are defined
    assert isinstance(pto.NodeType.LOCAL, pto.NodeType)
    assert isinstance(pto.NodeType.INCAST, pto.NodeType)
    assert isinstance(pto.NodeType.OUTCAST, pto.NodeType)
