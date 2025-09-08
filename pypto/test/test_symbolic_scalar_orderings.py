import pto


def test_symbolic_scalar_le():
    ten = pto.symbolic_scalar("scalar", 10)
    twenty = pto.symbolic_scalar("scalar", 20)

    assert ten <= twenty


def test_symbolic_scalar_lt():
    ten = pto.symbolic_scalar("scalar", 10)
    twenty = pto.symbolic_scalar("scalar", 20)

    assert ten < twenty


def test_symbolic_scalar_gt():
    ten = pto.symbolic_scalar("scalar", 10)
    twenty = pto.symbolic_scalar("scalar", 20)

    assert twenty > ten


def test_symbolic_scalar_ge():
    ten = pto.symbolic_scalar("scalar", 10)
    twenty = pto.symbolic_scalar("scalar", 20)

    assert twenty >= ten


def test_symbolic_scalar_ne():
    ten = pto.symbolic_scalar("scalar", 10)
    twenty = pto.symbolic_scalar("scalar", 20)

    assert twenty != ten


def test_symbolic_scalar_eq():
    ten = pto.symbolic_scalar("scalar", 10)
    another_ten = pto.symbolic_scalar("scalar", 10)

    assert ten == another_ten


def test_symbolic_scalar_comp_op():
    a = pto.symbolic_scalar(6)
    b = pto.symbolic_scalar(4)
    assert (a == b).concrete() == 0
    assert (a != b).concrete() == 1
    assert (a < b).concrete() == 0
    assert (a <= b).concrete() == 0
    assert (a > b).concrete() == 1
    assert (a >= b).concrete() == 1

    assert (a == 6).concrete() == 1
    assert (a != 6).concrete() == 0
    assert (a < 6).concrete() == 0
    assert (a <= 6).concrete() == 1
    assert (a > 6).concrete() == 0
    assert (a >= 6).concrete() == 1

    assert (6 == a).concrete() == 1
    assert (6 != a).concrete() == 0
    assert (6 < a).concrete() == 0
    assert (6 <= a).concrete() == 1
    assert (6 > a).concrete() == 0
    assert (6 >= a).concrete() == 1
