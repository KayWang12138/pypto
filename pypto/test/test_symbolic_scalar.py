import pto


def test_init_symbolic_scalar_no_args():
    scalar = pto.symbolic_scalar()

    assert scalar.concrete_valid() == False


def test_init_symbolic_scalar_value_arg():
    expected_value = 123
    scalar = pto.symbolic_scalar(expected_value)

    assert scalar.concrete_valid() == True
    assert scalar.concrete() == expected_value


def test_init_symbolic_scalar_name_value_args():
    expected_value = 123
    scalar = pto.symbolic_scalar("scalar", expected_value)

    assert scalar.concrete_valid() == True
    assert scalar.concrete() == expected_value


def test_init_symbolic_scalar_not_less_than():
    lower_bound = 100
    scalar = pto.symbolic_scalar("scalar", pto.not_less_than(lower_bound))

    assert scalar.concrete_valid() == False


def test_init_symbolic_scalar_not_greater_than():
    upper_bound = 200
    scalar = pto.symbolic_scalar("scalar", pto.not_greater_than(upper_bound))

    assert scalar.concrete_valid() == False


def test_init_symbolic_scalar_not_less_than_not_greater_than():
    lower_bound = 100
    upper_bound = 200
    scalar = pto.symbolic_scalar(
        "scalar", pto.not_less_than(lower_bound), pto.not_greater_than(upper_bound)
    )

    assert scalar.concrete_valid() == False


def test_symbolic_scalar_dump():
    scalar = pto.symbolic_scalar(10)
    dump_str = scalar.dump()
    dump_int = int(scalar)
    assert isinstance(dump_str, str)
    assert isinstance(dump_int, int)
    assert dump_str == "10"
    assert dump_int == 10


def test_symbolic_scalar_prop():
    scalar = pto.symbolic_scalar(10)
    assert scalar.is_symbol() == False
    assert scalar.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar.is_valid() == True
    assert scalar.concrete_valid() == True
    assert scalar.concrete() == 10

    scalar2 = pto.symbolic_scalar("s")
    assert scalar2.is_symbol() == True
    assert scalar2.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar.is_valid() == True
    assert scalar2.concrete_valid() == False

    scalar3 = scalar < 2
    assert isinstance(scalar3, pto.symbolic_scalar)
    assert scalar3.is_symbol() == False
    assert scalar3.is_expression() == False
    assert scalar3.is_immediate() == True
    assert scalar3.is_valid() == True
    assert scalar3.concrete_valid() == True
    assert scalar3.concrete() == 0

    scalar4 = scalar2 < 2
    assert isinstance(scalar4, pto.symbolic_scalar)
    assert scalar4.is_symbol() == False
    assert scalar4.is_expression() == True
    assert scalar4.is_immediate() == False
    assert scalar4.is_valid() == True
    assert scalar4.concrete_valid() == False


def test_symbolic_scalar_uniop():
    scalar = pto.symbolic_scalar(10)
    pos_s = +scalar
    neg_s = -scalar
    not_s = ~scalar
    assert isinstance(pos_s, pto.symbolic_scalar)
    assert isinstance(neg_s, pto.symbolic_scalar)
    assert isinstance(not_s, pto.symbolic_scalar)
    assert scalar.concrete() == 10
    assert pos_s.concrete() == 10
    assert neg_s.concrete() == -10
    assert not_s.concrete() == 0
