import textwrap
import ast
from pypto.ir.ast_transform import FunctionAstMutator


# TODO: test multi-level nested if and for
# TODO: test compound expressions like `a + b * c`


def transform_source(raw_source, print_comparison=True):
    """Helper function to transform source code using FunctionAstMutator.
    
    Args:
        raw_source: The raw source code string to transform
        print_comparison: If True, print raw_source vs new_source comparison
    
    Returns:
        The transformed source code string
    """
    source = textwrap.dedent(raw_source)
    tree = ast.parse(source)
    transformer = FunctionAstMutator()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_source = ast.unparse(new_tree)
    
    if print_comparison:
        print("\n" + "=" * 80)
        print("RAW SOURCE:")
        print("-" * 80)
        print(source)
        print("\nTRANSFORMED SOURCE:")
        print("-" * 80)
        print(new_source)
        print("=" * 80 + "\n")
    
    return new_source

def test_if_transformation():
    print("\n[Test: If Transformation]")
    raw_source = """
    if i:
        res = pto.block.Tile(shape, dtype, "name")
    else:
        res = None
    """
    new_source = transform_source(raw_source)

    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "builder.exit_if(ctx, ifs_0)" in new_source


def test_binary_ops_scalar_tile():
    """Test Scalar-Tile binary operations: OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS"""
    # Test addition
    print("\n[Test: Scalar-Tile Addition]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = res + scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_ADDS" in new_source
    assert "create_binary_scalar_op" in new_source
    
    # Test subtraction
    print("\n[Test: Scalar-Tile Subtraction]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = res - scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_SUBS" in new_source
    assert "create_binary_scalar_op" in new_source
    
    # Test multiplication
    print("\n[Test: Scalar-Tile Multiplication]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = res * scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MULS" in new_source
    assert "create_binary_scalar_op" in new_source
    
    # Test division
    print("\n[Test: Scalar-Tile Division]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = res / scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_DIVS" in new_source
    assert "create_binary_scalar_op" in new_source


def test_binary_ops_tile_tile():
    """Test Tile-Tile binary operations: OP_ADD, OP_SUB, OP_MUL, OP_DIV"""
    # Test addition
    print("\n[Test: Tile-Tile Addition]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = a + b
    """
    new_source = transform_source(raw_source)
    
    assert "OP_ADD" in new_source
    assert "create_binary_op" in new_source
    
    # Test subtraction
    print("\n[Test: Tile-Tile Subtraction]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = a - b
    """
    new_source = transform_source(raw_source)
    
    assert "OP_SUB" in new_source
    assert "create_binary_op" in new_source
    
    # Test multiplication
    print("\n[Test: Tile-Tile Multiplication]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = a * b
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MUL" in new_source
    assert "create_binary_op" in new_source
    
    # Test division
    print("\n[Test: Tile-Tile Division]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = a / b
    """
    new_source = transform_source(raw_source)
    
    assert "OP_DIV" in new_source
    assert "create_binary_op" in new_source


def test_augassign_ops():
    """Test augmented assignment operations: +=, -=, *=, /="""
    # Test +=
    print("\n[Test: Augmented Assignment +=]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res += scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_ADDS" in new_source
    assert "create_binary_scalar_op" in new_source
    
    # Test -=
    print("\n[Test: Augmented Assignment -=]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res -= scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_SUBS" in new_source
    
    # Test *=
    print("\n[Test: Augmented Assignment *=]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res *= scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MULS" in new_source
    
    # Test /=
    print("\n[Test: Augmented Assignment /=]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res /= scale
    """
    new_source = transform_source(raw_source)
    
    assert "OP_DIVS" in new_source


def test_min_max_scalar_tile():
    """Test min/max functions for Scalar-Tile operations: OP_MINS, OP_MAXS"""
    # Test max
    print("\n[Test: Scalar-Tile max()]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = max(res, scale)
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MAXS" in new_source
    assert "create_binary_scalar_op" in new_source
    
    # Test min
    print("\n[Test: Scalar-Tile min()]")
    raw_source = """
    res = pto.block.Tile(shape, dtype, "name")
    res = min(res, scale)
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MINS" in new_source
    assert "create_binary_scalar_op" in new_source


def test_min_max_tile_tile():
    """Test min/max functions for Tile-Tile operations: OP_MIN, OP_MAX"""
    # Test max
    print("\n[Test: Tile-Tile max()]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = max(a, b)
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MAX" in new_source
    assert "create_binary_op" in new_source
    
    # Test min
    print("\n[Test: Tile-Tile min()]")
    raw_source = """
    a = pto.block.Tile(shape, dtype, "a")
    b = pto.block.Tile(shape, dtype, "b")
    res = min(a, b)
    """
    new_source = transform_source(raw_source)
    
    assert "OP_MIN" in new_source
    assert "create_binary_op" in new_source


def test_nested_if_statements():
    """Test nested if statements (if inside if)"""
    print("\n[Test: Nested If Statements]")
    raw_source = """
    if i:
        res1 = pto.block.Tile(shape, dtype, "name1")
        if j:
            res2 = pto.block.Tile(shape, dtype, "name2")
        else:
            res3 = pto.block.Tile(shape, dtype, "name3")
    else:
        res4 = pto.block.Tile(shape, dtype, "name4")
    """
    new_source = transform_source(raw_source)
    
    # Check outer if
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    assert "builder.exit_if(ctx, ifs_0)" in new_source
    
    # Check inner if
    assert "ifs_1 = builder.create_if(ctx, j)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_1):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_1):" in new_source
    assert "builder.exit_if(ctx, ifs_1)" in new_source
    # Count occurrences to ensure nested if is properly transformed
    assert new_source.count("builder.create_if") >= 2


def test_nested_for_loops():
    """Test nested for loops (for inside for)"""
    print("\n[Test: Nested For Loops]")
    raw_source = """
    for i in pypto.block.loop(0, batch, step=1):
        res1 = pto.block.Tile(shape, dtype, "name1")
        for j in pypto.block.loop(0, batch, step=1):
            res2 = pto.block.Tile(shape, dtype, "name2")
    """
    new_source = transform_source(raw_source)
    
    # Check outer for
    assert "i = builder.create_scalar(ctx, ir.DataType.int32, 'i')" in new_source
    assert "fs_0 = builder.create_for(ctx, i," in new_source
    assert "with for_scope(builder, ctx, fs_0):" in new_source
    
    # Check inner for
    assert "j = builder.create_scalar(ctx, ir.DataType.int32, 'j')" in new_source
    assert "fs_1 = builder.create_for(ctx, j," in new_source
    assert "with for_scope(builder, ctx, fs_1):" in new_source
    # Should have multiple create_for calls
    assert new_source.count("builder.create_for") >= 2


def test_if_inside_for():
    """Test if statement inside for loop"""
    print("\n[Test: If Inside For Loop]")
    raw_source = """
    for i in pypto.block.loop(0, batch, step=1):
        res1 = pto.block.Tile(shape, dtype, "name1")
        if i:
            res2 = pto.block.Tile(shape, dtype, "name2")
        else:
            res3 = pto.block.Tile(shape, dtype, "name3")
    """
    new_source = transform_source(raw_source)
    
    # Check for loop
    assert "builder.create_for" in new_source
    assert "with for_scope(builder, ctx, fs_0):" in new_source
    
    # Check if statement inside for
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    assert "builder.exit_if(ctx, ifs_0)" in new_source


def test_for_inside_if():
    """Test for loop inside if statement"""
    print("\n[Test: For Loop Inside If Statement]")
    raw_source = """
    if i:
        for j in pypto.block.loop(0, batch, step=1):
            res1 = pto.block.Tile(shape, dtype, "name1")
    else:
        for k in pypto.block.loop(0, batch, step=1):
            res2 = pto.block.Tile(shape, dtype, "name2")
    """
    new_source = transform_source(raw_source)
    
    # Check if statement
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    
    # Check for loops inside if branches
    assert "j = builder.create_scalar(ctx, ir.DataType.int32, 'j')" in new_source
    assert "k = builder.create_scalar(ctx, ir.DataType.int32, 'k')" in new_source
    assert "fs_0 = builder.create_for(ctx, j," in new_source
    assert "fs_1 = builder.create_for(ctx, k," in new_source
    assert new_source.count("builder.create_for") >= 2


def test_multiple_nested_levels():
    """Test multiple levels of nesting (if inside for inside for)"""
    print("\n[Test: Multiple Nested Levels]")
    raw_source = """
    for i in pypto.block.loop(0, batch, step=1):
        for j in pypto.block.loop(0, batch, step=1):
            if i:
                res1 = pto.block.Tile(shape, dtype, "name1")
            else:
                res2 = pto.block.Tile(shape, dtype, "name2")
    """
    new_source = transform_source(raw_source)
    
    # Check outer for
    assert "i = builder.create_scalar(ctx, ir.DataType.int32, 'i')" in new_source
    assert "fs_0 = builder.create_for(ctx, i," in new_source
    
    # Check inner for
    assert "j = builder.create_scalar(ctx, ir.DataType.int32, 'j')" in new_source
    assert "fs_1 = builder.create_for(ctx, j," in new_source
    
    # Check if inside nested for
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    
    # Should have multiple for scopes
    assert "with for_scope(builder, ctx, fs_0):" in new_source
    assert "with for_scope(builder, ctx, fs_1):" in new_source


def test_nested_if_else_with_for():
    """Test nested if-else with for loops in both branches"""
    print("\n[Test: Nested If-Else With For Loops]")
    raw_source = """
    if i:
        for j in pypto.block.loop(0, batch, step=1):
            res1 = pto.block.Tile(shape, dtype, "name1")
            if j:
                res2 = pto.block.Tile(shape, dtype, "name2")
    else:
        for k in pypto.block.loop(0, batch, step=1):
            res3 = pto.block.Tile(shape, dtype, "name3")
    """
    new_source = transform_source(raw_source)
    
    # Check outer if
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    
    # Check for loops in both branches
    assert "j = builder.create_scalar(ctx, ir.DataType.int32, 'j')" in new_source
    assert "k = builder.create_scalar(ctx, ir.DataType.int32, 'k')" in new_source
    assert "fs_0 = builder.create_for(ctx, j," in new_source
    assert "fs_1 = builder.create_for(ctx, k," in new_source
    
    # Check inner if inside for in then branch
    assert "ifs_1 = builder.create_if(ctx, j)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_1):" in new_source
    
    # Should have multiple create_for and create_if calls
    assert new_source.count("builder.create_for") >= 2
    assert new_source.count("builder.create_if") >= 2


def test_for_with_nested_if_else():
    """Test for loop with nested if-else inside"""
    print("\n[Test: For Loop With Nested If-Else]")
    raw_source = """
    for i in pypto.block.loop(0, batch, step=1, unroll="4"):
        res1 = pto.block.Tile(shape, dtype, "name1")
        if i:
            res2 = pto.block.Tile(shape, dtype, "name2")
            if j:
                res3 = pto.block.Tile(shape, dtype, "name3")
            else:
                res4 = pto.block.Tile(shape, dtype, "name4")
        else:
            res5 = pto.block.Tile(shape, dtype, "name5")
    """
    new_source = transform_source(raw_source)
    
    # Check for loop
    assert "fs_0 = builder.create_for(ctx, i," in new_source
    assert "unroll" in new_source
    
    # Check outer if
    assert "ifs_0 = builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_0):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_0):" in new_source
    
    # Check inner if
    assert "ifs_1 = builder.create_if(ctx, j)" in new_source
    assert "with if_then_scope(builder, ctx, ifs_1):" in new_source
    assert "with if_else_scope(builder, ctx, ifs_1):" in new_source
    
    # Should have proper nesting structure
    assert new_source.count("builder.create_if") >= 2
    assert new_source.count("with if_then_scope") >= 2
    assert new_source.count("with if_else_scope") >= 2


if __name__ == "__main__":
    test_if_transformation()
    test_binary_ops_scalar_tile()
    test_binary_ops_tile_tile()
    test_augassign_ops()
    test_min_max_scalar_tile()
    test_min_max_tile_tile()
    test_nested_if_statements()
    test_nested_for_loops()
    test_if_inside_for()
    test_for_inside_if()
    test_multiple_nested_levels()
    test_nested_if_else_with_for()
    test_for_with_nested_if_else()
    print("All tests passed!")
