import textwrap
import ast
from pypto.ir.ast_transform import FunctionAstMutator


def test_if_transformation():
    raw_source = """
    if i:
        res = pto.block.Tile(shape, dtype, "name")
    else:
        res = None
    """
    source = textwrap.dedent(raw_source)
    tree = ast.parse(source)
    transformer = FunctionAstMutator()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_source = ast.unparse(new_tree)

    print(new_source)

    assert "builder.create_if(ctx, i)" in new_source
    assert "with if_then_scope(builder, ctx, ifs):" in new_source
    assert "builder.exit_if(ctx, ifs)" in new_source

if __name__ == "__main__":
    test_if_transformation()
