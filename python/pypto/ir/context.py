from contextlib import contextmanager

@contextmanager
def function_scope(builder, ctx, func):
    builder.enter_function(ctx, func)
    try:
        yield
    finally:
        ctx.pop_scope()

@contextmanager
def for_scope(builder, ctx, loop_node):
    builder.enter_for(ctx, loop_node)
    try:
        yield
    finally:
        ctx.pop_scope()
        builder.exit_for(ctx, loop_node)

@contextmanager
def if_then_scope(builder, ctx, if_node):
    builder.enter_if_then(ctx, if_node)
    try:
        yield
    finally:
        ctx.pop_scope()

@contextmanager
def if_else_scope(builder, ctx, if_node):
    builder.enter_if_else(ctx, if_node)
    try:
        yield
    finally:
        ctx.pop_scope()
