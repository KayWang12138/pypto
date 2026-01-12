from contextlib import contextmanager


class BlockBuilderHelper:
    def __init__(self, builder, ctx):
        self.builder = builder
        self.ctx = ctx

    @contextmanager
    def function_scope(self, func):
        self.builder.enter_function(self.ctx, func)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    @contextmanager
    def for_scope(self, loop_node):
        self.builder.enter_for(self.ctx, loop_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()
            self.builder.exit_for(self.ctx, loop_node)

    @contextmanager
    def if_then_scope(self, if_node):
        self.builder.enter_if_then(self.ctx, if_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    @contextmanager
    def if_else_scope(self, if_node):
        self.builder.enter_if_else(self.ctx, if_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    # NOTE: a pair of `if`` and `else` only call `builder.exit_if` once,
    #       thus do not wrap `exit`, like `for_scope`
