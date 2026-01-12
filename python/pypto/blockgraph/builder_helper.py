from contextlib import contextmanager

from pypto.pypto_impl import ir


class BlockBuilderHelper:
    def __init__(self, builder, ctx):
        self.builder = builder
        self.ctx = ctx

    # ===== Scope Management (Context Managers) =====

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

    # ===== Control Flow Statements =====

    def ForNode(self, var, start, end, step, **kwargs):
        fs = self.builder.create_for(self.ctx, var, start, end, step)
        if kwargs:
            props = fs.properties()
            for key, value in kwargs.items():
                props[key] = str(value)
        return fs

    def IfNode(self, cond):
        return self.builder.create_if(self.ctx, cond)

    def exit_if(self, if_node):
        return self.builder.exit_if(self.ctx, if_node)

    # ===== Function Creation =====

    def create_function(self, name, kind, sig):
        return self.builder.create_function(name, kind, sig)

    def create_return(self, values):
        return self.builder.create_return(self.ctx, values)

    # ===== Value Creation =====

    def Scalar(self, dtype, name):
        return self.builder.create_scalar(self.ctx, dtype, name)

    def Const(self, value, name):
        return self.builder.create_const(self.ctx, value, name)

    def Tile(self, shape, dtype, name):
        return self.builder.create_tile(self.ctx, shape, dtype, name)

    # ===== Operations =====

    def adds(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def muls(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_MULS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op
