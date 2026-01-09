`builder_api_level_3.py` simplifies `builder_api_level_2.py` by parsing type hints build `ir.FunctionSignature`

`builder_api_level_2.py` simplifies `builder_api_level_1.py`, by the following ast-transforms on `create_function`:

1) Mapping native `for` statement

```python
# `batch = ir.Scalar(ir.DataType.int32, None, "batch")` passed from closure
for i in pypto.block.loop(0, batch, step=1, unroll="4"):
    (loop body using `i`)
```

to:
```python
# batch = ir.Scalar(ir.DataType.int32, None, "batch") passed from closure
i = builder.create_scalar(ctx, ir.DataType.int32, "i")
constant0 = builder.create_const(ctx, 0, "const_0")  # start
constant1 = builder.create_const(ctx, 1, "const_1")  # step

fs = builder.create_for(ctx, i, constant0, batch, constant1)
fs.properties()["unroll"] = "4"

with for_scope(builder, ctx, fs):
    (loop body using `i`)
```


2) Mapping native `if`/`else`

```python
if i:
    (if body)
else:
    (else body)
```

to:
```python
ifs = builder.create_if(ctx, i)

with if_then_scope(builder, ctx, ifs):
    (if body)

with if_else_scope(builder, ctx, ifs):
    (else body)

builder.exit_if(ctx, ifs)  # this auto-inserted by transform
```

NOTE: only apply transform when the `cond` in `if cond:` is `ir.Scalar`.
      When `cond` is normal Python bool, do not transform the expression.

3) Mapping tile computation

Maps:
```python
res_loop_x = pto.block.Tile(tile_shape, ir.DataType.float, "outputX")
pto.block.add(res_loop_x, scale1, out=res_loop_x)
```

or even simpler:
```python
res_loop_x = pto.block.Tile(tile_shape, ir.DataType.float, "outputX")
res_loop_x += scale1
```

to:
```python
res_loop_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputX")
add_op_x = builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x)
builder.emit(ctx, add_op_x)
```

Another example, from
```python
res_if_y = pto.block.Tile(tile_shape, ir.DataType.float, "outputY")
res_if_y = res_loop_y * scale2
```

to
```python
res_if_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
mul_op_y = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
builder.emit(ctx, mul_op_y)
```

4) Mapping return

to
```python
return constant0
```

```python
builder.create_return(ctx, [constant0])
```
