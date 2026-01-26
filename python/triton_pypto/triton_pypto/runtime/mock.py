from ..language import operations as l


def mock():
    import triton.language as tl
    mocks = (
        l.abs,
        l.arange,
        l.cdiv,
        l.clamp,
        l.dot,
        l.exp,
        l.expand_dims,
        l.full,
        l.load,
        l.log,
        l.make_block_ptr,
        l.max,
        l.maximum,
        l.minimum,
        l.num_programs,
        l.permute,
        l.program_id,
        l.range,
        l.ravel,
        l.reshape,
        l.rsqrt,
        l.sigmoid,
        l.sqrt,
        l.static_range,
        l.store,
        l.sum,
        l.trans,
        l.where,
        l.zeros,
    )
    for m in mocks:
        setattr(tl, m.__name__, m)
