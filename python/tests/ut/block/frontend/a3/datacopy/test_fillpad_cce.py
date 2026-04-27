# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime validation for fillpad-family manual ops on CCE.

CCE manual.load lowers to a bare ``TLOAD(tile, tensor)``. For ND row-major vec
tiles, that means we must first load the full physical tile and only then
narrow the runtime valid-shape before the fillpad-family operation.
"""

import torch
import torch_npu

import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


def _print_case_header(name: str) -> None:
    print(f"------------{name}--------------", flush=True)


def _print_tensor_block(name: str, tensor: torch.Tensor) -> None:
    _print_case_header(name)
    print(tensor.shape, tensor.dtype)
    print(tensor)


def _make_fillpad_inputs(device: str, output_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.full((8, 8), -99, device=device, dtype=torch.int32)
    x[:5, :7] = torch.arange(35, device=device, dtype=torch.int32).reshape(5, 7)
    z = torch.empty(output_shape, device=device, dtype=torch.int32)
    z_ref = torch.zeros(output_shape, device=device, dtype=torch.int32)
    z_ref[:5, :7] = x[:5, :7]
    return x, z, z_ref


def _run_fillpad_cce_case(test_name: str, kernel, output_shape: tuple[int, int], output_label: str, device: str) -> None:
    compiled_lib = fe.compile(kernel, arch="a3", codegen_mode="cce")
    if compiled_lib is None:
        raise RuntimeError(f"compile failed for {kernel.__name__}")
    print("compiled lib path:", compiled_lib.lib_path)

    torch.npu.set_device(device)
    x, z, z_ref = _make_fillpad_inputs(device, output_shape)

    fe.launch(None, 1, compiled_lib, x, z)
    torch.npu.synchronize()

    _print_tensor_block(f"{test_name}_{output_label}", z)
    _print_tensor_block(f"{test_name}_golden", z_ref)

    torch.testing.assert_close(z, z_ref)
    print("result equal!")


@fe.kernel
def fillpad_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.INT32],
    z: pl.Tensor[[8, 8], pl.INT32],
) -> pl.Tensor[[8, 8], pl.INT32]:
    src_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=plm.TilePad.zero,
    )
    src = plm.make_tile(src_type, addr=0x0000, size=256)
    dst = plm.make_tile(dst_type, addr=0x0100, size=256)

    with pl.section_vector():
        plm.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        plm.set_validshape(src, 5, 7)
        plm.dump_tile(src)

        plm.fillpad(dst, src)

        plm.dump_tile(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, dst, [0, 0])
        pl.system.bar_all()

    return z


@fe.kernel
def fillpad_inplace_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.INT32],
    z: pl.Tensor[[8, 8], pl.INT32],
) -> pl.Tensor[[8, 8], pl.INT32]:
    src_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=plm.TilePad.zero,
        valid_shape=[-1, -1],
    )
    dst_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=plm.TilePad.zero,
    )
    src = plm.make_tile(src_type, addr=0x0000, size=256)
    dst = plm.make_tile(dst_type, addr=0x0000, size=256)

    with pl.section_vector():
        plm.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        plm.set_validshape(src, 5, 7)
        plm.dump_tile(src)

        plm.fillpad_inplace(dst, src)

        plm.dump_tile(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, dst, [0, 0])
        pl.system.bar_all()

    return z


@fe.kernel
def fillpad_expand_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.INT32],
    z: pl.Tensor[[8, 16], pl.INT32],
) -> pl.Tensor[[8, 16], pl.INT32]:
    src_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst_type = plm.TileType(
        shape=[8, 16],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=plm.TilePad.zero,
    )
    src = plm.make_tile(src_type, addr=0x0000, size=256)
    dst = plm.make_tile(dst_type, addr=0x0100, size=512)

    with pl.section_vector():
        plm.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        plm.set_validshape(src, 5, 7)
        plm.dump_tile(src)

        plm.fillpad_expand(dst, src)

        plm.dump_tile(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, dst, [0, 0])
        pl.system.bar_all()

    return z


@fe.jit()
def test_fillpad_dynamic_cce():
    _print_case_header("test_fillpad_dynamic_cce")
    _run_fillpad_cce_case("fillpad_dynamic_cce", fillpad_dynamic_cce_kernel, (8, 8), "output", "npu:7")


@fe.jit()
def test_fillpad_inplace_dynamic_cce():
    _print_case_header("test_fillpad_inplace_dynamic_cce")
    _run_fillpad_cce_case(
        "fillpad_inplace_dynamic_cce",
        fillpad_inplace_dynamic_cce_kernel,
        (8, 8),
        "output",
        "npu:7",
    )


@fe.jit()
def test_fillpad_expand_dynamic_cce():
    _print_case_header("test_fillpad_expand_dynamic_cce")
    _run_fillpad_cce_case("fillpad_expand_dynamic_cce", fillpad_expand_dynamic_cce_kernel, (8, 16), "output", "npu:0")


if __name__ == "__main__":
    cases = [test_fillpad_dynamic_cce, test_fillpad_inplace_dynamic_cce, test_fillpad_expand_dynamic_cce]
    for case in cases:
        case()
    print("\nAll tests passed!")
