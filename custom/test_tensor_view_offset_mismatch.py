import pypto
import torch
import os

def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None

@pypto.frontend.jit
def view_offset_mismatch_example(
    x: pypto.Tensor((8, 8), pypto.DT_FP32),
    out: pypto.Tensor((4, 4), pypto.DT_FP32),
) -> None:
    """
    错误示例：x 形状为 [8, 8]，但偏移 [100, 100] 超出范围
    这会触发 TENSOR_VIEW_OFFSET_MISMATCH (0x12004)
    """
    pypto.set_vec_tile_shapes(8, 8)
    out[:] = pypto.view(x, [4, 4], [100, 100])
    return

def test_tensor_view_offset_mismatch(device_id=None, run_mode="npu"):
    """
    测试 TENSOR_VIEW_OFFSET_MISMATCH 错误码
    
    测试场景：
    错误示例：View 操作的偏移参数错误
    - 源 Tensor 形状为 [8, 8]
    - 视图形状为 [4, 4]
    - 偏移为 [10, 10]，超出源 Tensor 范围
    - 这会触发 TENSOR_VIEW_OFFSET_MISMATCH
    """
    print("=" * 60)
    print("测试 TENSOR_VIEW_OFFSET_MISMATCH (0x12004)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # 测试数据：创建形状为 [8, 8] 的 tensor
    x_data = torch.randn(8, 8, dtype=torch.float32, device=device)
    out_data = torch.zeros(4, 4, dtype=torch.float32, device=device)
    
    print(f"\n输入 x shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"源 Tensor 形状: {list(x_data.shape)}")
    print(f"视图形状: [4, 4]")
    print(f"尝试使用的偏移: [100, 100]")
    print(f"有效偏移范围: [0, 4]（非负数且 offset + view_shape <= input_shape）")
    print(f"偏移无效: [100, 100] 超出范围")
    
    # 测试: 错误示例 - 预期会触发 TENSOR_VIEW_OFFSET_MISMATCH
    print("\n" + "-" * 60)
    print("测试: 错误示例（视图偏移超出范围）")
    print("-" * 60)
    try:
        view_offset_mismatch_example(x_data, out_data)
        print("错误：预期应该触发 TENSOR_VIEW_OFFSET_MISMATCH，但没有报错")
        print(f"结果 shape: {out_data.shape}")
    except Exception as e:
        print(f"✓ 成功捕获错误: {type(e).__name__}")
        error_msg = str(e)
        # 只打印错误信息的第一行，避免堆栈信息过长
        error_lines = error_msg.split('\n')
        print(f"错误信息: {error_lines[0]}")
        
        # 检查是否触发了预期的错误
        if (any(keyword in error_msg for keyword in [
            "TENSOR_VIEW_OFFSET_MISMATCH", 
            "0x12004",
            "offset.*out",
            "offset.*range",
            "offset.*mismatch"
        ])):
            print("✓ 确认触发了 TENSOR_VIEW_OFFSET_MISMATCH (0x12004)")
        elif "offset" in error_msg.lower():
            print("✓ 触发了偏移相关的错误（与 TENSOR_VIEW_OFFSET_MISMATCH 相关）")
        else:
            print(f"注意：触发了其他错误，请检查: {error_lines[0]}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TENSOR_VIEW_OFFSET_MISMATCH")
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"],
        help='Run mode: npu or sim'
    )
    args = parser.parse_args()
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            exit(1)
        import torch_npu
        torch_npu.npu.set_device(device_id)
    
    test_tensor_view_offset_mismatch(device_id, args.run_mode)
