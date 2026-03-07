import pypto
import torch
import os

def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, aone otherwise.
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
def axis_out_of_range_example(
    x: pypto.Tensor((4, 4), pypto.DT_FP32),
) -> pypto.Tensor((4,), pypto.DT_FP32):
    """
    错误示例：x 形状为 [4, 4]，但尝试访问轴 2
    这会触发 TENSOR_AXIS_OUT_OF_RANGE (0x12002)
    """
    pypto.set_vec_tile_shapes(4, 4)
    out = pypto.Tensor((4,), pypto.DT_FP32)
    out = pypto.sum(x, dim=2, keepdim=False)
    return out

def test_tensor_axis_out_of_range(device_id=None, run_mode="npu"):
    """
    测试 TENSOR_AXIS_OUT_OF_RANGE 错误码
    
    测试场景：
    错误示例：在 reduce 操作中使用了超出范围的轴参数
    - Tensor 形状为 [4, 4]，有效轴为 0 和 1
    - 尝试使用 axis=2，这会触发 TENSOR_AXIS_OUT_OF_RANGE
    """
    print("=" * 60)
    print("测试 TENSOR_AXIS_OUT_OF_RANGE (0x12002)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # 测试数据：创建形状为 [4, 4] 的 tensor
    x_data = torch.randn(4, 4, dtype=torch.float32, device=device)
    
    print(f"\n输入 x shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"Tensor 维度: {len(x_data.shape)}")
    print(f"有效轴范围: 0 到 {len(x_data.shape) - 1}")
    print(f"尝试访问的轴: 2（超出范围）")
    
    # 测试: 错误示例 - 预期会触发 TENSOR_AXIS_OUT_OF_RANGE
    print("\n" + "-" * 60)
    print("测试: 错误示例（轴超出范围）")
    print("-" * 60)
    try:
        result = axis_out_of_range_example(x_data)
        print("错误：预期应该触发 TENSOR_AXIS_OUT_OF_RANGE，但没有报错")
        print(f"结果 shape: {result.shape}")
    except Exception as e:
        print(f"✓ 成功捕获错误: {type(e).__name__}")
        error_msg = str(e)
        # 只打印错误信息的第一行，避免堆栈信息过长
        error_lines = error_msg.split('\n')
        print(f"错误信息: {error_lines[0]}")
        
        # 检查是否触发了预期的错误
        if "TENSOR_AXIS_OUT_OF_RANGE" in error_msg or "0x12002" in error_msg:
            print("✓ 确认触发了 TENSOR_AXIS_OUT_OF_RANGE (0x12002)")
        elif "axis" in error_msg.lower() and ("out of range" in error_msg.lower() or "out_of_range" in error_msg.lower()):
            print("✓ 触发了轴超出范围相关的错误")
        elif "axis" in error_msg.lower():
            print("✓ 触发了轴相关的错误（与 TENSOR_AXIS_OUT_OF_RANGE 相关）")
        else:
            print(f"注意：触发了其他错误，请检查: {error_lines[0]}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TENSOR_AXIS_OUT_OF_RANGE")
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
    
    test_tensor_axis_out_of_range(device_id, args.run_mode)