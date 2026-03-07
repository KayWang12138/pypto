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
def view_dimension_mismatch_example(
    x: pypto.Tensor((8, 8), pypto.DT_FP32),
) -> pypto.Tensor((8, 8), pypto.DT_FP32):
    """
    错误示例：x 形状为 [8, 8]，但视图形状为 [4, 4, 4]
    这会触发 TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)
    """
    pypto.set_vec_tile_shapes(8, 8)
    out = pypto.Tensor((8, 8), pypto.DT_FP32)
    view_tensor = pypto.view(x, [4, 4, 4], [0, 0, 0])
    out = pypto.mul(view_tensor, 2.0)
    return out

def test_tensor_view_dimension_mismatch(device_id=None, run_mode="npu"):
    """
    测试 TENSOR_VIEW_DIMENSION_MISMATCH 错误码
    
    测试场景：
    错误示例：View 操作的维度与源 Tensor 不匹配
    - 源 Tensor 形状为 [8, 8]（2维）
    - 视图形状为 [4, 4, 4]（3维）
    - 这会触发 TENSOR_VIEW_DIMENSION_MISMATCH
    """
    print("=" * 60)
    print("测试 TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # 测试数据：创建形状为 [8, 8] 的 tensor
    x_data = torch.randn(8, 8, dtype=torch.float32, device=device)
    
    print(f"\n输入 x shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"源 Tensor 维度: {len(x_data.shape)}")
    print(f"源 Tensor 形状: {list(x_data.shape)}")
    print(f"尝试创建视图形状: [4, 4, 4]（3维）")
    print(f"视图维度: 3")
    print(f"维度不匹配: 源 Tensor 是 2 维，但视图要求 3 维")
    
    # 测试: 错误示例 - 预期会触发 TENSOR_VIEW_DIMENSION_MISMATCH
    print("\n" + "-" * 60)
    print("测试: 错误示例（视图维度不匹配）")
    print("-" * 60)
    try:
        result = view_dimension_mismatch_example(x_data)
        print("错误：预期应该触发 TENSOR_VIEW_DIMENSION_MISMATCH，但没有报错")
        print(f"结果 shape: {result.shape}")
    except Exception as e:
        print(f"✓ 成功捕获错误: {type(e).__name__}")
        error_msg = str(e)
        # 只打印错误信息的第一行，避免堆栈信息过长
        error_lines = error_msg.split('\n')
        print(f"错误信息: {error_lines[0]}")
        
        # 检查是否触发了预期的错误
        if (any(keyword in error_msg for keyword in [
            "TENSOR_VIEW_DIMENSION_MISMATCH", 
            "0x12003",
            "view.*dimension",
            "dimension.*mismatch"
        ])):
            print("✓ 确认触发了 TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)")
        elif "view" in error_msg.lower() and "dimension" in error_msg.lower():
            print("✓ 触发了视图维度相关的错误")
        elif "view" in error_msg.lower():
            print("✓ 触发了视图相关的错误（与 TENSOR_VIEW_DIMENSION_MISMATCH 相关）")
        else:
            print(f"注意：触发了其他错误，请检查: {error_lines[0]}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TENSOR_VIEW_DIMENSION_MISMATCH")
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
    
    test_tensor_view_dimension_mismatch(device_id, args.run_mode)
