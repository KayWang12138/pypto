import pypto
import numpy as np
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

def golden_consistency_error_example(x, y):
    """
    Golden函数：错误示例 - 直接相加形状不一致的tensor
    在PyPTO中这会触发TENSOR_CONSISTENCY_ERROR
    """
    return x + y

def golden_correct_consistency_example(x, y):
    """
    Golden函数：正确示例 - 确保形状一致后再相加
    """
    if x.shape != y.shape:
        y = np.reshape(y, x.shape)
    return x + y

@pypto.frontend.jit
def consistency_error_example(
    x: pypto.Tensor((4, 4), pypto.DT_FP32),
    y: pypto.Tensor((2, 2), pypto.DT_FP32),
) -> pypto.Tensor((4, 4), pypto.DT_FP32):
    """
    错误示例：x 和 y 形状不一致且元素数量不一致时会导致错误
    这会触发形状相关的错误
    """
    pypto.set_vec_tile_shapes(4, 4)
    out = pypto.Tensor((4, 4), pypto.DT_FP32)
    out = pypto.add(x, y)
    return out

@pypto.frontend.jit
def correct_consistency_example(
    x: pypto.Tensor((4, 4), pypto.DT_FP32),
    y: pypto.Tensor((2, 8), pypto.DT_FP32),
) -> pypto.Tensor((4, 4), pypto.DT_FP32):
    """
    正确示例：元素数量相同但形状不同，先reshape确保形状一致
    """
    pypto.set_vec_tile_shapes(4, 4)
    out = pypto.Tensor((4, 4), pypto.DT_FP32)
    y_reshaped = pypto.reshape(y, [4, 4])
    out = pypto.add(x, y_reshaped)
    return out

@pypto.frontend.jit
def same_shape_example(
    x: pypto.Tensor((4, 4), pypto.DT_FP32),
    y: pypto.Tensor((4, 4), pypto.DT_FP32),
) -> pypto.Tensor((4, 4), pypto.DT_FP32):
    """
    形状一致的情况：直接相加
    """
    pypto.set_vec_tile_shapes(4, 4)
    out = pypto.Tensor((4, 4), pypto.DT_FP32)
    out = pypto.add(x, y)
    return out

def test_tensor_consistency_error(device_id=None, run_mode="npu"):
    """
    测试 TENSOR_CONSISTENCY_ERROR 错误码
    
    测试场景：
    1. 错误示例：直接相加形状不一致的tensor，预期会报错
    2. 正确示例：先reshape确保形状一致，预期成功
    3. 形状一致的情况：预期成功
    """
    print("=" * 60)
    print("测试 TENSOR_CONSISTENCY_ERROR (0x10007)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # 测试数据：创建形状不一致的tensor
    x_data = torch.randn(4, 4, dtype=torch.float32, device=device)
    y_data_error = torch.randn(2, 2, dtype=torch.float32, device=device)
    y_data_reshape = torch.randn(2, 8, dtype=torch.float32, device=device)
    
    print(f"\n输入 x shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"输入 y (错误测试) shape: {y_data_error.shape}, dtype: {y_data_error.dtype}")
    print(f"输入 y (reshape测试) shape: {y_data_reshape.shape}, dtype: {y_data_reshape.dtype}")
    
    # 测试1: 错误示例 - 预期会触发形状相关的错误
    print("\n" + "-" * 60)
    print("测试1: 错误示例（形状不一致直接相加）")
    print("-" * 60)
    
    result_error = consistency_error_example(x_data, y_data_error)
    print("错误：预期应该触发形状错误，但没有报错")
    print(f"结果 shape: {result_error.shape}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TENSOR_CONSISTENCY_ERROR")
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
    
    test_tensor_consistency_error(device_id, args.run_mode)
