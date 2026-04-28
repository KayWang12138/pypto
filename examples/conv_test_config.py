# ============================================================================
# Conv Dynamic Test Cases Configuration
# ============================================================================
# 
# 测试用例配置文件，方便管理新增用例
# 
# 配置说明：
# - name: 用例名称
# - conv_type: 卷积类型 (conv1d/conv2d/conv3d)
# - fmap_shape: 输入tensor形状
# - weight_shape: 权重tensor形状
# - bias_shape: bias tensor形状
# - out_shape: 输出tensor形状
# - strides: 步长
# - padding: 填充
# - dilations: 扩张
# - tile_config: tiling配置
#   - tile_batch: batch维度切分（固定为1，通过前端循环实现）
#   - tile_cout: cout维度切分（TileShape动态切分）
#   - tile_dout: dout维度切分 (仅conv3d)
#   - tile_hout: hout维度切分（TileShape动态切分）
#   - tile_wout: wout维度切分（前端循环）
#   - tile_cin: cin维度切分（前端循环 + add累加）
#               - 需满足 32字节对齐约束（FP16: %16==0）
#               - 切分后需累加多个tile结果（pypto.add）
#               - 第一次累加加bias，最后一次assemble到输出
#   - tile_l1: L1层tiling配置
#     - tileCinFmap/tileCinWeight: 需与tile_cin同步设置
#   - tile_l0: L0层tiling配置
#     - tileK: 需满足 kAL1%tileK==0, kBL1%tileK==0
#   - vec_tile: vector tiling配置
# ============================================================================

TEST_CASES = {
    # ============================================================================
    # Conv3D Test Cases (5D tensors: [batch, cin, din, hin, win])
    # ============================================================================
    "conv3d_small": {
        "name": "conv3d_small",
        "conv_type": "conv3d",
        "fmap_shape": (1, 32, 16, 8, 16),
        "weight_shape": (64, 32, 3, 3, 3),
        "bias_shape": (64,),
        "out_shape": (1, 64, 16, 8, 16),
        "strides": [1, 1, 1],
        "padding": [1, 1, 1, 1, 1, 1],
        "dilations": [1, 1, 1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 64,
            "tile_dout": 4,
            "tile_hout": 4,
            "tile_wout": 16,
            "tile_l1": {
                "tileHin": 4, "tileHout": 4, "tileWin": 16, "tileWout": 16,
                "tileCinFmap": 32, "tileCinWeight": 32, "tileN": 64, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 4, "tileW": 16, "tileK": 288, "tileN": 64
            },
            "vec_tile": (1, 64, 1, 4, 16)
        }
    },

    # ============================================================================
    # Conv2D Test Cases (4D tensors: [batch, cin, hin, win])
    # ============================================================================
    # Cin 动态轴切分示例（tile_cin 配置）
    # - tile_cin: cin 维度切分大小，需满足 32字节对齐（FP16: %16==0）
    # - cin 切分后需要累加多个 tile 的结果，使用 pypto.add 实现
    # - 第一次累加时加 bias，最后一次 assemble 到输出 tensor
    "conv2d_small": {
        "name": "conv2d_small",
        "conv_type": "conv2d",
        "fmap_shape": (1, 64, 64, 64),
        "weight_shape": (256, 64, 3, 3),
        "bias_shape": (256,),
        "out_shape": (1, 256, 64, 64),
        "strides": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilations": [1, 1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 256,
            "tile_hout": 16,
            "tile_wout": 16,
            "tile_cin": 64,  # Cin 不切分（完整 cin）
            "tile_l1": {
                "tileHin": 16, "tileHout": 16, "tileWin": 16, "tileWout": 16,
                "tileCinFmap": 64, "tileCinWeight": 64, "tileN": 256, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 16, "tileW": 16, "tileK": 64, "tileN": 256
            },
            "vec_tile": (1, 256, 16, 16)
        }
    },
    # Conv2D Cin 切分测试用例
    "conv2d_cin_tile": {
        "name": "conv2d_cin_tile",
        "conv_type": "conv2d",
        "fmap_shape": (1, 64, 64, 64),
        "weight_shape": (256, 64, 3, 3),
        "bias_shape": (256,),
        "out_shape": (1, 256, 64, 64),
        "strides": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilations": [1, 1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 256,
            "tile_hout": 16,
            "tile_wout": 16,
            "tile_cin": 32,  # Cin 切分：64 -> 2 tiles (32 each)
            "tile_l1": {
                "tileHin": 16, "tileHout": 16, "tileWin": 16, "tileWout": 16,
                "tileCinFmap": 32, "tileCinWeight": 32, "tileN": 256, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 4, "tileW": 16, "tileK": 288, "tileN": 64  # tileK = CeilAlign(32*3*3, 16) = 288
            },
            "vec_tile": (1, 4, 4, 16)
        }
    },
    "conv2d_large": {
        "name": "conv2d_large",
        "conv_type": "conv2d",
        "fmap_shape": (1, 512, 512, 512),
        "weight_shape": (128, 512, 1, 1),
        "bias_shape": (128,),
        "out_shape": (1, 128, 512, 512),
        "strides": [1, 1],
        "padding": [0, 0, 0, 0],
        "dilations": [1, 1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 128,
            "tile_hout": 32,
            "tile_wout": 32,
            "tile_cin": 512,  # Cin 不切分
            "tile_l1": {
                "tileHin": 32, "tileHout": 32, "tileWin": 32, "tileWout": 32,
                "tileCinFmap": 128, "tileCinWeight": 128, "tileN": 128, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 8, "tileW": 32, "tileK": 128, "tileN": 128
            },
            "vec_tile": (1, 128, 8, 32)
        }
    },
    # Conv2D Cin 大切分测试用例
    "conv2d_large_cin_tile": {
        "name": "conv2d_large_cin_tile",
        "conv_type": "conv2d",
        "fmap_shape": (1, 512, 512, 512),
        "weight_shape": (128, 512, 1, 1),
        "bias_shape": (128,),
        "out_shape": (1, 128, 512, 512),
        "strides": [1, 1],
        "padding": [0, 0, 0, 0],
        "dilations": [1, 1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 128,
            "tile_hout": 32,
            "tile_wout": 32,
            "tile_cin": 128,  # Cin 切分：512 -> 4 tiles (128 each)
            "tile_l1": {
                "tileHin": 32, "tileHout": 32, "tileWin": 32, "tileWout": 32,
                "tileCinFmap": 128, "tileCinWeight": 128, "tileN": 128, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 8, "tileW": 32, "tileK": 128, "tileN": 128  # tileK = CeilAlign(128*1*1, 16) = 128
            },
            "vec_tile": (1, 128, 8, 32)
        }
    },

    # ============================================================================
    # Conv1D Test Cases (3D tensors: [batch, cin, win])
    # ============================================================================
    "conv1d_small": {
        "name": "conv1d_small",
        "conv_type": "conv1d",
        "fmap_shape": (2, 16, 2048),
        "weight_shape": (16, 16, 3),
        "bias_shape": (16,),
        "out_shape": (2, 16, 2048),
        "strides": [1],
        "padding": [1, 1],
        "dilations": [1],
        "tile_config": {
            "tile_batch": 1,
            "tile_cout": 16,
            "tile_wout": 16,
            "tile_cin": 16,  # Cin 不切分
            "tile_l1": {
                "tileHin": 1, "tileHout": 1, "tileWin": 16, "tileWout": 16,
                "tileCinFmap": 16, "tileCinWeight": 16, "tileN": 16, "tileBatch": 1
            },
            "tile_l0": {
                "tileH": 1, "tileW": 16, "tileK": 16, "tileN": 16
            },
            "vec_tile": (1, 16, 16)
        }
    },
}


# ============================================================================
# Helper function to get test cases by conv type
# ============================================================================
def get_cases_by_conv_type(conv_type: str) -> dict:
    """
    Get all test cases for a specific conv type.
    
    Args:
        conv_type: "conv1d", "conv2d", or "conv3d"
    
    Returns:
        dict of test cases matching the conv type
    """
    return {name: case for name, case in TEST_CASES.items() 
            if case["conv_type"] == conv_type}


def get_case_by_name(case_name: str) -> dict:
    """
    Get a specific test case by name.
    
    Args:
        case_name: The name of the test case
    
    Returns:
        dict of the test case, or None if not found
    """
    return TEST_CASES.get(case_name)