"""
pypto.matmul ST测试用例配置
用于System Test自动化测试框架
"""

"""
+------------------+--------+-----------+
| 测试组           | 用例数  |  产品支持  |
+------------------+--------+-----------+
| basic            |   6    | 950, 910  |
| nz_format        |   2    | 950, 910  |
| bias             |   3    | 950, 910  |
| relu             |   2    | 950, 910  |
| quantization     |   4    | 950, 910  |
| product_specific |   8    | 950       |
+------------------+--------+-----------+
| 总计             |  25    |           |
+------------------+--------+-----------+
"""

# 基础功能测试用例 - 不同dtype组合（含转置）
BASIC_TESTS = [
    {
        "id": "B01",
        "name": "fp16_2d_nd_out_fp16",
        "desc": "FP16输入FP16输出",
        "input_a": {"dtype": "DT_FP16", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B02",
        "name": "fp16_2d_nd_out_fp32_trans_a",
        "desc": "FP16输入FP32输出+A转置",
        "input_a": {"dtype": "DT_FP16", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B03",
        "name": "bf16_2d_nd_out_bf16_trans_b",
        "desc": "BF16输入BF16输出+B转置",
        "input_a": {"dtype": "DT_BF16", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_BF16", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_BF16"},
        "transpose": {"a_trans": False, "b_trans": True},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B04",
        "name": "bf16_2d_nd_out_fp32",
        "desc": "BF16输入FP32输出",
        "input_a": {"dtype": "DT_BF16", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_BF16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B05",
        "name": "fp32_2d_nd_out_fp32_trans_both",
        "desc": "FP32输入FP32输出+双转置",
        "input_a": {"dtype": "DT_FP32", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP32", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": True, "b_trans": True},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B06",
        "name": "int8_2d_nd_out_int32",
        "desc": "INT8输入INT32输出",
        "input_a": {"dtype": "DT_INT8", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "products": ["950", "910"],
    },
]

# NZ格式测试用例（含转置）
NZ_FORMAT_TESTS = [
    {
        "id": "NZ01",
        "name": "fp16_2d_nz",
        "desc": "FP16 NZ格式",
        "input_a": {"dtype": "DT_FP16", "shape": [128, 256], "format": "NZ"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "NZ"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "alignment": {"inner_axis": 32, "outer_axis": 16},
        "products": ["950", "910"],
    },
    {
        "id": "NZ02",
        "name": "int8_2d_nz_trans_a",
        "desc": "INT8 NZ格式+A转置(16元素对齐)",
        "input_a": {"dtype": "DT_INT8", "shape": [256, 128], "format": "NZ"},
        "input_b": {"dtype": "DT_INT8", "shape": [256, 512], "format": "NZ"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {},
        "alignment": {"inner_axis": 16, "outer_axis": 16},
        "products": ["950", "910"],
    },
]

# 扩展参数测试用例 - Bias（含转置）
BIAS_TESTS = [
    {
        "id": "E01",
        "name": "fp16_bias_fp16",
        "desc": "FP16输入带FP16 Bias",
        "input_a": {"dtype": "DT_FP16", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {
            "bias_tensor": {"dtype": "DT_FP16", "shape": [1, 512]}
        },
        "products": ["950", "910"],
    },
    {
        "id": "E02",
        "name": "fp16_bias_fp32_trans_a",
        "desc": "FP16输入带FP32 Bias+A转置",
        "input_a": {"dtype": "DT_FP16", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {
            "bias_tensor": {"dtype": "DT_FP32", "shape": [1, 512]}
        },
        "products": ["950", "910"],
    },
    {
        "id": "E03",
        "name": "int8_bias_int32_trans_b",
        "desc": "INT8输入带INT32 Bias+B转置",
        "input_a": {"dtype": "DT_INT8", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": False, "b_trans": True},
        "extend_params": {
            "bias_tensor": {"dtype": "DT_INT32", "shape": [1, 512]}
        },
        "products": ["950", "910"],
    },
]

# 扩展参数测试用例 - ReLU（含转置）
RELU_TESTS = [
    {
        "id": "E04",
        "name": "fp16_bias_relu",
        "desc": "FP16带Bias和ReLU",
        "input_a": {"dtype": "DT_FP16", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {
            "bias_tensor": {"dtype": "DT_FP16", "shape": [1, 512]},
            "relu_type": "RELU"
        },
        "products": ["950", "910"],
    },
    {
        "id": "E05",
        "name": "fp16_relu_only_trans_a",
        "desc": "仅ReLU无Bias+A转置",
        "input_a": {"dtype": "DT_FP16", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP16", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {
            "relu_type": "RELU"
        },
        "products": ["950", "910"],
    },
]

# 量化测试用例（含转置）
QUANT_TESTS = [
    {
        "id": "Q01",
        "name": "int8_scale_pertensor",
        "desc": "INT8 PerTensor量化",
        "input_a": {"dtype": "DT_INT8", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {
            "scale": 0.125
        },
        "products": ["950", "910"],
    },
    {
        "id": "Q02",
        "name": "int8_scale_perchannel_trans_a",
        "desc": "INT8 PerChannel量化+A转置",
        "input_a": {"dtype": "DT_INT8", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {
            "scale_tensor": {"dtype": "DT_UINT64", "shape": [1, 512]}
        },
        "products": ["950", "910"],
    },
    {
        "id": "Q03",
        "name": "int8_scale_relu_trans_b",
        "desc": "INT8量化+ReLU+B转置",
        "input_a": {"dtype": "DT_INT8", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": False, "b_trans": True},
        "extend_params": {
            "scale": 0.125,
            "relu_type": "RELU"
        },
        "products": ["950", "910"],
    },
    {
        "id": "Q04",
        "name": "int8_full_quant_trans_both",
        "desc": "INT8量化+Bias+ReLU+双转置",
        "input_a": {"dtype": "DT_INT8", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_INT8", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_INT32"},
        "transpose": {"a_trans": True, "b_trans": True},
        "extend_params": {
            "scale": 0.125,
            "bias_tensor": {"dtype": "DT_INT32", "shape": [1, 512]},
            "relu_type": "RELU"
        },
        "products": ["950", "910"],
    },
]

# 950专属特性测试（含转置）
PRODUCT_SPECIFIC_TESTS = [
    {
        "id": "P01",
        "name": "fp8e5m2_basic",
        "desc": "FP8E5M2基础场景",
        "input_a": {"dtype": "DT_FP8E5M2", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP8E5M2", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P02",
        "name": "fp8e5m2_out_bf16_trans_a",
        "desc": "FP8E5M2输出BF16+A转置",
        "input_a": {"dtype": "DT_FP8E5M2", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP8E5M2", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_BF16"},
        "transpose": {"a_trans": True, "b_trans": False},
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P03",
        "name": "fp8e5m2_out_fp32_trans_b",
        "desc": "FP8E5M2输出FP32+B转置",
        "input_a": {"dtype": "DT_FP8E5M2", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP8E5M2", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": False, "b_trans": True},
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P04",
        "name": "fp8e4m3_basic_trans_both",
        "desc": "FP8E4M3+双转置",
        "input_a": {"dtype": "DT_FP8E4M3", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP8E4M3", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": True, "b_trans": True},
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P05",
        "name": "hf8_basic",
        "desc": "HF8基础场景",
        "input_a": {"dtype": "DT_HF8", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_HF8", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P06",
        "name": "fp32_tf32_rint_trans_b",
        "desc": "FP32使能TF32(RINT)+B转置",
        "input_a": {"dtype": "DT_FP32", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP32", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": False, "b_trans": True},
        "extend_params": {
            "trans_mode": "CAST_RINT"
        },
        "products": ["950"],
    },
    {
        "id": "P07",
        "name": "fp32_tf32_round",
        "desc": "FP32使能TF32(ROUND模式)",
        "input_a": {"dtype": "DT_FP32", "shape": [128, 256], "format": "ND"},
        "input_b": {"dtype": "DT_FP32", "shape": [256, 512], "format": "ND"},
        "output": {"dtype": "DT_FP32"},
        "transpose": {"a_trans": False, "b_trans": False},
        "extend_params": {
            "trans_mode": "CAST_ROUND"
        },
        "products": ["950"],
    },
    {
        "id": "P08",
        "name": "fp8_bias_trans_both",
        "desc": "FP8带FP32 Bias+双转置",
        "input_a": {"dtype": "DT_FP8E5M2", "shape": [256, 128], "format": "ND"},
        "input_b": {"dtype": "DT_FP8E5M2", "shape": [512, 256], "format": "ND"},
        "output": {"dtype": "DT_FP16"},
        "transpose": {"a_trans": True, "b_trans": True},
        "extend_params": {
            "bias_tensor": {"dtype": "DT_FP32", "shape": [1, 512]}
        },
        "products": ["950"],
    },
]

# 所有测试用例分组
TEST_GROUPS = {
    "basic": BASIC_TESTS,
    "nz_format": NZ_FORMAT_TESTS,
    "bias": BIAS_TESTS,
    "relu": RELU_TESTS,
    "quantization": QUANT_TESTS,
    "product_specific": PRODUCT_SPECIFIC_TESTS,
}


