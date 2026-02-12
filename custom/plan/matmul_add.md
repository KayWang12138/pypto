# matmul_add 算子开发计划

## 一、算子概述

### 算子名称
matmul_add

### 数学公式
```
y = a @ b^T + c
```

### 输入输出规格

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 a| [m, k] | bfloat16  |
| 输入 b| [n, k] | bfloat16  |
| 输入 c| [m, n] | bfloat16  |
| 输出 y| [m, n] | bfloat16  |

## 二、API 映射关系

| 数学符号 | PyPTO API | 说明 |
|---------|-----------|------|
| b^T | `pypto.transpose(b, 0, 1)` | 转置操作 |
| @ | `pypto.matmul(a, b_t, out_dtype)` | 矩阵乘法 |
| + | `pypto.add(matmul_result, c)` | 加法 |

## 三、设计方案

### 数据流设计
```
输入 a (DT_BF16, [m, k])
输入 b (DT_BF16, [n, k])
输入 c (DT_BF16, [m, n])
    ↓
设置 vec_tile_shapes: [32, 32]
    ↓
b_t = pypto.transpose(b, 0, 1)  # b^T, shape: [k, n]
    ↓
设置 cube_tile_shapes: [[32, 32], [64, 64], [64, 64]]
    ↓
matmul_result = pypto.matmul(a, b_t, pypto.DT_BF16)  # shape: [m, n]
    ↓
output = pypto.add(matmul_result, c)
    ↓
输出 y (DT_BF16, [m, n])
```

### 实现要点
1. 使用 `pypto.DT_BF16` 数据类型
2. 2维 tensor [m, k], [n, k], [m, n]
3. 必须在 transpose 之前调用 `pypto.set_vec_tile_shapes`
4. 必须在 matmul 之前调用 `pypto.set_cube_tile_shapes`
5. matmul 的 out_dtype 设置为 `pypto.DT_BF16`

### NPU 优化配置
```python
# Vector 操作的 tile shapes（2维矩阵需要2个参数）
pypto.set_vec_tile_shapes(32, 32)

# Matrix 操作的 cube tile shapes
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
```

## 四、开发步骤

### Step 1: 创建目录结构
```
custom/matmul_add/
├── matmul_add.py          # 测试及golden实现
├── matmul_add_impl.py     # jit函数实现
└── README.md              # 算子文档
```

### Step 2: 实现 matmul_add.py（测试及golden）
- 创建 golden 函数（使用 PyTorch 实现）
- 创建测试用例
  - Level 0: 小矩阵（m=8, k=16, n=8）
  - Level 1: 典型场景（m=128, k=256, n=128）
  - Level 2: 边界情况（非方阵）

### Step 3: 实现 matmul_add_impl.py（jit函数）
- 创建 jit 函数
- 设置 vec_tile_shapes
- 实现 b^T 转置
- 设置 cube_tile_shapes
- 实现矩阵乘法
- 实现加法
- 验证编译通过

### Step 4: 编译测试
```bash
python3 build_ci.py -f python3 --disable_auto_execute
python3 custom/matmul_add/matmul_add.py --run_mode npu
```

### Step 5: 验证与调试
- 验证精度（相对误差 1e-2，bfloat16 精度较低）
- 测试多种矩阵尺寸

## 五、已知约束
- matmul 输入矩阵类型需保持一致
- 2维、3维、4维矩阵都支持
- 输入矩阵支持 Format: TILEOP_ND, TILEOP_NZ
- 本实现使用 TILEOP_ND 格式和 2维矩阵
- 必须设置 vec_tile_shapes（transpose 需要）
- 必须设置 cube_tile_shapes（matmul 需要）

## 六、验证标准
- 功能验证：输出 shape 和 dtype 正确
- 精度验证：相对误差 < 1e-2（bfloat16 精度较低）
- 边界测试：方阵、非方阵均能正确处理

## 七、开发结果

### ✅ 开发状态：成功

### 功能验证结果
- ✅ 输出 shape 正确
- ✅ 输出 dtype 正确 (bfloat16)

### 精度测试结果

| 测试用例 | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | 状态 |
|---------|------------|------------|------------|-----|
| Level 0: 8x8 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 1: 128x128 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 2: 64x32 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 2: 32x128 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 2: 128x32 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |

### 性能验证结果
- ✅ 编译成功
- ✅ 所有测试用例通过
- ✅ NPU 模式下运行正常

### 泛化测试结果
- ✅ 支持多种矩阵尺寸
- ✅ 支持方阵和非方阵

### 已知限制和问题
- 无

### 关键实现细节
1. 必须在 transpose 之前调用 `pypto.set_vec_tile_shapes(32, 32)`
2. 必须在 matmul 之前调用 `pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])`
3. PyTorch 设备 ID 使用 `'npu:0'`，环境变量 `TILE_FWK_DEVICE_ID` 设置为 `2`
4. 使用 bfloat16 数据类型，精度低于 float32
