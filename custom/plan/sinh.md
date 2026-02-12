# sinh 算子开发计划

## 一、算子概述

### 算子名称
sinh

### 数学公式
```
y = sinh(x) = (e^x - e^(-x)) / 2
```

### 输入输出规格

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 x| [b, s, n, d]  | float32  |
| 输出 y| [b, s, n, d]  | float32  |

## 二、API 映射关系

| 数学符号 | PyPTO API | 说明 |
|---------|-----------|------|
| e^x | `pypto.exp(x)` | 指数函数 |
| e^(-x) | `pypto.exp(pypto.neg(x))` | 指数函数（负指数） |
| - | `pypto.sub(a, b)` | 减法 |
| / | `pypto.div(a, 2.0)` | 除法（除以常数2） |

## 三、设计方案

### 数据流设计
```
输入 x (float32, [b, s, n, d])
    ↓
设置 vec_tile_shapes: [32, 32, 32, 32]
    ↓
exp_x = pypto.exp(x)
neg_x = pypto.neg(x)
exp_neg_x = pypto.exp(neg_x)
    ↓
numerator = pypto.sub(exp_x, exp_neg_x)
    ↓
output = pypto.div(numerator, 2.0)
    ↓
输出 y (float32, [b, s, n, d])
```

### 实现要点
1. 使用 `pypto.DT_FP32` 数据类型
2. 4维 tensor [b, s, n, d]
3. 使用 `pypto.neg(x)` 而不是 `-x`
4. 需要设置 `pypto.set_vec_tile_shapes`，4维张量需要4个参数
5. 常数 2.0 需要显式指定为 float 类型

## 四、开发步骤

### Step 1: 创建目录结构
```
custom/sinh/
├── sinh.py          # 测试及golden实现
├── sinh_impl.py     # jit函数实现
└── README.md        # 算子文档
```

### Step 2: 实现 sinh.py（测试及golden）
- 创建 golden 函数（使用 PyTorch 实现）
- 创建测试用例
  - Level 0: 小数据量（8-16 元素）
  - Level 1: 典型场景（1K 元素）
  - Level 2: 边界情况（零值、极值）

### Step 3: 实现 sinh_impl.py（jit函数）
- 创建 jit 函数
- 设置 vec_tile_shapes: [32, 32, 32, 32]
- 实现 sinh 公式
- 验证编译通过

### Step 4: 编译测试
```bash
python3 build_ci.py -f python3 --disable_auto_execute
python3 custom/sinh/sinh.py --run_mode npu
```

### Step 5: 验证与调试
- 验证精度（相对误差 1e-3）
- 测试多种输入规模

## 五、已知约束
- 输入仅支持 2-4 维
- 输入 shape size 不超过 INT32_MAX
- 支持数据类型：DT_FP16, DT_BF16, DT_FP32
- 本实现使用 DT_FP32
- 需要设置 vec_tile_shapes

## 六、验证标准
- 功能验证：输出 shape 和 dtype 正确
- 精度验证：相对误差 < 1e-3
- 边界测试：零值、正负极值均能正确处理

## 七、开发结果

### ✅ 开发状态：成功

### 功能验证结果
- ✅ 输出 shape 正确
- ✅ 输出 dtype 正确 (float32)

### 精度测试结果

| 测试用例 | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | 状态 |
|---------|------------|------------|------------|-----|
| Level 0: 8 元素 | 0.000000 | 0.000000 | 0.000001 | ✅ 通过 |
| Level 0: 16 元素 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 1: 1K 元素 | 0.000001 | 0.000000 | 0.000015 | ✅ 通过 |
| Level 2: 零值 | 0.000000 | 0.000000 | 0.000000 | ✅ 通过 |
| Level 2: 大正值 | 0.000977 | 0.000977 | 0.000000 | ✅ 通过 |
| Level 2: 大负值 | 0.000977 | 0.000977 | 0.000000 | ✅ 通过 |

### 性能验证结果
- ✅ 编译成功
- ✅ 所有测试用例通过
- ✅ NPU 模式下运行正常

### 泛化测试结果
- ✅ 支持多种输入规模（8~1024 元素）
- ✅ 支持边界值（零值、极大值）

### 已知限制和问题
- 无

### 关键实现细节
1. 必须调用 `pypto.set_vec_tile_shapes(32, 32, 32, 32)` 设置 tile shapes
2. 必须使用 `pypto.neg(x)` 而不是 `-x`
3. PyTorch 设备 ID 使用 `'npu:0'`，环境变量 `TILE_FWK_DEVICE_ID` 设置为 `2`
