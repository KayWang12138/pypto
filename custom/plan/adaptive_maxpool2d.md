# AdaptiveMaxPool2D 算子开发计划

## 一、需求分析

### 1.1 算子功能
自适应最大池化（AdaptiveMaxPool2D），将任意尺寸的输入特征图池化到指定的输出尺寸。

### 1.2 数学公式
对于输入张量 `X` 的形状为 `[batch, channel, H_in, W_in]`，输出张量 `Y` 的形状为 `[batch, channel, H_out, W_out]`：

池化窗口大小计算：
```
kernel_h = ceil(H_in / H_out)
kernel_w = ceil(W_in / W_out)
```

输出计算：
```
Y[b, c, i, j] = max{X[b, c, h_start:h_end, w_start:w_end]}
```

其中：
- `h_start = floor(i * H_in / H_out)`
- `h_end = floor((i + 1) * H_in / H_out)`
- `w_start = floor(j * W_in / W_out)`
- `w_end = floor((j + 1) * W_in / W_out)`

### 1.3 输入输出规格

**输入**：
- `input`: Tensor，形状为 `[batch, channel, H, W]`
  - 支持的数据类型：`DT_FP16`, `DT_BF16`, `DT_FP32`
  - 维度：4D

**输出**：
- `output`: Tensor，形状为 `[batch, channel, output_H, output_W]`
  - 数据类型与输入相同

### 1.4 精度要求
- 相对误差容忍度：`rtol=1e-3`
- 绝对误差容忍度：`atol=1e-3`

## 二、技术方案

### 2.1 核心 API
1. **pypto.view(input, shape, offsets, valid_shape)** - 提取池化窗口
2. **pypto.amax(input, dim, keepdim)** - 计算窗口最大值（沿 H 和 W 维度）
3. **pypto.assemble(input, offsets, out)** - 写回输出位置
4. **pypto.loop(start, end, step, name, idx_name)** - 遍历输出位置
5. **pypto.set_vec_tile_shapes(...)** - 配置向量化 Tiling

### 2.2 实现流程

```
1. 对每个 batch, channel：
   2. 对每个输出位置 (i, j)：
      a. 计算输入窗口的起始和结束位置
         - h_start = floor(i * H_in / H_out)
         - h_end = floor((i + 1) * H_in / H_out)
         - w_start = floor(j * W_in / W_out)
         - w_end = floor((j + 1) * W_in / W_out)
      
      b. 使用 view 提取窗口
         window = view(input[b, c, :, :], 
                       [h_end-h_start, w_end-w_start],
                       [h_start, w_start])
      
      c. 使用 amax 计算最大值
         max_val = amax(amax(window, dim=-1, keepdim=True), dim=-2, keepdim=True)
      
      d. 使用 assemble 写回输出
         assemble(max_val, [b, c, i, j], output)
```

### 2.3 优化策略

#### 2.3.1 Tiling 策略
- 向量化计算：`pypto.set_vec_tile_shapes(tile_h, tile_w)`
- 建议配置：根据 NPU 向量单元大小，设置为 `(32, 64)` 或类似值

#### 2.3.2 并行策略
- Batch 和 Channel 维度可以并行
- 输出位置 (H_out, W_out) 的循环串行执行

### 2.4 边界处理
- 当 `H_in` 不能被 `H_out` 整除时，最后一个窗口可能较小
- 使用 `valid_shape` 参数处理边界情况

## 三、开发计划

### 3.1 文件结构
```
custom/adaptive_maxpool2d/
├── adaptive_maxpool2d.py          # 主文件：包含实现和测试
└── README.md                       # 文档
```

### 3.2 开发步骤

#### Step 1: 创建目录结构
```bash
mkdir -p custom/adaptive_maxpool2d
```

#### Step 2: 编写测试文件
- 实现 PyTorch golden 函数（使用 `torch.nn.functional.adaptive_max_pool2d`）
- 实现测试用例
  - 用例 1：小尺寸输入 (4x4 -> 2x2)
  - 用例 2：中等尺寸输入 (8x8 -> 3x3)
  - 用例 3：非整除情况 (7x7 -> 3x3)

#### Step 3: 实现 PyPTO 算子
- 使用 `@pypto.frontend.jit()` 装饰器
- 实现核心逻辑（view -> amax -> assemble）
- 处理边界情况

#### Step 4: 编译安装
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

#### Step 5: 测试验证
```bash
# 设置 NPU 设备
export TILE_FWK_DEVICE_ID=0

# 运行测试
python3 custom/adaptive_maxpool2d/adaptive_maxpool2d.py
```

#### Step 6: 编写文档
- 算子概述
- 实现原理
- 测试结果
- 已知限制

## 四、验证标准

### 4.1 功能验证
- [ ] 基本功能：正确计算最大池化结果
- [ ] 边界情况：处理非整除的输入尺寸
- [ ] 数据类型：支持 FP16, BF16, FP32

### 4.2 精度验证
- [ ] 与 PyTorch golden 结果对比
- [ ] 最大误差 < 1e-3
- [ ] 通过率 100%

### 4.3 性能验证
- [ ] 在 NPU 上成功运行
- [ ] 无编译警告或错误

## 五、风险与注意事项

### 5.1 技术风险
1. **循环嵌套深度**：4 层循环（batch, channel, H_out, W_out）可能导致性能问题
   - 缓解：尝试展开部分循环或优化 Tiling 策略

2. **边界情况**：非整除的窗口大小处理
   - 缓解：使用 `valid_shape` 参数确保正确性

### 5.2 环境依赖
- 需要设置 `TILE_FWK_DEVICE_ID` 环境变量
- 需要 NPU 设备可用

## 六、已知限制

1. 仅支持 4D 输入（batch, channel, H, W）
2. 不支持动态 shape（需要先实现静态 shape 版本）
3. 暂不优化性能（优先保证功能正确）

## 七、参考资料

- PyTorch AdaptiveMaxPool2D: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html
- PyPTO API 文档: `/docs/api/operation/`
- 示例代码: `/examples/01_beginner/basic/basic_ops.py`
