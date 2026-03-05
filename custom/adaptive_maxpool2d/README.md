# MaxPool2D (固定核) - PyPTO 实现

## 概述

本文档记录了基于 PyPTO 实现的 MaxPool2D 算子(固定 kernel 大小版本)。

**注意**: 原计划实现 AdaptiveMaxPool2D (自适应池化),但由于 PyPTO `view()` API 的限制(不支持动态窗口大小),改用固定 kernel 的 MaxPool2D 实现。
## 宺述描述
### 目录结构
```
custom/adaptive_maxpool2d/
├── adaptive_maxpool2d.py      # Adaptive 版本(未完成,已废弃)
├── maxpool2d_fixed.py        # 固定 kernel 版本(实际实现)
└── README.md                   # 本文档
```
### 开发背景
#### AdaptiveMaxPool2D 的挑战
原计划实现 PyTorch 的 `F.adaptive_max_pool2d()` 功能,但遇到以下技术限制:
1. **PyPTO view() 限制**: `view()` 的 shape 参数必须是编译时常量,不支持循环中动态计算的窗口大小
2. **Adaptive 特性**: AdaptiveMaxPool2D 的核心是窗口大小根据输入/输出尺寸动态计算
   例如: `window_size = ceil(input_size / output_size)`
#### 解决方案
改用 **固定 kernel 的 MaxPool2D**:
- 固定的 `kernel_size` 和 `stride`
- 所有窗口大小相同,可以直接使用 `view()`
- 简化了实现复杂度
- 优先保证功能正确性
## 实现细节
### 核心 API
```python
pypto.view(input, shape, offsets)     # 提取池化窗口
pypto.amax(input, dim, keepdim)      # 计算最大值(两次归约)
pypto.assemble(output, offsets, out) # 写回输出位置
```
### 关键约束
1. **32 字节对齐**: `amax()` 要求 tileShape 最后维度必须是 32 字节对齐
   - FP16: 黡足 `elements * 2 bytes % 32 == 0`
   - BF16: 需要特别注意,处理更复杂
2. **固定窗口**: 使用固定的 `kernel_size=2` 和 `stride=2`
3. **view+assemble 组合**:
   ```python
   window = pypto.view(input, [1, 1, kernel_size, align_size], offsets)
   max_w = pypto.amax(window, dim=-1, keepdim=True)
   max_val = pypto.amax(max_w, dim=-2, keepdim=True)
   result = pypto.view(max_val, [1, 1, 1, 1], [0, 0, 0, 0])
   pypto.assemble(result, [b_idx, c_idx, h_out_idx, w_out_idx], output)
   ```
## 测试结果
### 测试环境
- **服务器**: A3
- **CANN 版本**: 8.5.0
- **运行模式**: NPU
### 测试用例
#### 测试1: 基本用例 (4x4 -> 2x2)
```python
输入形状: [1, 1, 4, 4]
输出形状: [1, 1, 2, 2]
Kernel: 2x2, Stride: 2
数据类型: FP16
运行模式: NPU
最大误差: 0.000000
结果: ✅ 通过
```
#### 测试2: 中等规模 (8x8 -> 4x4)
```python
输入形状: [2, 3, 8, 8]
输出形状: [2, 3, 4, 4]
Kernel: 2x2, Stride: 2
数据类型: FP16
运行模式: NPU
最大误差: 0.000000
结果: ✅ 通过
```
## 使用方法
### 编译和安装
```bash
# 1. 编译 whl 包
python3 build_ci.py -f python3 --disable_auto_execute

# 2. 安装
pip install --force-reinstall build_out/pypto-0.1.0-*.whl

# 3. 设置环境变量
export TILE_FWK_DEVICE_ID=0
```
### 运行测试
```bash
# 运行所有测试
python3 custom/adaptive_maxpool2d/maxpool2d_fixed.py

# 运行单个测试
python3 custom/adaptive_maxpool2d/maxpool2d_fixed.py 1  # 测试1
python3 custom/adaptive_maxpool2d/maxpool2d_fixed.py 2  # 测试2

# 查看可用测试
python3 custom/adaptive_maxpool2d/maxpool2d_fixed.py --list
```
## 已知限制
1. **不支持动态窗口大小**: 由于 PyPTO `view()` API 限制,无法实现真正的 AdaptiveMaxPool2D
2. **固定 kernel 大小**: 当前实现仅支持 `kernel_size=2, stride=2`
3. **BF16 支持有限**: 由于 32 字节对齐要求复杂,BF16 测试用例未完全验证
4. **输入宽度限制**: 输入宽度需要是 `align_size` 的倍数以确保内存访问安全
## 性能说明
- **Tile 配置**: `[1, 1, kernel_size, align_size]`
- **向量化**: 利用 NPU 向量单元进行归约计算
- **内存访问**: 通过 view+assemble 实现高效的内存访问模式
## 技术总结
### PyPTO view() API 特性
1. **shape 参数**: 必须是编译时常量(整数列表)
2. **offsets 参数**: 可以是循环变量(SymbolicScalar)
3. **32 字节对齐**: TileShape 最后维度必须 32 字节对齐
### 实现经验
1. **优先保证正确性**: 复杂的优化应放在功能验证之后
2. **参考官方示例**: PyPTO 有很多技术限制,官方示例是最好的参考
3. **简化问题**: 遇到复杂问题时,考虑简化方案(如本文从 Adaptive 改为固定 kernel)
## 参考资料
- [PyPTO view() API 文档](../../docs/api/operation/pypto-view.md)
- [PyPTO amax() API 文档](../../docs/api/operation/pypto-amax.md)
- [PyPTO assemble() API 文档](../../docs/api/operation/pypto-assemble.md)
- [PyPTO 示例代码](../../examples/01_beginner/)
- [PyTorch MaxPool2D 文档](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
