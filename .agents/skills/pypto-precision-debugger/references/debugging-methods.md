# 通用定位方法建议

## 精度工具使用

### 1. 开启精度工具验证

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}
)
```

### 2. 使用 set_verify_golden_data

```python
# 注意：必须保存 golden tensor 的引用，避免 GC 回收
golden_data = torch_output  # 保存引用
pypto.set_verify_golden_data(output=torch_output, golden=golden_data)
```

### 3. 精度工具校验通过但上板失败的处理

当精度工具校验通过但上板结果不符时：
- 检查仿真路径是否覆盖实际运行路径
- 使用二分定位法定位问题 op
- 检查 Pass 变换是否在仿真与上板间存在差异

---

## 二分定位法

使用 `pypto-binary-search-verify` skill 进行二分定位：

```bash
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

**步骤**：
1. 在计算图中添加中间输出检查点
2. 对比每个检查点的 tensor 与 golden
3. 定位精度偏差首次出现的 op

---

## 常见定位技巧

### 1. 对比不同写法

```python
# 写法1：简洁写法
for i in pypto.loop(N):
    result = compute(a, i)

# 写法2：手动展开写法
for i in pypto.loop(N, unroll_list=[1]):  # 关闭展开
    result = compute(a, i)
```

### 2. 检查动态轴处理

```python
# 手动取出动态轴（规避写法）
c_shape_0 = a_tensor.shape[0]  # 手动取出
c_tensor = pypto.Tensor((c_shape_0, z), dtype)
```

### 3. 检查 valid_shape 传播

```python
# 打印 reshape 前后的 valid_shape
print(f"Before reshape: valid_shape = {input_tensor.valid_shape}")
output = pypto.reshape(input_tensor, new_shape)
print(f"After reshape: valid_shape = {output.valid_shape}")
```


