# PyTorch 集成与接入

> **目标**：把 PyPTO 作为“自定义算子/融合算子”的实现与执行框架，平滑接入到 PyTorch/torch_npu 的工程里。

---

## 1. 接入前提（你必须明确的边界）

- **运行模式**
  - 真实环境：NPU 执行（依赖 CANN + torch_npu）
  - 仿真环境：SIM 执行（用于功能验证/预估性能）
- **版本一致性**
  - PyTorch、torch_npu、PyPTO 的 Python 版本必须一致
- **输入约束**
  - 很多模型算子/融合算子对输入有约束：**不支持非连续 Tensor（non-contiguous）**

---

## 2. 最小接入范式（PyTorch Tensor ↔ PyPTO Tensor）

典型流程：
1. 用 PyTorch 构造输入/输出张量
2. 转换为 PyPTO Tensor（必要时标注 dynamic_axis）
3. 调用 `@pypto.jit` / `@pypto.frontend.jit` 编译并执行
4. 在 PyTorch 侧取回输出进行验证/继续网络计算

### 2.1 最小可用代码示例

```python
import torch
import pypto
import numpy as np

# 确保输入是 contiguous（很多算子不支持非连续）
x = torch.randn(32, 128, dtype=torch.float32).contiguous()
y = torch.randn(32, 128, dtype=torch.float32).contiguous()
z = torch.zeros_like(x)  # 输出张量

@pypto.jit
def add_kernel(x, y, z):
    """最小示例：z = x + y"""
    pypto.set_vec_tile_shapes(32, 128)
    z[:] = x + y  # 显式写回输出

# 转换为 PyPTO Tensor
x_pypto = pypto.from_torch(x)
y_pypto = pypto.from_torch(y)
z_pypto = pypto.from_torch(z)

# 执行
add_kernel(x_pypto, y_pypto, z_pypto)

# 验证结果（取回 PyTorch Tensor）
result = z.numpy()
expected = (x + y).numpy()
np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
print("✅ 精度验证通过")
```

关键注意：
- 输出通常要求在 kernel 内显式写回：`out[:] = ...`
- 对于动态 batch/动态序列长度场景，优先明确 **哪些轴是动态轴**
- 输入张量建议先调用 `.contiguous()` 确保内存布局符合要求

---

## 3. 设备选择与运行前配置（真实 NPU）

在真实环境中，建议显式设置设备：

```bash
export TILE_FWK_DEVICE_ID=0
```

Python 侧需要时：
- `torch.npu.set_device(0)`

并确保 CANN 环境变量已生效（见 `docs/note/00-getting-started/01-environment-setup.md`）。

---

## 4. 动态形状接入（动态轴的工程化建议）

建议按下面顺序推进，避免“动态轴一上来就全开”导致定位困难：

1. **先固定 shape 跑通**（静态）
2. **只开一个动态轴**（最常见是 batch）
3. **把动态轴变成约束**：同一个算子多次执行时，静态轴必须保持一致；动态轴必须被正确标注

---

## 5. 工程化接入建议（模型/服务场景）

- **把 kernel 设计为纯函数接口**
  - 输入/输出张量显式传入
  - 避免隐式依赖全局状态（便于复现与测试）
- **为每个 kernel 提供最小回归用例**
  - 固定输入、固定 seed、固定 dtype
  - 先对齐 PyTorch baseline，再谈性能
- **把“结果查看与产物”纳入默认流程**
  - 固定输出目录
  - 记录 `run.log`
  - 需要时查看计算图/泳道图

---

## 6. 接入 Checklist（建议逐项检查）

在接入 PyPTO 到 PyTorch 项目前，建议按以下清单逐项检查：

### 6.1 版本一致性检查

- [ ] **Python 版本一致**：PyTorch、torch_npu、PyPTO 的 Python 版本必须一致
- [ ] **PyTorch 版本**：使用项目验证过的版本组合（如 PyTorch 2.6.0 + torch_npu 2.6.0）
- [ ] **验证方法**：`python3 -c "import torch; import torch_npu; import pypto; print('版本检查通过')"`

### 6.2 输入数据检查

- [ ] **Contiguous 检查**：所有输入 Tensor 调用 `.contiguous()` 确保内存布局符合要求
- [ ] **验证方法**：`assert x.is_contiguous()` 或 `x = x.contiguous()`
- [ ] **非连续问题**：详见[问题库](../05-debugging/03-troubleshooting-and-known-issues.md)中的 contiguous 相关条目

### 6.3 设备 ID 检查（真实 NPU 环境）

- [ ] **环境变量设置**：`export TILE_FWK_DEVICE_ID=0`
- [ ] **Python 侧设置**：`torch.npu.set_device(0)`（如需要）
- [ ] **验证方法**：在 `run.log` 中搜索 `TILE_FWK_DEVICE_ID` 确认已设置

### 6.4 动态轴标注检查

- [ ] **动态轴标注**：明确标注哪些轴是动态轴（`dynamic_axis`）
- [ ] **静态轴一致性**：确保静态轴在多次执行时保持一致
- [ ] **验证方法**：固定 shape 先跑通，再逐步引入动态轴

### 6.5 输出写回检查

- [ ] **显式写回**：kernel 内对输出 tensor 显式写回（`out[:] = ...`）
- [ ] **验证方法**：检查输出 tensor 是否保持初始值（如全零）

### 6.6 CANN 环境检查（真实 NPU 环境）

- [ ] **CANN 安装**：已完成 CANN 安装并 `source set_env.sh`
- [ ] **环境变量**：`LD_LIBRARY_PATH` 包含 CANN 库路径
- [ ] **验证方法**：检查环境变量或参考[环境准备与安装](../00-getting-started/01-environment-setup.md)

---

## 7. 常见接入问题速查

- **结果不生效**
  - 检查 kernel 是否对出参显式写回
  - 详见[问题库](../05-debugging/03-troubleshooting-and-known-issues.md)中的"kernel 出参未写回"条目
- **结果不稳定**
  - 检查是否使用了未初始化 tensor
  - 检查是否存在非连续输入导致的未定义行为
  - 详见[问题库](../05-debugging/03-troubleshooting-and-known-issues.md)中的"使用未初始化的 Tensor"条目
- **多次执行第二次才报错**
  - 检查静态轴是否变化
  - 检查 dynamic_axis 标注是否正确
  - 详见[问题库](../05-debugging/03-troubleshooting-and-known-issues.md)中的"静态轴传入不同运行时值"条目
- **Contiguous 约束问题**
  - 详见[问题库](../05-debugging/03-troubleshooting-and-known-issues.md)中的 contiguous 相关条目


