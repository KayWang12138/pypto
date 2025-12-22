# 样例运行

## 仿真环境（无 NPU 真实硬件）

```bash
cd examples/hello_world
python3 hello_world.py --run_mode=sim --tensor_type=cpu
```

## 真实可运行环境（有 NPU 真实硬件）

```bash
cd examples/hello_world
python3 hello_world.py --run_mode=npu --tensor_type=npu
```
更多示例请参考 `examples/` 目录下的示例代码。
## 结果查看

该基础样例运行成功后, 在`${work_path}/out/`目录下生成编译和运行产物，相关产物包括[计算图](../docs/key-concepts/compute_graph.md)和[泳道图](../docs/key-concepts/swim_graph.md), 计算图和泳道图可通过PyPTO配套的[ToolKit插件](xxx), 在VS-CODE中查看并与代码关联, 相关ToolKit使用应参考[ToolKit-QuickStart](xxxxxxx)

## 快速开始
以下是一个简单的 PyPTO 使用示例：

```python
import pypto
import torch

# 定义计算函数
@pypto.jit
def add_kernel(x0, x1, y):
    pypto.set_vec_tile_shapes(4, 4)
    y[:] = x0 + x1

# 创建 Tensor
x0 = torch.ones(4, 4, dtype=torch.float32)
x1 = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)

# 执行计算
add_kernel(pypto.from_torch(x0), pypto.from_torch(x1), pypto.from_torch(y))
print(y)
```
- 对于真实环境，可以直接通过查看输出 `y` 的值查看运行结果
- 对于仿真环境，通过示例同级目录中 `out/` 下的泳道图查看仿真结果  

