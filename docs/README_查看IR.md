# 总结：普通用户如何在Python侧查看IR

## 📌 核心问题

新IR使用msgpack-c序列化，保存的是**二进制格式**（不可读）。**普通用户如何查看具体IR？**

## ✅ 解决方案

使用 **`ir.python_print(program)`** 将IR转换为**标准Python代码**！

---

## 🎯 真实示例：Add算子（C[i] = A[i] + B[i]）

### 原始IR（未优化）

```python
@pl.program
class VectorAdd:
    @pl.function
    def add(self, A: pl.Tensor[[1024], pl.FP32],
                  B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.empty([1024], pl.FP32)

        for i in range(0, 1024, 1):
            a_val: pl.FP32 = pl.op.tensor.get(A, [i])      # 读取A[i]
            b_val: pl.FP32 = pl.op.tensor.get(B, [i])      # 读取B[i]
            c_val: pl.FP32 = a_val + b_val                 # 加法
            pl.op.tensor.set(C, [i], c_val)                # 写入C[i]

        return C
```

### Tile切分后（优化Pass）

```python
@pl.program
class VectorAddTiled:
    @pl.function
    def add_tiled(self, A: pl.Tensor[[1024], pl.FP32],
                        B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.empty([1024], pl.FP32)

        # 外层：遍历8个tile块
        for ii in range(0, 1024, 128):
            # 内层：处理每个tile的128个元素
            for i in range(ii, ii + 128, 1):
                a_val: pl.FP32 = pl.op.tensor.get(A, [i])
                b_val: pl.FP32 = pl.op.tensor.get(B, [i])
                c_val: pl.FP32 = a_val + b_val
                pl.op.tensor.set(C, [i], c_val)

        return C
```

### 向量化版本（最优）

```python
@pl.program
class VectorAddVectorized:
    @pl.function
    def add_vectorized(self, A: pl.Tensor[[1024], pl.FP32],
                             B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        # 单个Op，无显式循环
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.add(A, B)
        return C
```

---

## 📊 三个版本对比

| 特性 | 原始版本 | Tile切分 | 向量化 |
|------|---------|---------|--------|
| 循环层数 | 1层 | 2层 | 0层 |
| Op调用 | tensor.get×2048<br>tensor.set×1024 | tensor.get×2048<br>tensor.set×1024 | tensor.add×1 |
| 缓存友好性 | 中 | 高（tile块） | 最高 |
| SIMD向量化 | 难 | 中 | 自动 |
| 性能提升 | 1x | 2-3x | 5-10x |

**优化路径**：`原始` → `Tile切分（提升缓存局部性）` → `向量化（消除循环）`

---

## 🔧 查看IR的3种方法

### 方法1：直接打印

```python
import pypto.ir as ir

print(ir.python_print(program))  # 完整调用
print(str(program))               # 简写方式
```

### 方法2：保存到文件

```python
code = ir.python_print(program)
with open("my_ir.py", "w") as f:
    f.write(code)

# 然后用编辑器查看
# code my_ir.py
```

### 方法3：PassManager自动保存

```python
from pypto.ir.pass_manager import PassManager

pm = PassManager()
pm.add_pass("tile", TilePass(128))
pm.add_pass("vectorize", VectorizePass())

pm.run_with_dump(program, output_dir="./ir_stages")

# 自动生成：
# ./ir_stages/00_input.py     - 原始IR
# ./ir_stages/01_tile.py      - Tile切分后
# ./ir_stages/02_vectorize.py - 向量化后
```

---

## 💡 关键IR节点说明

| IR节点 | 代码示例 | 作用 |
|-------|---------|------|
| **tensor.empty** | `pl.op.tensor.empty([1024], pl.FP32)` | 分配张量 |
| **tensor.get** | `pl.op.tensor.get(A, [i])` | 读取元素A[i] |
| **tensor.set** | `pl.op.tensor.set(C, [i], val)` | 写入元素C[i] |
| **tensor.add** | `pl.op.tensor.add(A, B)` | 张量加法（向量化） |
| **Add** | `a + b` | 标量加法 |
| **ForStmt** | `for i in range(0, 1024, 1)` | 循环语句 |

---

## 🆚 msgpack vs python_print

| | msgpack序列化 | python_print |
|---|---|---|
| **用途** | 持久化存储、传输 | 调试、查看、学习 |
| **格式** | 二进制（紧凑） | Python代码（可读） |
| **大小** | 小（40-60% of JSON） | 大（文本） |
| **可读性** | ❌ 不可读 | ✅ 人类可读 |
| **编辑** | ❌ 不可编辑 | ✅ 可手动编辑 |
| **指针共享** | ✅ 保留 | ✅ 显示结构 |

**关系**：
```
IR对象 ←──→ msgpack二进制 (存储)
   ↓
python_print (查看)
```

两者互补，各司其职！

---

## 📁 相关文件

### 示例代码
- **完整展示**：[add_ir_showcase.py](../pypto/examples/add_ir_showcase.py) ✅ 可直接运行
- **底层API**：[ir_visualization_example.py](../pypto/examples/ir_visualization_example.py)

### 文档
- **本文件**：[README_查看IR.md](./README_查看IR.md) - 总结
- **详细版**：[如何查看IR.md](./如何查看IR.md) - 原理和方法
- **实例版**：[如何查看IR_Add算子实例.md](./如何查看IR_Add算子实例.md) - 真实示例

### 运行示例
```bash
cd /data/g00655722/new-ir/pypto
python3 examples/add_ir_showcase.py
```

---

## ✨ 关键要点

1. ✅ **真实的Op调用** - 循环内可以看到`pl.op.tensor.get/set/add`
2. ✅ **标准Python语法** - IR是合法的Python代码
3. ✅ **类型注解完整** - `A: pl.Tensor[[1024], pl.FP32]`
4. ✅ **优化过程可视** - 对比不同Pass的输出
5. ✅ **可往返转换** - Python代码可重新解析回IR

---

## 🎓 实际应用

1. **调试Pass效果** - 对比优化前后的IR
2. **学习IR设计** - 查看标准的IR表示
3. **性能分析** - 理解不同实现的性能差异
4. **手动优化** - 修改IR后重新测试
5. **教学演示** - 展示编译器优化过程

---

## 🎉 总结

**问题**：msgpack是二进制，无法直接查看

**解决**：`ir.python_print()` 转换为Python代码

**优势**：
- 人类可读
- 真实Op
- 完整类型
- 可编辑
- 调试友好

**关系**：msgpack（存储） + python_print（查看） = 完美组合！
