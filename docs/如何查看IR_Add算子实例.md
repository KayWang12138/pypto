# 普通用户如何在Python侧查看IR - Add算子实例

## 背景

新IR项目使用msgpack-c进行序列化，保存的是二进制格式（不可读）。但提供了 **`python_print()`** 功能，可以将IR转换为标准Python代码。

---

## 真实示例：向量Add算子 (C[i] = A[i] + B[i])

### 示例1：原始IR - 逐元素循环

```python
# pypto.program: VectorAdd
import pypto.language as pl

@pl.program
class VectorAdd:
    @pl.function
    def add(self,
            A: pl.Tensor[[1024], pl.FP32],
            B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        # 分配输出张量
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.empty([1024], pl.FP32)

        # 逐元素遍历
        for i in range(0, 1024, 1):
            # 读取输入元素
            a_val: pl.FP32 = pl.op.tensor.get(A, [i])
            b_val: pl.FP32 = pl.op.tensor.get(B, [i])

            # 执行加法运算
            c_val: pl.FP32 = a_val + b_val

            # 写入结果
            pl.op.tensor.set(C, [i], c_val)

        return C
```

**关键IR节点**：
- `pl.op.tensor.empty([1024], pl.FP32)` - 分配张量
- `pl.op.tensor.get(A, [i])` - 读取元素（循环内调用1024次）
- `a_val + b_val` - 标量加法（IR节点：Add）
- `pl.op.tensor.set(C, [i], c_val)` - 写入元素
- `for i in range(0, 1024, 1)` - ForStmt循环

---

### 示例2：Tile切分后的IR

```python
# pypto.program: VectorAddTiled
import pypto.language as pl

@pl.program
class VectorAddTiled:
    @pl.function
    def add_tiled(self,
                  A: pl.Tensor[[1024], pl.FP32],
                  B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.empty([1024], pl.FP32)

        # 外层循环：遍历tile块 (8个块)
        for ii in range(0, 1024, 128):
            # 内层循环：处理每个tile块的128个元素
            for i in range(ii, ii + 128, 1):
                a_val: pl.FP32 = pl.op.tensor.get(A, [i])
                b_val: pl.FP32 = pl.op.tensor.get(B, [i])
                c_val: pl.FP32 = a_val + b_val
                pl.op.tensor.set(C, [i], c_val)

        return C
```

**Tile切分改进**：
- 循环结构：`for ii (8次) → for i (128次/tile)`
- 数据局部性：tile块内连续访问128个元素
- 缓存友好：小tile块可放入L1/L2缓存
- 循环边界：ii取值0, 128, 256, ..., 896（共8个tile）

---

### 示例3：向量化版本（最优）

```python
# pypto.program: VectorAddVectorized
import pypto.language as pl

@pl.program
class VectorAddVectorized:
    @pl.function
    def add_vectorized(self,
                       A: pl.Tensor[[1024], pl.FP32],
                       B: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
        # 单个tensor操作，替代1024次循环
        C: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.add(A, B)
        return C
```

**向量化优势**：
- 无显式循环
- 单个Op调用：`pl.op.tensor.add(A, B)`
- 自动SIMD向量化（AVX, NEON等）
- 代码最简洁（仅3行）

---

## 三个版本对比

| 特性 | 原始版本 | Tile切分版本 | 向量化版本 |
|------|---------|------------|----------|
| 循环层数 | 1层 | 2层 | 0层 |
| 总迭代次数 | 1024 | 8×128=1024 | 0 |
| tensor.get次数 | 2048 (A+B) | 2048 (A+B) | 0 |
| 标量add次数 | 1024 | 1024 | 0 |
| tensor.add次数 | 0 | 0 | 1 (向量) |
| 数据局部性 | 顺序访问 | tile块内密集 | 整体向量 |
| 缓存友好性 | 中 | 高 | 最高 |
| SIMD向量化 | 难 | 中 (tile内) | 易 (自动) |
| 性能 (相对) | 1x | 2-3x | 5-10x |

---

## 如何查看IR（3种方法）

### 方法1：直接打印到终端

```python
import pypto.ir as ir

# 构建或加载IR
program = VectorAdd  # 你的Program对象

# 打印查看
print(ir.python_print(program))

# 简写方式
print(str(program))
```

### 方法2：保存到文件

```python
code = ir.python_print(program)
with open("vector_add_ir.py", "w") as f:
    f.write(code)

# 然后用编辑器查看
# code vector_add_ir.py
```

### 方法3：PassManager自动保存每个Pass的IR

```python
from pypto.ir.pass_manager import PassManager

pm = PassManager()
pm.add_pass("tile_transform", TilePass(tile_size=128))
pm.add_pass("vectorize", VectorizePass())

# 自动保存每个阶段的IR
result = pm.run_with_dump(
    program,
    output_dir="./ir_stages"
)

# 生成的文件：
# ./ir_stages/00_input.py          - 原始IR
# ./ir_stages/01_tile_transform.py - Tile切分后
# ./ir_stages/02_vectorize.py      - 向量化后
```

每个文件都是标准Python代码，可以直接用编辑器查看！

---

## msgpack vs python_print

| | msgpack序列化 | python_print |
|---|---|---|
| **用途** | 持久化存储、跨进程通信 | 调试、查看、理解 |
| **格式** | 二进制（不可读，但高效） | Python代码（人类可读） |
| **文件大小** | 小（40-60% of JSON） | 大（文本） |
| **可读性** | ❌ 不可读 | ✅ 人类可读 |
| **指针共享** | ✅ 保留 | ✅ 显示结构 |
| **编辑** | ❌ 不可编辑 | ✅ 可手动编辑 |
| **性能** | ✅ 高（序列化快4x） | ⚠️ 中等（文本生成） |

**两者关系**：
```
IR对象 ←────→ msgpack二进制 (存储)
   ↓
python_print输出 (查看)
```

msgpack负责**存储**，python_print负责**查看**，两者互补！

---

## 运行示例

```bash
cd /data/g00655722/new-ir/pypto

# 运行Add算子IR展示示例
python3 examples/add_ir_showcase.py
```

示例位置：
- 完整示例：[add_ir_showcase.py](../pypto/examples/add_ir_showcase.py)
- 文档说明：本文件

---

## 关键要点

1. ✅ **真实的Op调用** - 循环内可以看到`pl.op.tensor.get/set/add`等真实操作
2. ✅ **标准Python语法** - IR是合法的Python代码，可用任何编辑器查看
3. ✅ **类型注解完整** - `A: pl.Tensor[[1024], pl.FP32]`清晰明了
4. ✅ **可往返转换** - `IR对象 → python_print → Parser → IR对象`
5. ✅ **优化过程可视** - 对比不同Pass的输出，理解编译器变换

---

## 实际应用场景

1. **调试Pass效果** - 对比优化前后的IR差异
2. **学习IR设计** - 查看标准的IR表示
3. **性能分析** - 理解不同实现的性能差异
4. **手动优化** - 修改IR后重新编译测试
5. **教学演示** - 展示编译器优化过程

---

## 总结

**问题**：msgpack序列化的IR是二进制格式，普通用户无法直接查看

**解决**：使用 `ir.python_print(program)` 将IR转换为标准Python代码

**优势**：
- ✅ 人类可读 - Python语法，易于理解
- ✅ 真实Op - 看到具体的tensor.get/set/add操作
- ✅ 完整类型 - 带类型注解，信息丰富
- ✅ 可编辑 - 可修改后重新解析
- ✅ 调试友好 - 任何编辑器都能查看

**与msgpack的关系**：
- msgpack负责高效存储（二进制）
- python_print负责易读查看（文本）
- 两者互补，既保证性能又保证可用性！
