# 普通用户如何在Python侧查看IR

## 问题背景

data/g00655722/new-ir/pypto这个目录里实现的新的IR，使用了msgpack-c库进行IR的序列化和反序列化，保存数据格式是二进制，二进制对用户不可读。**那么普通用户在python测如何查看具体IR？**

## 解决方案：python_print()

新IR项目提供了完善的 **`python_print()`** 功能，可以将二进制序列化的IR转换为**标准Python代码**，非常易读易懂。

---

## 实际示例：matmul + tile切分

### 场景说明
- **前端**：构建一个简单的matmul算子 (A[128,256] @ B[256,512] = C[128,512])
- **Pass优化**：应用tile切分Pass (tile size: i=32, j=64, k=32)
- **查看IR**：使用`python_print()`查看每个阶段的IR

### 原始IR（未优化，三重循环）

```python
# pypto.program: MatmulOriginal
import pypto.language as pl

@pl.program
class MatmulOriginal:
    @pl.function
    def matmul_original(self,
                       A: pl.Tensor[[128, 256], pl.FP32],
                       B: pl.Tensor[[256, 512], pl.FP32]) -> pl.Tensor[[128, 512], pl.FP32]:
        for i in range(0, 128, 1):      # 外层：i维度 (128次)
            for j in range(0, 512, 1):  # 中层：j维度 (512次)
                for k in range(0, 256, 1):  # 内层：k维度 (256次)
                    acc: pl.FP32 = result
```

**特点**：
- 三重循环嵌套: `i(128) -> j(512) -> k(256)`
- 总迭代次数: 128 × 512 × 256 = 16,777,216 次
- 数据局部性: 较差（大范围跳跃访问）
- 缓存友好性: 低

---

### Tile切分后的IR（六重循环）

```python
# pypto.program: MatmulTiled
import pypto.language as pl

@pl.program
class MatmulTiled:
    @pl.function
    def matmul_tiled(self,
                     A: pl.Tensor[[128, 256], pl.FP32],
                     B: pl.Tensor[[256, 512], pl.FP32]) -> pl.Tensor[[128, 512], pl.FP32]:
        # 外层循环：遍历tile块
        for ii in range(0, 128, 32):      # 4个tile块 (128/32)
            for jj in range(0, 512, 64):  # 8个tile块 (512/64)
                for kk in range(0, 256, 32):  # 8个tile块 (256/32)
                    # 内层循环：tile块内的计算
                    for i in range(ii, ii + 32, 1):  # tile内32次
                        for j in range(jj, jj + 64, 1):  # tile内64次
                            for k in range(kk, kk + 32, 1):  # tile内32次
                                acc: pl.FP32 = result
```

**特点**：
- 六重循环嵌套:
  - 外层: `ii(4) -> jj(8) -> kk(8)` (遍历tile块)
  - 内层: `i(32) -> j(64) -> k(32)` (tile块内计算)
- 总迭代次数: 相同 (4×8×8) × (32×64×32) = 16,777,216 次
- 数据局部性: 显著改善（小tile块内密集访问）
- 缓存友好性: 高（tile块可放入L1/L2缓存）

**性能优势**：
- ✅ 提高缓存命中率
- ✅ 减少内存带宽压力
- ✅ 更好的向量化机会
- ✅ 便于后续并行化

---

## 普通用户如何查看IR（3种方法）

### 方法1：直接打印到终端

```python
import pypto.ir as ir

# 构建或加载IR
program = build_matmul_ir()  # 或从文件反序列化

# 直接打印查看
print(ir.python_print(program))

# 简写方式（推荐）
print(str(program))
```

### 方法2：保存到文件

```python
import pypto.ir as ir

# 生成Python代码
code = ir.python_print(program)

# 保存到文件
with open("my_matmul_ir.py", "w") as f:
    f.write(code)

# 然后用任何编辑器查看
# code my_matmul_ir.py
# vim my_matmul_ir.py
```

### 方法3：使用PassManager自动保存每个Pass的IR

```python
from pypto.ir.pass_manager import PassManager

pm = PassManager()
pm.add_pass("tile_transform", tile_pass)
pm.add_pass("loop_fusion", fusion_pass)
pm.add_pass("vectorize", vectorize_pass)

# 运行Pass并自动保存每个阶段的IR
output = pm.run_with_dump(
    program,
    output_dir="./ir_stages",
    prefix="pl"
)

# 自动生成：
# ./ir_stages/00_input.py            - 输入IR
# ./ir_stages/01_tile_transform.py   - tile切分后
# ./ir_stages/02_loop_fusion.py      - 循环融合后
# ./ir_stages/03_vectorize.py        - 向量化后
```

每个文件都是标准Python代码，可以直接用编辑器打开查看！

---

## IR格式特点

### 易读性 ✅
- 标准Python语法，不是伪代码
- 带完整类型注解：`A: pl.Tensor[[128, 256], pl.FP32]`
- 清晰的循环结构：`for i in range(0, 128, 1):`

### 完整性 ✅
- 包含所有IR节点信息（表达式、语句、函数、Program）
- 保留类型信息、循环边界、操作细节

### 可编辑性 ✅
- 可以手动修改后重新解析
- 支持往返转换：`IR对象 -> Python代码 -> IR对象`

### 调试友好性 ✅
- 可以用任何文本编辑器查看
- 便于理解IR结构和优化效果
- 适合学习和教学

---

## msgpack vs python_print

| | msgpack序列化 | python_print |
|---|---|---|
| **用途** | 持久化存储、跨进程通信 | 调试、查看、理解 |
| **格式** | 二进制（紧凑） | Python代码（可读） |
| **文件大小** | 小（40-60% of JSON） | 大（文本） |
| **可读性** | ❌ 不可读 | ✅ 人类可读 |
| **指针共享** | ✅ 保留 | ✅ 显示引用结构 |
| **编辑** | ❌ 不可编辑 | ✅ 可以手动编辑 |
| **性能** | ✅ 高（序列化快4x） | ⚠️ 中等（文本生成） |
| **适用场景** | 生产环境持久化 | 开发调试和学习 |

**两者关系**：
- 相互转换，各司其职
- msgpack用于**存储**（二进制，高效）
- python_print用于**查看**（文本，易读）

---

## 完整示例代码

示例文件位置：
```
/data/g00655722/new-ir/pypto/examples/ir_visualization_example.py
```

生成的IR文件：
```
/data/g00655722/new-ir/pypto/examples/ir_output/
├── 01_original_matmul.py    # 原始IR（三重循环）
└── 02_tiled_matmul.py        # Tile切分后（六重循环）
```

运行示例：
```bash
cd /data/g00655722/new-ir/pypto
python3 examples/ir_visualization_example.py
```

---

## 总结

**问题**：msgpack序列化的IR是二进制格式，普通用户无法直接查看

**解决**：使用`ir.python_print(program)`将IR转换为标准Python代码

**优势**：
1. ✅ 人类可读 - Python语法，易于理解
2. ✅ 类型完整 - 带类型注解，信息丰富
3. ✅ 可编辑 - 可修改后重新解析
4. ✅ 调试友好 - 任何编辑器都能查看
5. ✅ 往返一致 - 打印的代码可重新解析回IR

**实际应用**：
- 调试IR Pass的效果
- 理解编译器优化过程
- 学习IR结构和设计
- 手动编辑IR进行实验

**与msgpack的关系**：
- msgpack负责高效存储（二进制）
- python_print负责易读查看（文本）
- 两者互补，既保证性能又保证可用性！
