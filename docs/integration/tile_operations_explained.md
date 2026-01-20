# PyPTO Tile 操作底层实现详解

本文档详细解释 `test_ffn.py` 中几个关键 pypto 操作的 C++ 底层实现，特别是如何设置不同级别的片上存储大小以及如何取数据。

## 代码示例

```python
pypto.set_vec_tile_shapes(unroll_level, hidden_size)
x_tile = pypto.view(x, shape=[unroll_level, hidden_size], offsets=[idx, 0])
w1_tile = pypto.view(w1, shape=[hidden_size, ffn_hidden_size], offsets=[0, 0])

pypto.set_cube_tile_shapes([unroll_level, unroll_level], [tile_k, tile_k], [tile_n, tile_n], True, False)
pypto.set_matrix_size([unroll_level, hidden_size, ffn_hidden_size])
```

## 1. `pypto.set_vec_tile_shapes(unroll_level, hidden_size)`

### Python 层实现
```python
# pypto_zimo/python/pypto/_controller.py:65-91
def set_vec_tile_shapes(*shapes: int):
    concrete_shapes = [
        it.concrete() if isinstance(it, SymbolicScalar) else it for it in shapes
    ]
    pypto_impl.SetScope({"vec_tile_shapes": concrete_shapes})
```

### C++ 层实现
```cpp
// pypto_zimo/framework/src/interface/operation/tile_shape.cpp:77-80
void TileShape::SetVecTile(const std::vector<int64_t> &tile) {
    vecTile = {tile};
    ConfigManagerNg::CurrentScope()->UpdateValue("vec_tile_shapes", tile);
}
```

### 作用
- **设置向量计算的 tile 形状**：用于元素级操作（如 add, mul 等）
- **存储位置**：存储在 `TileShape::Current()` 的 `vecTile` 成员中
- **使用场景**：后续的向量操作会使用这个 tile 大小来决定如何分块处理数据

### 数据流
```
Python 调用 → pypto_impl.SetScope() → ConfigManagerNg::UpdateValue() 
→ TileShape::vecTile 更新
```

## 2. `pypto.view(x, shape=[unroll_level, hidden_size], offsets=[idx, 0])`

### Python 层实现
```python
# pypto_zimo/python/pypto/operation.py:209-281
def view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, ...):
    if dtype is not None:
        return pypto_impl.View(input, dtype)
    elif valid_shape is not None:
        return pypto_impl.View(input, shape, offsets, valid_shape)
    else:
        return pypto_impl.View(input, shape, offsets)
```

### C++ 层实现
```cpp
// pypto_zimo/framework/src/interface/operation/operation_impl.cpp:946-961
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, 
            const std::vector<SymbolicScalar> &newOffsets, const void *lr) {
    // 1. 创建新的 Tensor，但共享底层存储
    Tensor result(operand.GetStorage()->Datatype(), shapes,
        "View_" + operand.GetStorage()->GetRawTensor()->GetSymbol(),
        operand.Format());
    
    // 2. 更新动态有效形状
    result.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(shapes));
    
    // 3. 添加 VIEW 操作到计算图
    auto function = Program::GetInstance().GetCurrentFunction();
    auto &op = function->AddOperation(Opcode::OP_VIEW, 
                                     {operand.GetStorage()}, 
                                     {result.GetStorage()});
    
    // 4. 计算有效形状（考虑 offsets）
    auto validShape = GetViewValidShape(operand.GetStorage()->GetDynValidShape(), 
                                        {}, newOffsets, shapes);
    result.GetStorage()->UpdateDynValidShape(validShape);
    
    // 5. 设置操作属性（包含 offsets）
    std::vector<int64_t> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, 
                                                         newOffsets, validShape));
    return result;
}
```

### 作用
- **创建 tensor 视图**：不复制数据，只是创建一个新的 tensor 对象，指向原 tensor 的一部分
- **数据访问**：通过 `offsets` 指定在原 tensor 中的起始位置
- **内存效率**：零拷贝操作，只是改变访问方式

### 数据取法
```
原 tensor: [total_tokens, hidden_size]
offsets: [idx, 0]
shape: [unroll_level, hidden_size]

实际访问的数据：
- 从原 tensor 的第 idx 行开始
- 取 unroll_level 行
- 每行取 hidden_size 列
- 数据在内存中仍然是连续的（如果原 tensor 是连续的）
```

### 底层数据访问

**重要：VIEW 操作本身不会产生数据传输，但会根据内存层级转换**

View 操作在编译过程中会被处理：

1. **VIEW 操作本身**：只是一个逻辑视图，记录偏移和形状，不复制数据
   ```cpp
   // VIEW 操作被注册为 MOVE_LOCAL 类型
   // 在 opcode.cpp 中：
   registerInfo(Opcode::OP_VIEW, OpCoreType::ANY, "VIEW", {}, {}, 
                {"VIEW", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
   ```

2. **编译时转换规则**：在 `GenerateMoveOp` Pass 中，VIEW 操作会根据输入输出的内存类型被转换

   转换逻辑在 `CreateMoveOpForView` 函数中（`generate_move_op.cpp:66-112`）：

   **规则 1：输入是 GM（全局内存）**
   ```cpp
   if (isGmInput) {
       // 情况 1.1：输出也是 GM，且消费者是特定操作（INDEX_OUTCAST 或 RESHAPE）
       if (isGmOutput && HasSpecificConsumer(op)) {
           return SUCCESS;  // 保持 VIEW，不转换
       }
       // 情况 1.2：输出不是 GM（需要加载到片上存储）
       if (!isGmOutput) {
           op.SetOpCode(Opcode::OP_COPY_IN);  // 转换为 COPY_IN（对应 TLOAD）
       }
       // 情况 1.3：输入输出都是 GM，且消费者不是特定操作
       // 保持 VIEW，不转换（同一内存层级）
   }
   ```

   **规则 2：输出是 L0A/L0B（Cube 计算单元输入）**
   ```cpp
   else if (output == MEM_L0A) {
       // 检查是否需要转置
       if (isTrans) {
           op.SetOpCode(Opcode::OP_L1_TO_L0_AT);  // L1 → L0A（转置）
       } else {
           op.SetOpCode(Opcode::OP_L1_TO_L0A);     // L1 → L0A（对应 TEXTRACT）
       }
   }
   else if (output == MEM_L0B) {
       // 类似处理 L0B
       op.SetOpCode(Opcode::OP_L1_TO_L0B or OP_L1_TO_L0_BT);
   }
   ```

   **规则 3：其他内存层级间转换**
   ```cpp
   else {
       auto from = input->GetMemoryTypeOriginal();
       auto to = output->GetMemoryTypeOriginal();
       
       // 情况 3.1：同一内存层级
       if (from == to) {
           return SUCCESS;  // 保持 VIEW，不转换，不传输数据
       }
       
       // 情况 3.2：不同内存层级，查找路径映射表
       Status status = SetOpcodeByMemPath(op, from, to);
       // 根据 platformPathMap 查找对应的 Opcode
   }
   ```

3. **内存路径映射表**（`generate_move_op.h:32-44`）

   编译器维护了一个内存路径到操作码的映射表：
   ```cpp
   const std::map<std::pair<MemoryType, MemoryType>, Opcode> platformPathMap = {
       // GM → 片上存储
       {{MEM_DEVICE_DDR, MEM_L1},  Opcode::OP_COPY_IN},   // GM → L1
       {{MEM_DEVICE_DDR, MEM_UB},  Opcode::OP_COPY_IN},   // GM → UB
       
       // L1 → L0（Cube 计算）
       {{MEM_L1, MEM_L0A},         Opcode::OP_L1_TO_L0A}, // L1 → L0A
       {{MEM_L1, MEM_L0B},         Opcode::OP_L1_TO_L0B}, // L1 → L0B
       
       // L0C → 其他（计算结果输出）
       {{MEM_L0C, MEM_DEVICE_DDR}, Opcode::OP_COPY_OUT},  // L0C → GM
       {{MEM_L0C, MEM_L1},         Opcode::OP_L0C_TO_L1}, // L0C → L1
       {{MEM_L0C, MEM_UB},         Opcode::OP_L0C_COPY_UB},// L0C → UB
       
       // UB ↔ L1
       {{MEM_UB, MEM_DEVICE_DDR},  Opcode::OP_COPY_OUT},  // UB → GM
       {{MEM_UB, MEM_L1},          Opcode::OP_UB_COPY_L1}, // UB → L1
       
       // 其他特殊路径
       {{MEM_L1, MEM_BT},          Opcode::OP_L1_TO_BT},  // L1 → Bias
       {{MEM_L1, MEM_FIX_QUANT_PRE}, Opcode::OP_L1_TO_FIX_QUANT_PRE}, // L1 → Fix
   };
   ```

4. **转换决策流程图**

   ```
   VIEW 操作
      ↓
   检查输入内存类型
      ↓
   ┌─────────────────┐
   │ 输入是 GM?      │
   └─────────────────┘
      ↓ 是          ↓ 否
   ┌─────────┐   ┌──────────────────┐
   │输出是GM?│   │ 输出是 L0A/L0B?  │
   └─────────┘   └──────────────────┘
      ↓ 是          ↓ 是              ↓ 否
   ┌─────────┐   ┌──────────┐   ┌──────────────┐
   │保持VIEW │   │L1→L0A/B  │   │ 查找路径映射 │
   │(不传输) │   │(TEXTRACT)│   │   表转换     │
   └─────────┘   └──────────┘   └──────────────┘
      ↓ 否
   ┌─────────┐
   │COPY_IN  │
   │(TLOAD)  │
   └─────────┘
   ```

5. **实际代码生成示例**

   ```cpp
   // 示例 1：VIEW 输入输出都在 GM
   // Python: x_tile = pypto.view(x, shape=[...], offsets=[idx, 0])
   // 转换后：保持 VIEW，代码生成时只是计算偏移
   __gm__ float* x_ptr = x + idx * hidden_size;  // 只计算地址，不传输数据
   
   // 示例 2：VIEW 从 GM 到 L1
   // Python: x_tile = pypto.view(x, ...)  // x 在 GM，x_tile 需要 L1
   // 转换后：OP_COPY_IN，代码生成时生成 TLOAD
   TLOAD(l1_tile, gm_tensor + offset);  // 实际的数据传输：GM → L1
   
   // 示例 3：VIEW 从 L1 到 L0A
   // Python: a_tile = pypto.view(a_l1, ...)  // a_l1 在 L1，a_tile 需要 L0A
   // 转换后：OP_L1_TO_L0A，代码生成时生成 TEXTRACT
   TEXTRACT(l0a_tile, l1_tile, offset0, offset1);  // 实际的数据传输：L1 → L0A
   ```

**总结 - 什么时候涉及数据传输，什么时候不涉及**：

| 输入内存类型 | 输出内存类型 | 转换结果 | 是否传输数据 |
|------------|------------|---------|------------|
| GM | GM | 保持 VIEW | ❌ 不传输（只计算偏移） |
| GM | L1/UB | OP_COPY_IN | ✅ 传输（TLOAD: GM → L1/UB） |
| L1 | L0A/L0B | OP_L1_TO_L0A/B | ✅ 传输（TEXTRACT: L1 → L0） |
| L1 | L1 | 保持 VIEW | ❌ 不传输（同一层级） |
| UB | UB | 保持 VIEW | ❌ 不传输（同一层级） |
| L0C | GM | OP_COPY_OUT | ✅ 传输（TSTORE: L0C → GM） |
| 其他 | 其他（相同） | 保持 VIEW | ❌ 不传输（同一层级） |
| 其他 | 其他（不同） | 查表转换 | ✅ 传输（根据路径映射） |

**关键判断**：
- **同一内存层级**（`from == to`）：保持 VIEW，不传输数据
- **跨越内存层级**：转换为相应的数据传输操作（COPY_IN, L1_TO_L0A, COPY_OUT 等）
- **特殊情况**：GM → GM 且消费者是特定操作时，也保持 VIEW

## 3. `pypto.set_cube_tile_shapes([m0, m1], [k0, k1], [n0, n1], enable_multi_data_load, enable_split_k)`

### Python 层实现
```python
# pypto_zimo/python/pypto/_controller.py:117-161
def set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], 
                         enable_multi_data_load: bool = False, 
                         enable_split_k: bool = False):
    cube_tile = CubeTile(m, k, n, enable_multi_data_load, enable_split_k)
    pypto_impl.SetScope({"cube_tile_shapes": cube_tile.impl()})
```

### C++ 层实现
```cpp
// pypto_zimo/framework/src/interface/operation/tile_shape.cpp:87-97
void TileShape::SetCubeTile(const std::array<int64_t, MAX_M_DIM_SIZE> &m,
                            const std::array<int64_t, MAX_K_DIM_SIZE> &k,
                            const std::array<int64_t, MAX_N_DIM_SIZE> &n,
                            bool enableMultiDataLoad, bool enableSplitK) {
    auto nk = k;
    if (nk[2] == 0) {
        nk[2] = nk[1];  // 如果 k[2] 为 0，则使用 k[1]
    }
    cubeTile = {m, nk, n, enableMultiDataLoad, enableSplitK};
    ConfigManagerNg::CurrentScope()->UpdateValue("cube_tile_shapes", cubeTile);
}
```

### 片上存储级别说明

#### L0 Cache（最接近计算单元）
- **m[0], k[0], n[0]**：L0 级别的 tile 大小
- **位置**：直接在 Cube 计算单元附近
- **大小**：通常较小（如 128x64x256）
- **用途**：直接参与矩阵乘法计算

#### L1 Cache（中间缓存）
- **m[1], k[1], n[1]**：L1 级别的 tile 大小
- **位置**：在 L0 和全局内存（GM）之间
- **大小**：通常比 L0 大（如 1536x6144x1024）
- **用途**：缓存从 GM 加载的数据，然后分块传输到 L0

### 数据流示例

以 `set_cube_tile_shapes([unroll_level, unroll_level], [tile_k, tile_k], [tile_n, tile_n], True, False)` 为例：

```
全局内存 (GM)
    ↓ TLOAD (GM → L1)
L1 Cache: [unroll_level, tile_k]  (A矩阵)
L1 Cache: [tile_k, tile_n]         (B矩阵)
    ↓ TEXTRACT (L1 → L0)
L0A: [unroll_level, tile_k]       (A矩阵分块)
L0B: [tile_k, tile_n]              (B矩阵分块)
    ↓ TMATMUL (L0 计算)
L0C: [unroll_level, tile_n]        (结果)
    ↓ TSTORE (L0C → GM)
全局内存 (GM)
```

### 参数说明

1. **m[0], m[1]**：M 维度的 tile 大小
   - `m[0]`：L0 级别（如 128）
   - `m[1]`：L1 级别（如 1536）

2. **k[0], k[1], k[2]**：K 维度的 tile 大小
   - `k[0]`：L0 级别（如 64）
   - `k[1]`：A 矩阵的 L1 级别（如 6144）
   - `k[2]`：B 矩阵的 L1 级别（如 6144）

3. **n[0], n[1]**：N 维度的 tile 大小
   - `n[0]`：L0 级别（如 256）
   - `n[1]`：L1 级别（如 1024）

4. **enable_multi_data_load**：是否启用多数据加载
   - `True`：L1 → L0 时可以使用多数据加载优化

5. **enable_split_k**：是否启用 K 维度切分
   - `True`：在 GM 中累加部分结果

### 在 cube_operation_impl.cpp 中的使用

```cpp
// pypto_zimo/framework/src/interface/operation/cube_operation_impl.cpp:954-1005
void SetMatmulTileInfo(const TileShape &tileShape, ...) {
    auto &cubeTile = tileShape.GetCubeTile();
    
    // 提取 L0 和 L1 的 tile 大小
    tileInfo.tileML0 = cubeTile.m[0];      // L0 M 维度
    tileInfo.tileML1 = cubeTile.m[1];      // L1 M 维度
    tileInfo.tileNL0 = cubeTile.n[0];      // L0 N 维度
    tileInfo.tileNL1 = cubeTile.n[1];      // L1 N 维度
    tileInfo.tileKL0 = cubeTile.k[0];      // L0 K 维度
    tileInfo.tileKAL1 = cubeTile.k[1];     // L1 A 矩阵 K 维度
    tileInfo.tileKBL1 = cubeTile.k[2];     // L1 B 矩阵 K 维度
}
```

这些 tile 信息会在构建计算图时使用，决定：
- 如何从 GM 加载数据到 L1（TLOAD）
- 如何从 L1 提取数据到 L0（TEXTRACT）
- 如何进行矩阵乘法（TMATMUL）

## 4. `pypto.set_matrix_size([unroll_level, hidden_size, ffn_hidden_size])`

### C++ 层实现
```cpp
// pypto_zimo/framework/src/interface/operation/tile_shape.cpp:99-102
void TileShape::SetMatrixSize(const std::vector<int64_t> &size) {
    this->matrixSize = size;
    ConfigManagerNg::CurrentScope()->UpdateValue("matrix_size", size);
}
```

### 作用
- **设置矩阵的实际大小**：用于验证和优化
- **格式**：`[M, K, N]`，表示矩阵乘法的维度
- **用途**：
  - 验证 tile 大小是否合理
  - 优化内存访问模式
  - 计算循环次数

## 数据加载过程详解

### 完整的数据流

以 `x_tile` 和 `w1_tile` 的矩阵乘法为例：

```
1. View 操作（Python/C++）
   - x_tile: 从 x[idx:idx+unroll_level, :] 创建视图
   - w1_tile: 从 w1[:, :] 创建视图（完整矩阵）
   - 不复制数据，只是记录偏移和形状

2. 设置 Cube Tile（Python/C++）
   - L0: [unroll_level, tile_k, tile_n]
   - L1: [unroll_level, tile_k, tile_n]

3. 构建计算图（C++）
   - 根据 tile 大小计算循环次数
   - 生成 TLOAD, TEXTRACT, TMATMUL, TSTORE 操作

4. 代码生成（C++）
   - 生成实际的 kernel 代码
   - 包含内存地址计算、循环展开等

5. 执行（NPU）
   - TLOAD: GM → L1，加载 [unroll_level, tile_k] 和 [tile_k, tile_n]
   - TEXTRACT: L1 → L0，提取小块到 L0
   - TMATMUL: L0 中计算
   - TSTORE: L0C → GM，写回结果
```

### 内存地址计算

在 `cube_pto.h` 中的 TLoad 操作：

```cpp
// pypto_zimo/framework/src/interface/tileop/cube/cube_pto.h:54-97
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoad(T &dst, U &src, const Coord &coord, const int64_t &curH, const int64_t &curW) {
    // 1. 获取偏移
    uint16_t offset0 = coord.GetValue();
    uint16_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    
    // 2. 计算全局内存偏移
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    
    // 3. 创建 GlobalTensor（指向 GM 中的特定位置）
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset), ...);
    
    // 4. 创建 L1 Tile（目标位置）
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    
    // 5. 执行加载：GM → L1
    pto::TLOAD(dstL1, src0Global);
}
```

## 总结

1. **set_vec_tile_shapes**：设置向量操作的 tile 大小，存储在配置中
2. **view**：创建 tensor 视图，零拷贝，通过 offsets 访问原数据的一部分
3. **set_cube_tile_shapes**：设置矩阵乘法的 L0 和 L1 tile 大小，控制数据在片上存储的分布
4. **set_matrix_size**：设置矩阵实际大小，用于验证和优化

这些操作共同决定了：
- **数据如何从全局内存加载到片上存储**
- **数据如何在 L1 和 L0 之间传输**
- **计算如何进行分块处理**

通过合理设置这些参数，可以优化内存访问模式，提高计算效率。

