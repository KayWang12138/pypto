# TileOP 开发准则

## 一、概述

TileOP 是 PyPTO 框架中最底层的计算实现单元，直接调用 PTO 指令执行具体的算子计算。TileOP 位于整个调用链路的最末端，由 Codegen 层生成的代码调用。

### 架构位置

```
Python前端层 → Operation接口层 → Function管理层 → Codegen代码生成层 → TileOp执行层 → PTO指令层
```

## 二、核心设计模式

### 2.1 三层实现模式

TileOP 采用标准的三层实现模式：

```cpp
// 第一层：具体计算实现（ComputeImpl）
template <BinaryOp op, TileOp::BroadcastOperand operand, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    // 直接调用 PTO 指令
    PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2, n3);
}

// 第二层：通用计算入口（Compute）
template <BinaryOp op, TileOp::BroadcastOperand operand, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    // 处理静态/动态布局
    // 循环切分处理
    // 调用 ComputeImpl
}

// 第三层：用户接口（宏定义封装）
#define OP_TILE_OP_ADD TAdd
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD, operand, LastUse>(dst, src0, src1);
}
```

### 2.2 关键设计原则

1. **编译期优化**：使用模板和 constexpr 在编译期计算 Shape、Stride 等信息
2. **类型安全**：使用模板参数约束输入输出类型
3. **广播支持**：通过 `BroadcastOperand` 枚举支持左右操作数广播
4. **LastUse 优化**：通过 `LastUse` 参数支持数据流优化

## 三、开发规范

### 3.1 文件组织

```
interface/tileop/vector/
├── binary.h          # 二元运算（Add, Sub, Mul, Div 等）
├── unary.h           # 一元运算（Exp, Sqrt, Abs 等）
├── reduce.h          # 规约运算（Sum, Max, Min 等）
├── permute.h         # 维度变换
├── gather.h          # 数据收集
└── pto_tile.h        # PTO Tile 封装工具
```

### 3.2 命名规范

| 类型 | 命名规则 | 示例 |
|------|----------|------|
| TileOP 函数 | T + 操作名 | `TAdd`, `TSub`, `TMul` |
| 宏定义 | OP_TILE_OP_ + 操作名 | `OP_TILE_OP_ADD` |
| 计算实现 | 操作名 + ComputeImpl | `BinaryComputeImpl` |
| 计算入口 | 操作名 + Compute | `BinaryCompute` |
| 枚举类型 | 操作名 + Op | `BinaryOp`, `UnaryOp`, `ReduceOp` |

### 3.3 必须包含的头文件

```cpp
#include "pto_tile.h"           // PTO Tile 封装
#include "utils/layout.h"       // Layout 工具
#include "utils/tile_tensor.h"  // TileTensor 定义
```

## 四、开发步骤

### 4.1 定义操作类型枚举

```cpp
// 在 pto_tile.h 或单独的头文件中定义
enum class BinaryOp {
    ADD,
    SUB,
    MUL,
    DIV,
    // ... 其他操作
};
```

### 4.2 实现 ComputeImpl 函数

```cpp
template <BinaryOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2, n3);
    } else if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TSUB(dst, src0, src1), n1, n2, n3);
    }
    // ... 其他操作
}
```

### 4.3 实现 Compute 函数

```cpp
template <BinaryOp op, TileOp::BroadcastOperand operand, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    // 1. 检查是否为静态连续布局
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        
        BinaryComputeImpl<op, operand, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }
    
    // 2. 动态布局处理：获取 Shape 信息
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    
    // 3. 创建 PtoTile 对象
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    
    // 4. 循环处理每个 Tile
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                BinaryComputeImpl<op, operand, LastUse>(dstTile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}
```

### 4.4 定义用户接口

```cpp
#define OP_TILE_OP_ADD TAdd
template <typename LastUse = LastUse3Dim<0, 0, 0>, 
          TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, 
          typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD, operand, LastUse>(dst, src0, src1);
}
```

## 五、关键工具类

### 5.1 PtoTile

用于将 TileTensor 转换为 PTO 可用的 Tile 对象：

```cpp
template <typename T, pto::BLayout Layout = pto::BLayout::RowMajor, bool Mergeable = false>
class PtoTile {
public:
    using Type = pto::Tile<pto::TileType::Vec, Dtype, tileH, tileW, Layout, validH, validW>;
    
    // 从地址构造
    __aicore__ inline PtoTile(const uint64_t &addr);
    
    // 从动态 Shape 构造
    __aicore__ inline PtoTile(const int &h, const int &w);
    
    // 从 TileTensor 构造
    __aicore__ inline PtoTile(const T &tensor);
    
    // 获取 PTO Tile 数据
    __aicore__ inline Type &Data();
    
    // 分配地址
    __aicore__ inline void Assign(uint64_t addr);
};
```

### 5.2 PtoGlobal

用于处理全局内存（GM）张量：

```cpp
template <typename T, typename Shape, typename Stride, bool need_mask = false>
class PtoGlobal {
public:
    using Type = pto::GlobalTensor<Dtype, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    
    __aicore__ inline PtoGlobal(__gm__ typename T::Type *addr, const Shape &shape, const Stride &stride);
    __aicore__ inline void Assign(__gm__ typename T::Type *addr);
    inline Type &Data();
};
```

### 5.3 TileOffset

用于计算 Tile 偏移量：

```cpp
template <typename T>
__aicore__ inline size_t GenTileOffset(const T &tensor, const TileOffset &offsets) {
    const auto layout = tensor.GetLayout();
    size_t offset = Std::get<DIM_1ST>(offsets) * layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    offset += Std::get<DIM_2ND>(offsets) * layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    offset += Std::get<DIM_3RD>(offsets) * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    return offset;
}
```

## 六、广播操作处理

### 6.1 广播类型定义

```cpp
enum class BroadcastOperand {
    NONE,           // 无广播
    LEFT_OPERAND,   // 左操作数广播
    RIGHT_OPERAND   // 右操作数广播
};
```

### 6.2 广播布局选择

```cpp
// 根据广播类型选择布局
using Src0PtoTile = typename std::conditional<
    (src0TileW == 1 && operand == TileOp::BroadcastOperand::LEFT_OPERAND),
    PtoTile<T1, pto::BLayout::ColMajor, true>, 
    PtoTile<T1, pto::BLayout::RowMajor, true>
>::type;
```

## 七、同步与流水线

### 7.1 同步屏障

在需要同步时使用 `pipe_barrier`：

```cpp
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
```

### 7.2 事件标志

使用事件标志进行流水线同步：

```cpp
set_flag(PIPE_V, PIPE_S, EVENT_ID7);
wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
```

## 八、特殊场景处理

### 8.1 需要临时空间的操作

```cpp
template <typename T0, typename T1, typename T2>
TILEOP void TRowSumSingle(T0 dst, T1 src, T2 tmp) {
    // tmp 作为临时空间传入
    ReduceLastAxisCompute<ReduceOp::SUM, LastUse>(dst, src, tmp);
}
```

### 8.2 复杂操作的组合

```cpp
template <typename LastUse, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2>
TILEOP void TMod(T0 dst, T1 src0, T2 src1) {
    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
        pto::TFMOD(dst, src0, src1);
    } else {
        // 组合多个操作实现 MOD
        pto::TROWEXPANDDIV(dst, src0, src1);
        pipe_barrier(PIPE_V);
        pto::TCVT(dst, dst, pto::RoundMode::CAST_TRUNC);
        pipe_barrier(PIPE_V);
        pto::TROWEXPANDMUL(dst, dst, src1);
        pipe_barrier(PIPE_V);
        pto::TROWEXPANDSUB(dst, src0, dst);
    }
}
```

## 九、调试与验证

### 9.1 编译期检查

使用 `static_assert` 进行编译期验证：

```cpp
static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::GM);
static_assert(T::IsStaticLayout(), "Only valid for static layout tile tensor.");
```

### 9.2 运行时断言

```cpp
assert(3 * totalBytes <= ubSize && "UB memory size insufficient for permute operation");
```

## 十、最佳实践

1. **优先使用编译期计算**：尽可能使用 `constexpr` 和模板参数
2. **减少运行时分支**：使用 `if constexpr` 替代运行时 `if`
3. **合理使用同步**：仅在必要时添加同步屏障
4. **内存对齐**：确保内存访问对齐以提高性能
5. **复用现有工具**：使用 `PtoTile`、`PtoGlobal` 等工具类

## 十一、示例：实现新的 TileOP

以下是一个完整的 TileOP 实现示例：

```cpp
// 1. 定义操作类型
enum class MyOp {
    CUSTOM_ADD,
    CUSTOM_SUB
};

// 2. 实现 ComputeImpl
template <MyOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void MyBinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    
    if constexpr (op == MyOp::CUSTOM_ADD) {
        PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2);
    }
}

// 3. 实现 Compute
template <MyOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void MyBinaryCompute(T0 dst, T1 src0, T2 src1) {
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        
        MyBinaryComputeImpl<op, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }
    
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    
    for (LoopVar i = 0; i < shape0; ++i) {
        for (LoopVar j = 0; j < shape1; ++j) {
            auto offsets = TileOffset(i, j);
            dstTile.Assign(dst, offsets);
            src0Tile.Assign(src0, offsets);
            src1Tile.Assign(src1, offsets);
            MyBinaryComputeImpl<op, LastUse>(dstTile.Data(), src0Tile.Data(), src1Tile.Data());
        }
    }
}

// 4. 定义用户接口
#define OP_TILE_OP_CUSTOM_ADD TCustomAdd
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T2>
TILEOP void TCustomAdd(T0 dst, T1 src0, T2 src1) {
    MyBinaryCompute<MyOp::CUSTOM_ADD, LastUse>(dst, src0, src1);
}
```

## 十二、参考文件

- [binary.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/tileop/vector/binary.h) - 二元运算实现
- [unary.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/tileop/vector/unary.h) - 一元运算实现
- [reduce.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/tileop/vector/reduce.h) - 规约运算实现
- [pto_tile.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/tileop/vector/pto_tile.h) - PTO Tile 工具类
