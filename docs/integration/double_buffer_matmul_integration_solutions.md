# Double Buffer Matmul Kernel 集成方案

## 问题背景

`gemm_performance_kernel.cpp` 是一个高性能的 double buffer matmul kernel，包含精细的流水线同步控制（SetFlag/WaitFlag）。但 pypto 的 `loop_unroll` 会将 Python 代码展开成 graph，然后经过 pass optimization 和 codegen，这样就无法在 Python 端直接写成可控制的底层 kernel。

## 解决方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **方案1: torch.ops 注册** | 完全绕过 graph 化，保留所有优化 | 需要手动管理内存和参数 | 需要精确控制的场景 |
| **方案2: C++ 后端自动选择** | 自动化，用户无感知 | 需要修改框架代码 | 通用场景，作为 fallback |
| **方案3: 混合模式** | 灵活性高，兼顾性能和易用性 | 实现复杂度较高 | 生产环境推荐 |

---

## 方案1: 通过 torch.ops 直接调用（推荐用于精确控制）

### 1.1 创建 Host 端算子实现

创建文件 `pto-isa/demos/baseline/gemm_performance/csrc/host/gemm_performance_op.cpp`:

```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include "utils.h"
#include "aclrtlaunch_gemm_performance.h"  // 构建系统生成的头文件

at::Tensor gemm_performance(
    const at::Tensor &src0,
    const at::Tensor &src1,
    uint32_t blockDim,
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN,
    uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN
) {
    // 分配输出 tensor
    auto options = torch::TensorOptions()
        .dtype(src0.dtype())
        .device(src0.device());
    at::Tensor out = torch::empty({m, n}, options);

    // 准备 kernel 参数
    uint8_t *out_ptr = reinterpret_cast<uint8_t *>(out.data_ptr());
    uint8_t *src0_ptr = reinterpret_cast<uint8_t *>(src0.data_ptr());
    uint8_t *src1_ptr = reinterpret_cast<uint8_t *>(src1.data_ptr());

    // 获取 stream
    auto stream = at_npu::native::getCurrentNPUStream();

    // 启动 kernel
    EXEC_KERNEL_CMD(
        LaunchGEMME2E_Templated<uint16_t, blockDim, m, k, n, 
            singleCoreM, singleCoreK, singleCoreN, 
            baseM, baseK, baseN, stepM, stepKa, stepKb, stepN>,
        out_ptr, src0_ptr, src1_ptr, stream
    );

    return out;
}

// 注册算子 schema
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("gemm_performance(Tensor src0, Tensor src1, int blockDim, "
          "int m, int k, int n, "
          "int singleCoreM, int singleCoreK, int singleCoreN, "
          "int baseM, int baseK, int baseN, "
          "int stepM, int stepKa, int stepKb, int stepN) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, CPU, m) {
    m.impl("gemm_performance", torch::CppFunction::makeFromBoxedFunction<&gemm_performance>());
}

TORCH_LIBRARY_IMPL(npu, NPU, m) {
    m.impl("gemm_performance", torch::CppFunction::makeFromBoxedFunction<&gemm_performance>());
}
```

### 1.2 Python 端使用

在 `test_ffn.py` 中使用：

```python
import torch
import torch_npu

@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_with_double_buffer(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    out: pypto.Tensor,
    tile_n: int,
    tile_k: int,
    task_id: int = 0,
    hidden_size: int = 6144,
    ffn_hidden_size: int = 6144,
    total_tokens: int = 6144,
    unroll_level: int = 32,
) -> None:
    """使用 double buffer kernel 的版本"""
    
    for idx, loop_base in pypto.loop_unroll(
        total_tokens, 
        name="token_loop_{}".format(task_id), 
        unroll_list=[unroll_level*4, unroll_level*2, unroll_level, 24, 12, 6, 3, 1]
    ):
        # 获取对应的 torch tensor
        x_torch = x._get_torch_tensor()  # 假设有这个方法获取底层 torch tensor
        w1_torch = w1._get_torch_tensor()
        
        # 计算 tile 的偏移
        x_tile_torch = x_torch[idx:idx+loop_base, :]
        w1_tile_torch = w1_torch
        
        # 直接调用 double buffer kernel（绕过 pypto graph）
        # 注意：这里的调用是在 Python 层面，不会进入 pypto 的 graph 优化
        res_torch = torch.ops.npu.gemm_performance(
            x_tile_torch, w1_tile_torch,
            blockDim=24,
            m=loop_base, k=hidden_size, n=ffn_hidden_size,
            singleCoreM=loop_base, singleCoreK=hidden_size, singleCoreN=ffn_hidden_size,
            baseM=128, baseK=64, baseN=256,
            stepM=1, stepKa=4, stepKb=4, stepN=1
        )
        
        # 将结果写回 pypto tensor
        out._copy_from_torch_tensor(res_torch, offset=[idx, 0])
```

**注意事项：**
- 这种方案会跳出 pypto 的 graph 系统，所以无法在同一个 `@pypto.jit` 函数中混合使用
- 如果需要混合使用，需要在不同的函数间分割

### 1.3 改进版本：分离 JIT 函数

更好的方式是分离成两个函数：

```python
# 不使用 @pypto.jit 的 wrapper 函数，直接调用 torch.ops
def gemm_performance_wrapper(x_torch, w1_torch, loop_base, hidden_size, ffn_hidden_size):
    """包装 double buffer kernel 调用"""
    return torch.ops.npu.gemm_performance(
        x_torch, w1_torch,
        blockDim=24,
        m=loop_base, k=hidden_size, n=ffn_hidden_size,
        singleCoreM=loop_base, singleCoreK=hidden_size, singleCoreN=ffn_hidden_size,
        baseM=128, baseK=64, baseN=256,
        stepM=1, stepKa=4, stepKb=4, stepN=1
    )

@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_hybrid(
    x: torch.Tensor,  # 直接使用 torch.Tensor，不用 pypto.Tensor
    w1: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int = 6144,
    ffn_hidden_size: int = 6144,
    total_tokens: int = 6144,
    unroll_level: int = 32,
) -> None:
    """混合模式：在 Python 循环中调用 double buffer kernel"""
    
    for idx in range(0, total_tokens, unroll_level):
        loop_base = min(unroll_level, total_tokens - idx)
        
        # 切分 tile
        x_tile = x[idx:idx+loop_base, :]
        
        # 调用 double buffer kernel
        res_tile = gemm_performance_wrapper(
            x_tile, w1, loop_base, hidden_size, ffn_hidden_size
        )
        
        # 写回结果
        out[idx:idx+loop_base, :] = res_tile
```

---

## 方案2: 在 C++ 后端自动选择（透明集成）

在 pypto 的 C++ 后端集成，当检测到特定形状时自动使用 double buffer kernel。

### 2.1 修改 cube_operation_impl.cpp

```cpp
// 在 cube_operation_impl.cpp 中添加

#include "kernels/manual/gemm_performance/gemm_performance_kernel.h"

bool MatchesDoubleBufferGemm(const Tensor &a, const Tensor &b) {
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();
    
    // 检查是否匹配 gemm_performance_kernel 支持的配置
    // 例如：m=6144, k=6144, n=6144 等
    // 这里需要根据实际支持的配置来判断
    if (shapeA.size() == 2 && shapeB.size() == 2) {
        uint32_t m = shapeA[0];
        uint32_t k = shapeA[1];
        uint32_t n = shapeB[1];
        
        // 检查是否匹配已知的高性能配置
        // (可以根据实际情况扩展)
        return (m == 6144 && k == 6144 && n == 6144) ||
               (m == 6144 && k == 6144 && n == 12288) ||
               (m == 4096 && k == 4096 && n == 8192) ||
               (m == 4096 && k == 4096 && n == 4096);
    }
    return false;
}

Tensor CallDoubleBufferGemm(DataType outType, const Tensor &a, const Tensor &b) {
    // 从 Tensor 获取形状和指针
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();
    uint32_t m = shapeA[0];
    uint32_t k = shapeA[1];
    uint32_t n = shapeB[1];
    
    // 准备输出 tensor
    Tensor out = Tensor::Empty({m, n}, outType);
    
    // 获取设备指针
    void *out_ptr = out.GetDataPtr();
    void *a_ptr = a.GetDataPtr();
    void *b_ptr = b.GetDataPtr();
    
    // 调用 kernel（需要根据实际情况调整参数）
    LaunchGEMME2E_Templated<uint16_t, 24, m, k, n, 
        m, k, n, 128, 64, 256, 1, 4, 4, 1>(
        reinterpret_cast<uint8_t *>(out_ptr),
        reinterpret_cast<uint8_t *>(a_ptr),
        reinterpret_cast<uint8_t *>(b_ptr),
        nullptr  // stream
    );
    
    return out;
}

template <bool isATrans, bool isBTrans, bool isCMatrixNZ>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix) {
    // 如果匹配 double buffer kernel，使用它
    if (!isATrans && !isBTrans && MatchesDoubleBufferGemm(aMatrix, bMatrix)) {
        return CallDoubleBufferGemm(outType, aMatrix, bMatrix);
    }
    
    // 否则使用标准实现
    return ConstructTensorGraph<isATrans, isBTrans, isCMatrixNZ>(
        outType, aMatrix, bMatrix, Tensor());
}
```

### 2.2 Python 端使用（无变化）

```python
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_v3(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    out: pypto.Tensor,
    # ... 其他参数
) -> None:
    for idx, loop_base in pypto.loop_unroll(...):
        x_tile = pypto.view(x, shape=[loop_base, hidden_size], offsets=[idx, 0])
        w1_tile = pypto.view(w1, shape=[hidden_size, ffn_hidden_size], offsets=[0, 0])
        
        # 这里会自动调用 double buffer kernel（如果形状匹配）
        res = pypto.matmul(x_tile, w1_tile, pypto.DT_FP16)
        pypto.assemble(res, [idx, 0], out)
```

**优点：**
- 用户无需修改代码
- 自动选择最优 kernel

**缺点：**
- 无法精确控制 double buffer 的细节（如 sync 标志）
- 需要修改框架代码

---

## 方案3: 混合模式（推荐用于生产环境）

结合方案1和方案2，提供两种调用方式：

1. **精确控制模式**：通过 `torch.ops.npu.gemm_performance` 直接调用
2. **自动选择模式**：通过 `pypto.matmul`，框架自动选择最优实现

### 3.1 实现示例

```python
def get_double_buffer_matmul_fn():
    """获取 double buffer matmul 函数（如果可用）"""
    try:
        return torch.ops.npu.gemm_performance
    except AttributeError:
        return None

@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_smart(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    out: pypto.Tensor,
    use_double_buffer: bool = True,  # 是否使用 double buffer
    # ... 其他参数
) -> None:
    """智能选择 matmul 实现"""
    
    double_buffer_fn = get_double_buffer_matmul_fn() if use_double_buffer else None
    
    for idx, loop_base in pypto.loop_unroll(...):
        x_tile = pypto.view(x, shape=[loop_base, hidden_size], offsets=[idx, 0])
        w1_tile = pypto.view(w1, shape=[hidden_size, ffn_hidden_size], offsets=[0, 0])
        
        if double_buffer_fn and MatchesDoubleBufferConfig(loop_base, hidden_size, ffn_hidden_size):
            # 使用精确控制的 double buffer kernel
            x_torch = x_tile._get_torch_tensor()
            w1_torch = w1_tile._get_torch_tensor()
            res_torch = double_buffer_fn(x_torch, w1_torch, ...)
            pypto.assemble(pypto.tensor_from_torch(res_torch), [idx, 0], out)
        else:
            # 使用标准 pypto.matmul（会自动选择最优实现）
            res = pypto.matmul(x_tile, w1_tile, pypto.DT_FP16)
            pypto.assemble(res, [idx, 0], out)
```

---

## 推荐实施方案

### 短期方案（快速集成）
使用**方案1**，通过 `torch.ops` 直接调用，完全绕过 graph 化，保留所有 double buffer 优化。

### 长期方案（生产环境）
采用**方案3**，同时提供：
1. 精确控制接口（`torch.ops.npu.gemm_performance`）
2. 自动选择机制（在 C++ 后端集成，`pypto.matmul` 自动选择）

这样既满足性能调优需求，又保持框架的易用性。

---

## 注意事项

1. **内存对齐**：确保 tensor 的内存布局符合 kernel 要求
2. **数据类型**：`gemm_performance_kernel` 目前只支持 `uint16_t` (FP16)
3. **形状限制**：只支持特定的形状配置，需要添加检查
4. **编译依赖**：需要将 kernel 加入到构建系统中
5. **Stream 同步**：确保在多 stream 环境下正确处理同步

---

## 参考示例

参考 `pto-isa/demos/baseline/add/` 中的 add 算子示例，了解如何注册和调用自定义 kernel。
