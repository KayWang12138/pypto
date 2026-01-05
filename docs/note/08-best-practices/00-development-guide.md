# PyPTO 开发指南

> **适用对象：** 想要参与PyPTO开发的贡献者  
> **学习时间：** 40-60分钟  
> **前置知识：** 已阅读[架构设计](../02-core/13-architecture-design.md)  
> **学习目标：** 掌握开发流程、代码规范和协作规范

## 概述

本文档总结 PyPTO 项目的开发流程、代码规范、测试规范和协作规范，帮助开发者快速融入项目开发。

**开发流程：**
- 🔧 **环境搭建**：开发环境配置
- 📝 **代码开发**：开发规范和流程
- 🧪 **测试验证**：测试方法和标准
- 📤 **提交审查**：PR流程和规范

**相关文档：**
- [架构设计总结](../02-core/13-architecture-design.md) - 理解架构设计理念
- [API 使用总结](../02-core/02-api-reference.md) - 了解 API 使用方式
- [设计模式与最佳实践](00-design-patterns.md) - 核心设计模式

---

## 目录

- [开发环境搭建](#开发环境搭建)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [测试规范](#测试规范)
- [代码审查](#代码审查)
- [发布流程](#发布流程)
- [故障排查](#故障排查)

---

## 开发环境搭建

### 系统要求

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|----------|----------|------|
| **操作系统** | Ubuntu 18.04 | Ubuntu 20.04 | 支持其他 Linux 发行版 |
| **Python** | 3.8 | 3.9 | 3.10+ 支持实验性功能 |
| **GCC** | 9.0 | 11.0 | 支持 C++17 标准 |
| **CMake** | 3.16 | 3.24 | 构建系统 |
| **CUDA** | 11.0 | 11.8 | GPU 支持 (可选) |
| **Ascend** | 7.0 | 8.0 | NPU 支持 |

### 环境安装

#### 1. 基础环境

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential cmake git vim gdb valgrind

# 安装 Python
sudo apt install -y python3 python3-dev python3-pip python3-venv

# 验证安装
python3 --version
cmake --version
gcc --version
```

#### 2. PyPTO 依赖

```bash
# 克隆代码
git clone https://github.com/your-org/pypto.git
cd pypto

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_npu
```

#### 3. 昇腾环境 (可选)

```bash
# 下载昇腾工具包
wget https://www.huawei.com/ascend/ascend-cann-X.X.X-linux.x86_64.run

# 安装 CANN
bash ascend-cann-X.X.X-linux.x86_64.run --no-op

# 设置环境变量
export ASCEND_HOME_PATH="/usr/local/Ascend"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$ASCEND_HOME_PATH/bin:$PATH"

# 验证安装
npu-smi info
```

### 项目构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置构建
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_WITH_CANN=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_TESTS=ON

# 编译
make -j$(nproc)

# 安装
make install

# 运行测试
make test
```

### 开发工具配置

#### VS Code 配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "cmake.configureOnOpen": true,
    "cmake.buildDirectory": "${workspaceFolder}/build"
}
```

#### Git 配置

```bash
# 配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置代码风格
git config --global core.autocrlf input
git config --global core.safecrlf true

# 配置别名
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

---

## 开发流程

### 1. 任务规划

#### 需求分析

```markdown
# 需求文档模板

## 需求概述
- 背景：为什么需要这个功能
- 目标：要实现什么
- 范围：包含哪些内容

## 功能规格
- 输入：函数签名和参数
- 输出：返回值和副作用
- 约束：性能要求、兼容性要求

## 设计考虑
- 架构影响：对现有架构的影响
- 性能影响：性能预算和优化策略
- 兼容性：向后兼容性保证

## 验收标准
- 功能测试：测试用例
- 性能测试：性能基准
- 文档更新：相关文档更新
```

#### 任务分解

```markdown
# 任务分解示例

## 主任务：实现新的优化 Pass

### 子任务
1. **设计 Pass 接口** (2天)
   - 分析现有 Pass 结构
   - 定义新 Pass 的接口
   - 编写接口文档

2. **实现 Pass 逻辑** (3天)
   - 实现核心优化算法
   - 添加配置选项
   - 编写单元测试

3. **集成和测试** (2天)
   - 集成到 Pass 管理器
   - 端到端测试
   - 性能基准测试

4. **文档和审查** (1天)
   - 更新相关文档
   - 代码审查
   - 合并到主分支
```

### 2. 分支管理

#### 分支策略

```bash
# 主分支
git branch -a
# * main          # 主分支，稳定版本
#   develop       # 开发分支，最新功能

# 功能分支
git checkout -b feature/new-optimization-pass

# 修复分支
git checkout -b fix/memory-leak-issue

# 发布分支
git checkout -b release/v1.1.0
```

#### 提交规范

```bash
# 提交消息格式
<type>(<scope>): <subject>

# 常用类型
feat:     新功能
fix:      修复bug
docs:     文档更新
style:    代码风格调整
refactor: 重构
test:     测试相关
chore:    构建过程或工具配置

# 示例
git commit -m "feat(optimizer): add new memory reuse pass

- Implement GlobalMemoryReuse pass for tensor memory optimization
- Add configuration options for memory threshold
- Update documentation with usage examples

Closes #123"

# 关联 Issue
git commit -m "fix: resolve segmentation fault in tensor operations

Fix null pointer access in tensor reshape operation.
Add bounds checking for array access.

Fixes #456"
```

### 3. 代码开发

#### 开发步骤

```python
# 1. 创建功能分支
git checkout -b feature/new-feature

# 2. 编写代码
# 实现新功能

# 3. 编写测试
# 添加单元测试和集成测试

# 4. 运行测试
python -m pytest tests/ -v

# 5. 代码格式化
black .
clang-format -i framework/src/**/*.cpp framework/include/**/*.h

# 6. 提交更改
git add .
git commit -m "feat: implement new feature"

# 7. 推送到远程
git push origin feature/new-feature
```

#### 调试技巧

```python
# 启用调试模式
# 调试配置（当前版本不支持高级调试选项）
# 使用环境变量 GLOBAL_LOG_LEVEL=0 替代

# 添加调试日志
def debug_function(x):
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    result = x * 2

    print(f"Output shape: {result.shape}")
    return result

# 使用断言
assert tensor.shape[0] > 0, "Batch size must be positive"
assert tensor.dtype == pypto.DT_FP32, "Expected FP32 tensor"  # (@docs/api/datatype/DataType.md)

# 性能调试
import time
start = time.time()
result = expensive_operation(data)
end = time.time()
print(f"Operation took {end - start:.3f} seconds")
```

---

## 代码规范

### Python 代码规范

#### PEP 8 遵循

```python
# 好的示例
def calculate_score(predictions, targets, weights=None):
    """
    Calculate weighted accuracy score.

    Args:
        predictions: Model predictions tensor
        targets: Ground truth labels tensor
        weights: Optional weighting tensor

    Returns:
        Weighted accuracy score
    """
    if weights is None:
        weights = torch.ones_like(predictions)

    correct = (predictions == targets).float()
    weighted_correct = correct * weights
    score = weighted_correct.sum() / weights.sum()

    return score


# 不好的示例
def calc_score(preds, targs, w=None):  # 变量名不清晰
    if w is None: w = torch.ones_like(preds)  # 缺乏空格
    correct = (preds == targs).float()
    weighted_correct = correct * w
    score = weighted_correct.sum() / w.sum()  # 缺乏中间变量
    return score
```

#### 类型注解

```python
from typing import Optional, Union, List, Tuple
import torch

def process_tensor(
    tensor: torch.Tensor,
    target_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Process tensor with optional reshaping and type conversion.

    Args:
        tensor: Input tensor
        target_shape: Target shape for reshaping
        dtype: Target data type

    Returns:
        Processed tensor
    """
    if target_shape is not None:
        tensor = tensor.reshape(target_shape)

    if dtype is not None:
        # 注意：PyPTO Tensor 不提供 to() 方法
        # 数据类型转换请使用 cast 操作（参考 @docs/api/operation/pypto-cast.md）

    return tensor
```

#### 文档字符串

```python
def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with broadcasting support.

    Performs matrix multiplication between tensors a and b, with automatic
    broadcasting for compatible dimensions.

    Args:
        a: Left matrix tensor of shape (..., M, K)
        b: Right matrix tensor of shape (..., K, N)

    Returns:
        Result tensor of shape (..., M, N)

    Raises:
        ValueError: If tensor dimensions are incompatible

    Examples:
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = matrix_multiply(a, b)
        >>> c.shape
        torch.Size([3, 5])

    Note:
        This function uses PyTorch's optimized matmul implementation
        for best performance on supported hardware.
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible dimensions: {a.shape} vs {b.shape}")

    return torch.matmul(a, b)
```

### C++ 代码规范

#### Google C++ 风格

```cpp
// framework/include/pypto/tensor.h
#ifndef PYTO_TENSOR_H_
#define PYTO_TENSOR_H_

#include <memory>
#include <vector>

#include "pypto/common.h"
#include "pypto/shape.h"

namespace pypto {

// Tensor 类定义
class Tensor {
 public:
  // 构造函数
  explicit Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  // 析构函数
  ~Tensor();

  // 公共方法
  const Shape& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }

  // 数据访问
  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(data_.get());
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_.get());
  }

  // 运算符重载
  Tensor operator+(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;

 private:
  Shape shape_;
  DataType dtype_;
  std::unique_ptr<void, decltype(&free)> data_;
};

}  // namespace pypto

#endif  // PYTO_TENSOR_H_
```

#### 实现文件

```cpp
// framework/src/tensor.cpp
#include "pypto/tensor.h"

#include <cstring>
#include <stdexcept>

namespace pypto {

Tensor::Tensor(const Shape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), data_(nullptr, free) {
  size_t size_bytes = shape.num_elements() * DataTypeSize(dtype);
  void* data = malloc(size_bytes);
  if (data == nullptr) {
    throw std::runtime_error("Failed to allocate tensor memory");
  }
  data_ = std::unique_ptr<void, decltype(&free)>(data, free);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      data_(std::move(other.data_)) {
  other.dtype_ = DataType::UNKNOWN;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    data_ = std::move(other.data_);
    other.dtype_ = DataType::UNKNOWN;
  }
  return *this;
}

Tensor::~Tensor() = default;

Tensor Tensor::operator+(const Tensor& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch in tensor addition");
  }
  if (dtype_ != other.dtype_) {
    throw std::invalid_argument("Data type mismatch in tensor addition");
  }

  Tensor result(shape_, dtype_);
  // 实现加法逻辑
  return result;
}

}  // namespace pypto
```

### 命名规范

#### 变量和函数命名

```python
# Python 命名
def calculate_weighted_sum(values, weights):
    """Calculate weighted sum of values."""
    weighted_values = values * weights
    total_sum = weighted_values.sum()
    return total_sum

class TensorProcessor:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def process_batch(self, batch_tensors):
        """Process a batch of tensors."""
        processed_batch = []
        for tensor in batch_tensors:
            processed = self._process_single_tensor(tensor)
            processed_batch.append(processed)
        return processed_batch

    def _process_single_tensor(self, tensor):
        """Process a single tensor (private method)."""
        # Implementation
        pass
```

```cpp
// C++ 命名
class TensorProcessor {
 public:
  explicit TensorProcessor(const Shape& input_shape, const Shape& output_shape);
  ~TensorProcessor() = default;

  // 公共方法
  Tensor ProcessBatch(const std::vector<Tensor>& batch_tensors);
  void SetProcessingOptions(const ProcessingOptions& options);

 private:
  // 私有方法
  Tensor ProcessSingleTensor(const Tensor& tensor);

  // 成员变量
  Shape input_shape_;
  Shape output_shape_;
  ProcessingOptions options_;
};
```

#### 文件和目录命名

```
# 项目结构命名
pypto/
├── python/pypto/              # Python 包
│   ├── __init__.py           # 包初始化
│   ├── tensor.py             # 张量操作
│   ├── functions.py          # 函数装饰器
│   └── config.py             # 配置管理
├── framework/                 # C++ 框架
│   ├── include/pypto/        # 公共头文件
│   ├── src/                  # 源代码
│   │   ├── interface/        # 接口层
│   │   ├── passes/           # 优化层
│   │   ├── codegen/          # 代码生成层
│   │   └── machine/          # 执行层
│   └── CMakeLists.txt        # 构建配置
├── examples/                  # 示例代码
├── tests/                     # 测试代码
└── docs/                      # 文档
```

---

## 测试规范

### 单元测试

#### 测试文件组织

```
tests/
├── unit/                      # 单元测试
│   ├── test_tensor.py        # 张量测试
│   ├── test_operations.py    # 操作测试
│   ├── test_passes.py        # Pass 测试
│   └── test_codegen.py       # 代码生成测试
├── integration/               # 集成测试
│   ├── test_end_to_end.py    # 端到端测试
│   └── test_compilation.py   # 编译流程测试
├── performance/               # 性能测试
│   ├── test_benchmarks.py    # 基准测试
│   └── test_regression.py    # 回归测试
└── conftest.py                # 测试配置
```

#### 单元测试示例

```python
# tests/unit/test_tensor.py
import pytest
import numpy as np
import pypto


class TestTensor:
    """Tensor 类单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.test_data = np.random.randn(10, 20).astype(np.float32)

    def test_tensor_creation_from_numpy(self):
        """测试从 NumPy 数组创建张量"""
        tensor = pypto.zeros(self.test_data.shape)

        assert tensor.shape == (10, 20)
        assert tensor.dtype == pypto.DT_FP32  # (@docs/api/datatype/DataType.md)
        # 注意：PyPTO Tensor 不提供 to_numpy() 方法
        # 测试时请使用其他验证方式

    def test_tensor_addition(self):
        """测试张量加法"""
        a = pypto.zeros(self.test_data.shape)
        b = pypto.ones_like(a)

        result = a + b

        expected = self.test_data + 1.0
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_tensor_reshape(self):
        """测试张量重塑"""
        tensor = pypto.zeros(self.test_data.shape)

        # 重塑为不同形状
        reshaped = tensor.reshape((5, 40))

        assert reshaped.shape == (5, 40)
        assert reshaped.numel() == tensor.numel()
        np.testing.assert_array_equal(
            reshaped.to_numpy().flatten(),
            tensor.to_numpy().flatten()
        )

    @pytest.mark.parametrize("dtype", [pypto.DT_FP32, pypto.DT_FP16, pypto.DT_INT32])  # (@docs/api/datatype/DataType.md)
    def test_tensor_dtype_conversion(self, dtype):
        """测试数据类型转换"""
        tensor = pypto.zeros(self.test_data.shape)

        converted = tensor.to(dtype)

        assert converted.dtype == dtype
        assert converted.shape == tensor.shape

    def test_tensor_invalid_operations(self):
        """测试无效操作的错误处理"""
        a = pypto.zeros((5, 10))
        b = pypto.zeros((3, 8))  # 不兼容形状

        with pytest.raises(ValueError, match="Shape mismatch"):
            a + b

    def teardown_method(self):
        """测试后清理"""
        # 清理测试资源
        pass
```

#### 测试运行

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/unit/test_tensor.py

# 运行特定测试用例
python -m pytest tests/unit/test_tensor.py::TestTensor::test_tensor_addition

# 运行带覆盖率的测试
python -m pytest --cov=pypto --cov-report=html tests/

# 运行性能测试
python -m pytest tests/performance/ -v

# 运行失败重试
python -m pytest --reruns 3 --reruns-delay 1 tests/
```

### 集成测试

#### 端到端测试

```python
# tests/integration/test_end_to_end.py
import pytest
import numpy as np
import pypto


class TestEndToEnd:
    """端到端测试"""

    @pytest.fixture
    def sample_data(self):
        """测试数据fixture"""
        return {
            'input': np.random.randn(32, 784).astype(np.float32),
            'weights': np.random.randn(784, 256).astype(np.float32),
            'bias': np.random.randn(256).astype(np.float32)
        }

    def test_simple_neural_network(self, sample_data):
        """测试简单神经网络"""
        @pypto.jit
        def neural_network(x, w1, b1):
            h = (x @ w1 + b1).relu()
            return h.sum(dim=-1)

        result = neural_network(
            pypto.zeros(sample_data['input'].shape),
            pypto.zeros(sample_data['weights'].shape),
            pypto.zeros(sample_data['bias'].shape)
        )

        assert result.shape == (32,)
        assert not np.isnan(result.to_numpy()).any()

    def test_matrix_multiplication(self, sample_data):
        """测试矩阵乘法"""
        @pypto.jit
        def matrix_mul(a, b):
            return a @ b
            # 注意：pypto.jit 不支持 run_mode 参数

        a = pypto.zeros(sample_data['input'].shape)
        b = pypto.zeros(sample_data['weights'].T.shape)

        result = matrix_mul(a, b)

        expected_shape = (32, 256)
        assert result.shape == expected_shape

        # 数值验证
        expected = sample_data['input'] @ sample_data['weights'].T
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    def test_compilation_and_execution(self, sample_data):
        """测试编译和执行流程"""
        @pypto.jit
        def complex_computation(x, w, b):
            # 复杂的计算流程
            h1 = (x @ w + b).relu()
            h2 = h1 @ w.T
            output = h2.softmax(dim=-1)
            return output

        # 测试编译
        compiled_func = complex_computation

        # 测试执行
        result = compiled_func(
            pypto.zeros(sample_data['input'].shape),
            pypto.zeros(sample_data['weights'].shape),
            pypto.zeros(sample_data['bias'].shape)
        )

        # 验证结果
        assert result.shape[0] == 32
        assert result.shape[1] == 784  # softmax 输出
        assert np.allclose(result.to_numpy().sum(axis=1), 1.0, atol=1e-6)
```

### 性能测试

#### 基准测试

```python
# tests/performance/test_benchmarks.py
import pytest
import time
import numpy as np
import pypto


class TestBenchmarks:
    """性能基准测试"""

    @pytest.fixture
    def benchmark_data(self):
        """基准测试数据"""
        return {
            'small': np.random.randn(128, 128).astype(np.float32),
            'medium': np.random.randn(512, 512).astype(np.float32),
            'large': np.random.randn(1024, 1024).astype(np.float32)
        }

    def test_matrix_multiplication_performance(self, benchmark_data, benchmark):
        """矩阵乘法性能测试"""

        @pypto.jit
        def matmul_optimized(a, b):
            return a @ b

        for size_name, data in benchmark_data.items():
            a = pypto.zeros(data.shape)
            b = pypto.zeros(data.T.shape)

            # 使用 pytest-benchmark
            result = benchmark(matmul_optimized, a, b)

            assert result.shape == (data.shape[0], data.shape[0])
            # 基准测试会自动记录性能数据

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_batch_processing_performance(self, batch_size, benchmark):
        """批处理性能测试"""

        @pypto.jit
        def batch_process(batch):
            # 模拟批处理操作
            result = batch * 2
            result = result.relu()
            return result.sum(dim=-1)

        data = np.random.randn(batch_size, 784).astype(np.float32)
        batch = pypto.zeros(data.shape)

        result = benchmark(batch_process, batch)

        assert result.shape == (batch_size,)

    def test_memory_usage_benchmark(self, benchmark_data):
        """内存使用基准测试"""

        @pypto.jit
        def memory_intensive_operation(data):
            # 内存密集型操作
            temp1 = data * 2
            temp2 = temp1 @ data.T
            temp3 = temp2.relu()
            return temp3.sum()

        initial_memory = get_memory_usage()

        for size_name, data in benchmark_data.items():
            tensor = pypto.zeros(data.shape)

            # 记录内存峰值
            peak_memory = track_memory_peak(memory_intensive_operation, tensor)

            # 验证内存使用在合理范围内
            assert peak_memory < initial_memory * 3  # 不超过 3 倍初始内存

    def test_compilation_time_benchmark(self, benchmark):
        """编译时间基准测试"""

        def create_complex_function(num_operations):
            operations = []
            for i in range(num_operations):
                operations.append(f"temp{i} = x * {i}")

            operations.append("return temp0")
            function_code = f"""
@pypto.jit
def complex_function(x):
    {chr(10).join(operations)}
    return temp0
"""
            return function_code

        # 测试不同复杂度的函数编译时间
        for complexity in [10, 50, 100]:
            code = create_complex_function(complexity)

            # 编译时间基准测试
            compile_time = benchmark(exec, code)

            # 验证编译时间在合理范围内
            assert compile_time < 10.0  # 编译时间不超过 10 秒
```

---

## 代码审查

### 审查清单

#### 功能完整性

- [ ] **需求覆盖**: 实现是否满足所有需求
- [ ] **边界情况**: 是否处理了边界情况和异常输入
- [ ] **错误处理**: 是否有适当的错误处理和日志记录
- [ ] **向后兼容**: 是否保持向后兼容性

#### 代码质量

- [ ] **代码规范**: 是否遵循项目的代码规范
- [ ] **命名清晰**: 变量、函数、类命名是否清晰准确
- [ ] **注释完整**: 是否有必要的注释和文档字符串
- [ ] **复杂度控制**: 函数和类的复杂度是否在合理范围内

#### 测试覆盖

- [ ] **单元测试**: 是否有完整的单元测试
- [ ] **集成测试**: 是否有集成测试验证功能
- [ ] **边界测试**: 是否测试了边界情况
- [ ] **错误测试**: 是否测试了错误处理

#### 性能和资源

- [ ] **性能影响**: 新代码是否影响现有性能
- [ ] **内存使用**: 是否有内存泄漏或其他资源问题
- [ ] **可扩展性**: 代码是否具有良好的可扩展性

### 审查流程

#### 1. 自我审查

```markdown
# 自我审查清单

## 功能验证
- [ ] 本地测试通过
- [ ] 示例代码工作正常
- [ ] 边界情况测试通过

## 代码质量
- [ ] 代码格式化 (black, clang-format)
- [ ] 静态分析通过 (pylint, cppcheck)
- [ ] 单元测试覆盖率 > 80%

## 文档更新
- [ ] 相关文档已更新
- [ ] API 文档已更新
- [ ] 示例代码已添加

## 兼容性检查
- [ ] 向后兼容性保证
- [ ] 依赖关系正确
- [ ] 构建配置正确
```

#### 2. 同行审查

```markdown
# 审查意见模板

## 总体评价
[ ] 批准合并  [ ] 需要修改  [ ] 拒绝合并

## 主要问题
1. **问题描述**: 详细描述发现的问题
   **严重程度**: [严重/中等/轻微]
   **建议修改**: 具体的修改建议

2. **问题描述**: ...
   **严重程度**: ...
   **建议修改**: ...

## 次要问题
- 代码风格问题
- 注释完善建议
- 性能优化建议

## 积极方面
- 好的设计决策
- 清晰的代码结构
- 充分的测试覆盖

## 附加要求
- 需要补充测试
- 需要更新文档
- 需要性能基准测试
```

#### 3. 审查响应

```markdown
# 审查响应模板

## 已解决问题
- [ ] 问题1: 已按照建议修改，具体改动...
- [ ] 问题2: 已修复，原因...

## 保留的决定
- [ ] 问题3: 经过考虑，保留当前实现，理由...

## 新增修改
- [ ] 根据审查意见，新增了以下改进...
- [ ] 补充了测试用例...
- [ ] 更新了文档...

## 验证结果
- [ ] 所有测试通过
- [ ] 性能测试通过
- [ ] 兼容性测试通过
```

---

## 发布流程

### 版本管理

#### 语义化版本

```
版本格式: MAJOR.MINOR.PATCH

MAJOR: 不兼容的 API 变更
MINOR: 向后兼容的功能新增
PATCH: 向后兼容的问题修复
```

#### 版本分支管理

```bash
# 创建发布分支
git checkout -b release/v1.1.0 develop

# 版本号更新
echo "1.1.0" > VERSION
# 更新代码中的版本号

# 提交版本更新
git commit -m "chore: bump version to 1.1.0"

# 创建标签
git tag -a v1.1.0 -m "Release version 1.1.0"

# 推送到远程
git push origin release/v1.1.0
git push origin v1.1.0
```

### 发布检查清单

#### 代码准备

- [ ] **分支检查**: 从正确的分支创建发布
- [ ] **测试通过**: 所有测试用例通过
- [ ] **代码审查**: 关键修改已通过审查
- [ ] **文档更新**: 相关文档已更新
- [ ] **版本号**: 版本号已正确更新

#### 构建验证

- [ ] **编译成功**: 在所有支持的平台上编译成功
- [ ] **依赖完整**: 所有依赖项正确指定
- [ ] **包大小**: 包大小在合理范围内
- [ ] **安装测试**: 可以正确安装和导入

#### 功能验证

- [ ] **基本功能**: 核心功能正常工作
- [ ] **向后兼容**: 现有代码仍然工作
- [ ] **性能基准**: 性能没有明显下降
- [ ] **错误处理**: 错误情况正确处理

#### 文档和示例

- [ ] **README 更新**: README 包含最新信息
- [ ] **API 文档**: API 文档已更新
- [ ] **使用示例**: 示例代码可以正常运行
- [ ] **发布说明**: 发布说明详细列出变更

### 发布执行

#### 1. 预发布准备

```bash
# 创建发布分支
git checkout -b release/v1.1.0
git push origin release/v1.1.0

# 最终测试
make clean && make -j$(nproc)
make test
make package  # 创建发布包

# 文档生成
make docs
```

#### 2. 发布执行

```bash
# GitHub Release
# 1. 转到 Releases 页面
# 2. 点击 "Create a new release"
# 3. 选择标签 v1.1.0
# 4. 填写发布说明
# 5. 上传发布包

# PyPI 发布 (如果适用)
python setup.py sdist bdist_wheel
twine upload dist/*
```

#### 3. 发布后处理

```bash
# 合并到主分支
git checkout main
git merge release/v1.1.0
git push origin main

# 更新开发分支
git checkout develop
git merge release/v1.1.0
git push origin develop

# 删除发布分支
git branch -d release/v1.1.0
git push origin --delete release/v1.1.0
```

### 发布后监控

#### 问题监控

```python
# 发布后监控脚本
import time
import requests
from datetime import datetime, timedelta

class ReleaseMonitor:
    def __init__(self, repo_owner, repo_name, version):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.version = version
        self.start_time = datetime.now()

    def monitor_issues(self):
        """监控新问题"""
        # 检查 GitHub Issues
        issues = self.get_github_issues()

        new_issues = []
        for issue in issues:
            if issue['created_at'] > self.start_time:
                new_issues.append(issue)

        if new_issues:
            self.report_new_issues(new_issues)

    def monitor_downloads(self):
        """监控下载量"""
        # 检查 PyPI 下载统计
        downloads = self.get_pypi_downloads()

        if downloads < self.expected_downloads:
            self.report_low_downloads(downloads)

    def monitor_ci_status(self):
        """监控 CI 状态"""
        # 检查 GitHub Actions 状态
        ci_status = self.get_ci_status()

        if not ci_status['success']:
            self.report_ci_failure(ci_status)

    def get_github_issues(self):
        """获取 GitHub Issues"""
        # 实现 GitHub API 调用
        pass

    def get_pypi_downloads(self):
        """获取 PyPI 下载统计"""
        # 实现 PyPI API 调用
        pass

    def get_ci_status(self):
        """获取 CI 状态"""
        # 实现 GitHub API 调用
        pass

    def report_new_issues(self, issues):
        """报告新问题"""
        print(f"⚠️  New issues reported after release {self.version}:")
        for issue in issues:
            print(f"  - {issue['title']} ({issue['number']})")

    def report_low_downloads(self, downloads):
        """报告下载量异常"""
        print(f"⚠️  Low download count: {downloads}")

    def report_ci_failure(self, ci_status):
        """报告 CI 失败"""
        print(f"❌ CI failure: {ci_status['message']}")

# 使用发布监控
monitor = ReleaseMonitor("your-org", "pypto", "1.1.0")

# 持续监控一周
for _ in range(7 * 24):  # 每小时检查一次
    monitor.monitor_issues()
    monitor.monitor_downloads()
    monitor.monitor_ci_status()
    time.sleep(3600)  # 等待一小时
```

---

## 故障排查

### 常见开发问题

#### 编译问题

**问题**: 编译失败，提示 "undefined reference"

**排查步骤**:
1. 检查 CMakeLists.txt 中的链接依赖
2. 验证头文件包含路径
3. 确认符号定义位置
4. 检查编译顺序

**解决方案**:
```cmake
# CMakeLists.txt 添加依赖
target_link_libraries(pypto
    PUBLIC interface
    PRIVATE passes codegen machine
)
```

#### 测试问题

**问题**: 测试失败，但代码看似正确

**排查步骤**:
1. 检查测试环境设置
2. 验证测试数据准备
3. 确认断言条件正确
4. 检查测试依赖关系

**解决方案**:
```python
# 正确的测试设置
class TestExample(unittest.TestCase):
    def setUp(self):
        # 正确的环境准备
        self.test_data = generate_test_data()
        self.expected_result = compute_expected(self.test_data)

    def test_functionality(self):
        result = function_under_test(self.test_data)
        self.assertTrue(torch.allclose(result, self.expected_result))
```

#### 性能问题

**问题**: 新功能显著降低性能

**排查步骤**:
1. 运行性能基准测试
2. 使用分析工具定位热点
3. 检查配置是否正确应用
4. 验证优化 Pass 是否启用

**解决方案**:
```python
# 启用性能优化
# Pass 配置（参考 @docs/api/config/pypto-set_pass_config.md）
from pypto import PassConfigKey
pypto.set_pass_config("default", "all", PassConfigKey.KEY_DUMP_GRAPH, True)

# 检查 Tile 形状
pypto.set_vec_tile_shapes(128, 512)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

### 调试工具使用

#### 核心调试工具

```bash
# GDB 调试 C++
gdb python3
(gdb) run <your_script.py>  # 注意：<your_script.py> 为占位符，请替换为实际脚本路径
(gdb) bt                    # 查看堆栈
(gdb) break function_name   # 设置断点
(gdb) print variable        # 查看变量

# Valgrind 内存检查
valgrind --tool=memcheck python3 <your_script.py>  # 注意：<your_script.py> 为占位符，请替换为实际脚本路径

# strace 系统调用跟踪
strace -e trace=network,process python3 <your_script.py>  # 注意：<your_script.py> 为占位符，请替换为实际脚本路径

# perf 性能分析
perf record -g python3 <your_script.py>  # 注意：<your_script.py> 为占位符，请替换为实际脚本路径
perf report
```

#### Python 调试工具

```python
# pdb 调试
import pdb
pdb.set_trace()  # 设置断点

# 远程调试
import pydevd
pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)

# 内存分析
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 函数实现
    pass

# 性能分析
# 性能分析使用系统工具（perf、gprof 等）
```

---

## 总结

PyPTO 的开发指南涵盖了从环境搭建到发布流程的完整开发周期：

### 核心原则

1. **质量优先**: 严格的代码规范和测试覆盖
2. **协作开发**: 清晰的分支策略和代码审查流程
3. **持续集成**: 自动化的构建、测试和发布流程
4. **文档驱动**: 完善的文档体系和开发指导

### 关键实践

1. **开发环境**: 统一的开发环境配置
2. **代码规范**: 一致的编码风格和最佳实践
3. **测试策略**: 多层次的测试覆盖
4. **审查流程**: 严格的代码审查和质量控制
5. **发布管理**: 规范的版本管理和发布流程

### 工具链

1. **版本控制**: Git 分支策略和提交规范
2. **构建工具**: CMake 和 Make
3. **测试框架**: pytest 和 Google Test
4. **代码质量**: 静态分析和格式化工具
5. **文档工具**: Markdown 和 Sphinx

遵循这些指南可以确保：
- **高质量代码**: 一致的代码风格和完善的测试
- **高效协作**: 清晰的流程和沟通机制
- **稳定发布**: 可靠的发布流程和监控机制
- **持续改进**: 基于反馈的持续优化

---

**相关文档：**
- [上手指南](../00-getting-started/00-quick-start.md) - 日常开发常用入口与速查
- [架构设计总结](../02-core/13-architecture-design.md) - 理解架构设计理念
- [API 使用总结](../02-core/02-api-reference.md) - 了解 API 使用方式

