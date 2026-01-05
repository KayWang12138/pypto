# 从源码构建与跑测试

> **目标读者：** 需要从源码编译 PyPTO 或运行测试的开发者  
> **前置条件：** 已完成[环境准备与安装](00-getting-started/01-environment-setup.md)

---

## 快速导航

- **构建 PyPTO**：见[构建流程](#构建流程)
- **运行测试**：见[运行测试](#运行测试)
- **构建选项**：见[常用构建开关](#常用构建开关)
- **环境变量**：见[构建相关环境变量](#构建相关环境变量)

---

## 构建流程

### 1. 基本构建命令

```bash
# 进入项目根目录
cd /path/to/pypto

# 基本构建（默认 Release 模式）
python3 build_ci.py --build_type Release --editable

# Debug 模式构建（用于调试）
python3 build_ci.py --build_type Debug --editable --clean

# 完整清理后构建
python3 build_ci.py --build_type Release --editable --clean
```

**参数说明：**
- `--build_type Release/Debug`：构建类型（Release 性能好，Debug 包含调试符号）
- `--editable`：以可编辑模式安装（开发时常用）
- `--clean`：清理之前的构建产物

### 2. 构建产物

构建成功后，会在 `build_out/` 目录生成：
- **Python 包**：`pypto-*.whl` 文件
- **C++ 库**：`python/pypto/pypto_impl*.so`（共享库）

### 3. 验证构建

```bash
# 检查 Python 包是否安装
python3 -c "import pypto; print('✅ PyPTO 构建成功')"

# 检查共享库（Debug 版本应包含调试符号）
file python/pypto/pypto_impl*.so
# Debug 版本应显示：with debug_info, not stripped
# Release 版本会显示：stripped
```

---

## 运行测试

### 1. 运行所有测试

```bash
# 进入项目根目录
cd /path/to/pypto

# 运行 Python 测试（使用 pytest）
pytest python/tests -v

# 运行特定测试文件
pytest python/tests/test_tensor.py -v

# 运行特定测试用例
pytest python/tests/test_tensor.py::test_tensor_basic -v
```

### 2. 测试目录结构

```
python/tests/
├── test_tensor.py          # Tensor 相关测试
├── test_operation.py       # Operation 相关测试
├── test_controlflow.py     # 控制流相关测试
└── ...
```

**详细说明：** 参考[测试方法文档](06-testing/00-methodology.md)

---

## 常用构建开关

| 开关 | 说明 | 默认值 |
|------|------|--------|
| `--build_type` | 构建类型（Release/Debug） | Release |
| `--editable` | 可编辑模式安装 | False |
| `--clean` | 清理构建产物 | False |
| `--skip_tests` | 跳过测试 | False |

**详细说明：** 参考[Build 系统文档](02-core/11-build.md#构建流程详解)

---

## 构建相关环境变量

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `PYPTO_THIRD_PARTY_PATH` | 第三方源码包路径 | `export PYPTO_THIRD_PARTY_PATH=/path/to/third-party` |
| `CMAKE_BUILD_TYPE` | CMake 构建类型 | `export CMAKE_BUILD_TYPE=Release` |
| `CC` / `CXX` | 编译器路径 | `export CC=gcc` |

**详细说明：** 参考[环境准备与安装](00-getting-started/01-environment-setup.md)和[构建与调试工具机制详解](03-mechanisms/02-build-and-debug-mechanisms.md)

---

## 构建与调试的关系

- **Debug 构建**：用于调试段错误、内存问题，详见[Segfault 调试实战](05-debugging/01-segfault-practice.md)
- **Release 构建**：用于性能测试和生产环境
- **构建机制详解**：详见[构建与调试工具机制详解](03-mechanisms/02-build-and-debug-mechanisms.md)

---

## 常见问题

- **构建失败**：检查依赖是否安装完整，参考[环境准备与安装](00-getting-started/01-environment-setup.md)
- **测试失败**：参考[测试方法文档](06-testing/00-methodology.md)和[问题库](05-debugging/03-troubleshooting-and-known-issues.md)
- **构建慢**：考虑使用 `--build_type Release` 或并行构建

---

*如果你在阅读其它文档时遇到"从源码构建"的入口链接，统一以本文件为准。*

