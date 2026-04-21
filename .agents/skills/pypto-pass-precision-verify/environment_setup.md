# PyPTO Pass 精度验证 - 环境配置指南

本文档介绍 PyPTO Pass 精度验证所需的环境依赖和配置。

---

## 环境依赖检查

在开始使用本技能前，请检查以下环境依赖：

### 必需依赖

- [ ] CANN 工具链已安装
- [ ] PyPTO 已正确安装
- [ ] 编译环境（g++, make）可用
- [ ] Python 环境版本 >= 3.8
- [ ] torch 版本 >= 2.1.0
- [ ] 头文件 `pto/comm/pto_comm_inst.hpp` 存在

### 可选依赖

- [ ] NPU 硬件（用于 NPU 模式测试）
- [ ] SIM 模式环境（用于模拟测试）

### 依赖检查方法

```bash
# 检查 Python 版本
python3 --version

# 检查 torch 版本
python3 -c "import torch; print(f'torch version: {torch.__version__}')"

# 检查 PyPTO 安装
python3 -c "import pypto; print('PyPTO installed')"

# 检查编译环境
g++ --version
make --version

# 检查头文件
find /usr/include -name "pto_comm_inst.hpp" 2>/dev/null
find ~/.local -name "pto_comm_inst.hpp" 2>/dev/null
```

### 依赖缺失处理

**如果缺少 CANN 工具链**：
1. 参考 CANN 安装文档进行安装
2. 设置 CANN 相关环境变量

**如果缺少头文件**：
1. 检查 CANN 安装是否完整
2. 重新安装 CANN 或 PyPTO
3. 检查编译器的 include 路径设置

---

## 环境变量配置

以下环境变量必须在使用本技能前设置：

### 必需环境变量

```bash
# 必需：设置工作目录
export ASCEND_WORK_PATH="/path/to/work/directory"

# 必需：设置日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=0
```

### 可选环境变量

```bash
# 可选：设置设备 ID（NPU 模式必需）
export TILE_FWK_DEVICE_ID=0

# 可选：设置 PyPTO 日志级别
export PYPTO_LOG_LEVEL=0
```

### 验证环境变量设置

```bash
# 验证必需环境变量
echo "ASCEND_WORK_PATH: $ASCEND_WORK_PATH"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"

# 验证可选环境变量
echo "TILE_FWK_DEVICE_ID: $TILE_FWK_DEVICE_ID"
echo "PYPTO_LOG_LEVEL: $PYPTO_LOG_LEVEL"
```

### 环境变量设置时机

| 环境变量 | 设置时机 | 说明 |
|---------|---------|------|
| `ASCEND_WORK_PATH` | 运行测试前必须设置 | 组件日志输出目录 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为 0（DEBUG） | 获取详细调试信息 |
| `TILE_FWK_DEVICE_ID` | NPU 模式运行前必须设置 | 指定 NPU 设备 ID |
| `PYPTO_LOG_LEVEL` | 可选设置 | PyPTO 特定日志级别 |

---

## 快速检查脚本

一键检查所有环境配置：

```bash
#!/bin/bash
echo "=== PyPTO Pass 精度验证环境检查 ==="

# 检查 Python
echo "Python 版本: $(python3 --version)"

# 检查 torch
python3 -c "import torch; print('torch 版本:', torch.__version__)" || echo "torch 未安装"

# 检查 PyPTO
python3 -c "import pypto; print('PyPTO: 已安装')" || echo "PyPTO 未安装"

# 检查编译环境
echo "g++: $(g++ --version | head -1)"
echo "make: $(make --version | head -1)"

# 检查环境变量
echo "ASCEND_WORK_PATH: ${ASCEND_WORK_PATH:-未设置}"
echo "ASCEND_GLOBAL_LOG_LEVEL: ${ASCEND_GLOBAL_LOG_LEVEL:-未设置}"
echo "TILE_FWK_DEVICE_ID: ${TILE_FWK_DEVICE_ID:-未设置}"

# 检查头文件
if find /usr/include -name "pto_comm_inst.hpp" 2>/dev/null | head -1; then
    echo "pto_comm_inst.hpp: 已找到"
else
    echo "pto_comm_inst.hpp: 未找到"
fi
```

---

## 相关文档

- 主流程文档：[SKILL.md](./SKILL.md)
- 配置指南：[config-guide.md](./config-guide.md)