# 环境信息获取命令

本文档定义了获取环境信息的标准命令。

---

## 快速参考

| 信息类型 | 命令 | 输出示例 |
|---------|------|---------|
| 服务器/NPU 型号 | `lspci -n -D \| grep '19e5:d80[23]' \| sed 's/.*d80\\([23]\\).*/A\\1/' \| head -n1` | `A3` |
| CANN 版本 | `echo $ASCEND_HOME_PATH \| grep -oP 'cann-\\K[\\d.]+'` | `8.5.0` |
| PyPTO Commit | `COMMIT=$(git merge-base HEAD $(git remote -v \| grep 'gitcode.com/cann/pypto.git' \| head -1 \| cut -f1)/master 2>/dev/null \|\| git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%h %ci' $COMMIT \|\| echo "Unknown"` | `abc123... 2025-03-10 10:00:00` |
| Python 版本 | `python --version` | `Python 3.10.12` |
| 操作系统 | `cat /etc/os-release \| grep PRETTY_NAME` | `PRETTY_NAME="Ubuntu 22.04.3 LTS"` |
| torch 版本 | `python -c "import torch; print(torch.__version__)"` | `2.6.0` |
| torch_npu 版本 | `python -c "import torch_npu; print(torch_npu.__version__)"` | `2.6.0.post3` |

---

## 详细说明

### CANN 版本

```bash
# 备选方法：从 npu-smi 获取
npu-smi info | grep Version

# 备选方法：从 CANN 安装目录
ls -d /usr/local/Ascend/ascend-toolkit/* | grep -oP '\d+\.\d+\.\d+'
```

### PyPTO Commit

```bash
# 仅获取短哈希（不依赖 remote）
git rev-parse --short HEAD

# 获取分支信息
git branch --show-current
```

### 服务器/NPU 型号

**设备ID说明**:

建议统一使用写法：**Ascend 910B** 或 **Ascend 910C**

- `19e5:d802` → A2 服务器（对应 Ascend 910B）
- `19e5:d803` → A3 服务器（对应 Ascend 910C）


### Python 版本

```bash
# 仅版本号
python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'

# 检查 Python 路径
which python
```

### 操作系统

```bash
# 发行版名称
cat /etc/os-release | grep PRETTY_NAME

# 内核版本
uname -r

# 完整系统信息
uname -a
```

## 批量获取脚本

```bash
#!/bin/bash
# 获取所有环境信息

echo "=== 环境信息 ==="
echo "CANN 版本: $(echo $ASCEND_HOME_PATH | grep -oP 'cann-\K[\d.]+')"
echo "PyPTO Commit: $(COMMIT=$(git merge-base HEAD $(git remote -v | grep 'gitcode.com/cann/pypto.git' | head -1 | cut -f1)/master 2>/dev/null || git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%h %ci' $COMMIT 2>/dev/null || echo 'Unknown')"
echo "服务器/NPU 型号: $(lspci -n -D 2>/dev/null | grep '19e5:d80[23]' | sed 's/.*d80\([23]\).*/A\1/' | head -1 || echo 'Unknown')"
echo "Python 版本: $(python --version 2>&1)"
echo "操作系统: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo 'Unknown')"
echo "torch 版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Unknown')"
echo "torch_npu 版本: $(python -c 'import torch_npu; print(torch_npu.__version__)' 2>/dev/null || echo 'Unknown')"
```

---

## 注意

- 本文件定义的所有环境信息字段名称与 [issue-templates.md](issue-templates.md) 中各模板的 Environment 章节保持一致
- 哪些 Issue 类型需要获取环境信息、获取优先级策略由 [SKILL.md](../SKILL.md) 定义
