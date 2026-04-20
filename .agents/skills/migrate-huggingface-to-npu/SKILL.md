---
name: migrate-huggingface-to-npu
description: 将大语言模型迁移到NPU环境运行。当用户想要在NPU上运行大模型、解决torch/torch-npu版本兼容问题、创建NPU推理脚本时使用此skill。适用于需要将HuggingFace模型部署到华为Ascend NPU的场景。
---

# 大语言模型迁移到NPU运行指南

## 概述

本skill记录将HuggingFace大语言模型迁移到华为Ascend NPU环境的完整步骤。

**关键要求：**
- 询问用户项目目录位置
- 在用户目录下创建模型名文件夹，包含models和scripts两个子目录
- models目录直接存放模型文件（无hub层级）
- scripts目录存放运行脚本和文档
- 检查并验证ask脚本能否运行

## 完整迁移流程（7步）

### 步骤1：检查NPU环境与内存预估

**1. 检查NPU状态和内存容量：**
```bash
npu-smi info
```

Ascend910 单卡内存：**64GB HBM**

**2. 预估模型内存占用（关键！）：**

**计算公式：**
```
模型内存 (GB) = 参数量 (B) × 精度字节数

精度字节数：float16=2, float32=4, int8=1
总占用 ≈ 模型权重 + KV缓存(约20%权重) + 系统开销(2GB)
```

**快速估算表（单卡64GB）：**
| 模型参数 | float16占用 | 单卡能否运行 |
|---------|------------|-------------|
| 3B | ~7GB | ✅ 可以 |
| 7B | ~17GB | ✅ 可以 |
| 13B | ~30GB | ✅ 可以 |
| 30B | ~60GB | ✅ 勉强 |
| 70B+ | >140GB | ❌ 需多卡 |

**实际验证：**
```python
import torch_npu
torch.npu.set_device(0)
mem = torch.npu.get_device_properties(0).total_memory / 1024**3
print(f"NPU内存: {mem:.1f}GB")
# 若模型预估占用 > mem × 0.8，则单卡无法运行
```

### 步骤2：安装依赖（版本匹配是关键）

**torch和torch-npu版本必须完全一致！**

```bash
pip install torch==2.7.1 torch-npu==2.7.1
pip install transformers accelerate sentencepiece protobuf

# 验证安装
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}'); print(f'NPU可用: {torch.npu.is_available()}')"
```

### 步骤3：下载模型

**重要：首先询问用户项目目录位置！**

询问：项目要放在哪个目录下？（如：`/home/user/projects` 或 `/data/llm`）

**简化的目录结构：**
```
用户指定目录/
└── 模型名/                    # 以模型名命名的项目文件夹
    ├── models/                # 模型缓存目录（必须）
    │   ├── config.json        # 必须
    │   ├── model.safetensors  # 必须
    │   ├── tokenizer.json     # 必须
    │   └── download.log       # 可选
    │
    └── scripts/               # 脚本目录
        ├── ask_模型名.py      # 必须（核心）
        ├── deploy_模型名.py   # 可选
        ├── README.md          # 可选
        └── run.sh             # 可选
```

**下载模型到models目录（使用local_dir直接下载）：**
```bash
# 设置镜像（如果网络不通）
export HF_ENDPOINT=https://hf-mirror.com

# 创建目录
mkdir -p /用户指定目录/模型名/models
mkdir -p /用户指定目录/模型名/scripts

# 后台下载
nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='模型ID',
    local_dir='/用户指定目录/模型名/models',
    local_dir_use_symlinks=False
)
" > /用户指定目录/模型名/models/download.log 2>&1 &
```

**检查下载进度：**
```bash
ps aux | grep snapshot_download
du -sh 项目目录/models/
ls -la 项目目录/models/
```

### 步骤4：创建ask脚本

**脚本存放在scripts目录：** `scripts/ask_模型名.py`

**核心要点：**
1. 所有参数有默认值，可直接运行
2. 默认模型路径：`../models`（脚本所在目录的父目录）
3. 使用 `local_files_only=True` 离线加载
4. 使用 `torch.npu.set_device(device)` 指定NPU

**关键代码片段：**

```python
import argparse
import os
import torch
import torch_npu

# 参数定义（都有默认值）
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="你好，请介绍一下你自己")
parser.add_argument("--device", default=15, help="NPU卡号")
parser.add_argument("--model-path", default=None)
args = parser.parse_args()

# 自动获取模型路径（脚本目录的父目录下的models）
if not args.model_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_path = os.path.join(os.path.dirname(script_dir), "models")

# 设置NPU并加载模型
torch.npu.set_device(args.device)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"},
    local_files_only=True,
)
```

**可选：使用生成工具**
```bash
python3 scripts/generate_ask_script.py --model-id "组织名/模型名" --model-name "模型简称"
```

### 步骤5：检查并验证脚本

```bash
# 检查scripts目录下是否生成ask脚本
ls -la scripts/ask_模型名.py

# 检查脚本内容
head -20 scripts/ask_模型名.py
python3 scripts/ask_模型名.py --help

# 验证脚本能否运行
cd 项目目录/scripts
python3 ask_模型名.py
```

**验证通过标准：**
- 脚本成功加载模型
- 模型加载到指定NPU卡
- 生成回复并输出
- 无错误退出

### 步骤6：运行测试

```bash
cd 项目目录/scripts

# 使用所有默认参数
python3 ask_模型名.py

# 指定问题
python3 ask_模型名.py --prompt "1+1等于几？"

# 指定NPU卡号
python3 ask_模型名.py --device 7

# 指定模型路径
python3 ask_模型名.py --model-path ../models

# 组合使用
python3 ask_模型名.py --prompt "你好" --device 15 --model-path /custom/path
```

### 步骤7：完善项目结构

**最终目录结构：**
```
用户指定目录/
└── 模型名/
    ├── models/                    # 模型缓存（必须）
    │   ├── config.json            # 模型配置（必须）
    │   ├── model.safetensors      # 模型权重（必须）
    │   ├── tokenizer.json         # 分词器（必须）
    │   └── download.log           # 下载日志（可选）
    │
    └── scripts/
        ├── ask_模型名.py          # 单次问答脚本（必须，核心）
        ├── deploy_模型名.py       # 交互式部署脚本（可选）
        ├── README.md              # 说明文档（可选）
        └── run.sh                 # 启动脚本（可选）
```

**必须文件：**
- `models/` 目录及其中的模型权重文件
- `scripts/ask_模型名.py` 单次问答脚本

**可选文件：**
- `deploy_模型名.py` 交互式部署脚本
- `README.md` 说明文档
- `run.sh` 启动脚本
- `download.log` 下载日志

**路径关系：**
```
项目根目录 = 用户指定目录/模型名
模型路径 = 项目根目录/models/           # 直接指向models目录（无hub）
脚本路径 = 项目根目录/scripts/

ask脚本位置：scripts/ask_模型名.py
ask脚本引用：MODEL_PATH = ../models 或 项目根目录/models
```

## 常见问题解决

### Q1: transformers报错 "PyTorch >= 2.4 is required"
升级torch和torch-npu到2.7.1

### Q2: torch_npu报错 "undefined symbol"
确保torch和torch-npu版本完全一致

### Q3: "Network is unreachable"
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q4: 模型加载慢
使用 `local_dir_use_symlinks=False` 和 `local_files_only=True`

## 用户核心要求总结

### 1. 明确询问项目目录位置
开始前询问：项目要放在哪个目录下？

### 2. 统一的项目目录结构
```
用户指定目录/
└── 模型名/
    ├── models/                # 必须：模型缓存
    │   ├── config.json        # 必须
    │   ├── model.safetensors  # 必须
    │   └── tokenizer.json     # 必须
    │
    └── scripts/               
        ├── ask_模型名.py      # 必须：单次问答脚本
        ├── deploy_模型名.py   # 可选：交互式部署
        ├── README.md          # 可选：说明文档
        └── run.sh             # 可选：启动脚本
```

**必须文件：**
- `models/` 目录及模型权重文件
- `scripts/ask_模型名.py` 问答脚本

**可选文件：**
- `deploy_模型名.py`、`README.md`、`run.sh`

### 3. 命令行格式标准化
```bash
python3 scripts/ask_模型名.py --prompt "问题" --device 卡号 [--model-path 路径]
```

### 4. 所有参数有默认值
脚本可直接运行：`python3 ask_模型名.py`

### 5. 支持NPU卡号指定
`python3 ask_模型名.py --device 15`

### 6. 支持模型路径指定
`python3 ask_模型名.py --model-path ../models`

### 7. 离线运行能力
从本地models目录加载，不依赖网络

### 8. 检查验证脚本运行
- 检查 `scripts/ask_模型名.py` 是否生成
- 验证脚本能否成功运行
- 确保无错误退出