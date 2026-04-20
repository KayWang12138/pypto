---
name: migrate-huggingface-to-npu
description: 将大语言模型迁移到NPU环境运行。当用户想要在NPU上运行大模型、解决torch/torch-npu版本兼容问题、创建NPU推理脚本时使用此skill。适用于需要将HuggingFace模型部署到华为Ascend NPU的场景。
---

# 大语言模型迁移到NPU运行指南

## 概述

本skill记录将HuggingFace大语言模型迁移到华为Ascend NPU环境的完整步骤。

**强制要求：** 必须使用真实NPU硬件，最终验证推理脚本成功运行。

**关键要求：**
- **询问HuggingFace模型链接**（如：https://huggingface.co/Qwen/Qwen2-7B）
- 询问项目目录位置
- 强制检查NPU环境可用性
- 验证ask脚本在NPU上成功运行

## 完整迁移流程（7步）

### 步骤0：获取用户信息

**必须询问：**
1. HuggingFace模型链接 → 提取 `repo_id`（如：Qwen/Qwen2-7B）和 `model_name`（如：Qwen2-7B）
2. 项目目录位置（如：/data/llm）

### 步骤1：检查NPU环境与内存预估

**1. 检查NPU状态：**
```bash
npu-smi info
```

**验证标准：** 输出显示NPU设备列表，至少一张卡可用。失败则使用 `pypto-environment-setup` skill。

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

### 步骤3：项目目录结构

**创建目录：**
```bash
mkdir -p {project_dir}/{model_name}/models
mkdir -p {project_dir}/{model_name}/scripts
```

**目录结构：**
```
{project_dir}/{model_name}/
├── models/                    # 模型缓存（必须）
│   ├── config.json            # 模型配置（必须）
│   ├── model.safetensors      # 模型权重（必须）
│   ├── tokenizer.json         # 分词器（必须）
│   └── download.log           # 下载日志（可选）
└── scripts/
    ├── ask_{model_name}.py    # 单次问答脚本（必须，核心）
    ├── deploy_{model_name}.py # 交互式部署脚本（可选）
    ├── README.md              # 说明文档（可选）
    └── run.sh                 # 启动脚本（可选）
```

**路径关系：**
```
项目根目录 = {project_dir}/{model_name}
模型路径 = 项目根目录/models/
脚本路径 = 项目根目录/scripts/
```

### 步骤4：下载模型

**下载模型：**
```bash
export HF_ENDPOINT=https://hf-mirror.com

nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='{repo_id}',
    local_dir='{project_dir}/{model_name}/models',
    local_dir_use_symlinks=False
)
" > {project_dir}/{model_name}/models/download.log 2>&1 &
```

**检查下载进度：**
```bash
ps aux | grep snapshot_download
du -sh {project_dir}/{model_name}/models/
ls -la {project_dir}/{model_name}/models/
```

### 步骤5：创建ask脚本

**脚本命名：** `scripts/ask_{model_name}.py`

**核心要点：**
1. 所有参数有默认值，可直接运行
2. 默认模型路径：`../models`
3. 使用 `local_files_only=True` 离线加载
4. 使用 `torch.npu.set_device(device)` 指定NPU

**关键代码片段：**

```python
import argparse
import os
import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="你好，请介绍一下你自己")
parser.add_argument("--device", default=15, help="NPU卡号")
parser.add_argument("--model-path", default=None)
args = parser.parse_args()

if not args.model_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_path = os.path.join(os.path.dirname(script_dir), "models")

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
python3 scripts/generate_ask_script.py \
    --model-id "{repo_id}" \
    --model-name "{model_name}" \
    --project-dir "{project_dir}/{model_name}"
```

### 步骤6：检查并验证脚本

```bash
ls -la scripts/ask_{model_name}.py
head -20 scripts/ask_{model_name}.py
python3 scripts/ask_{model_name}.py --help

cd {project_dir}/{model_name}/scripts
python3 ask_{model_name}.py
```

**验证通过标准：**
- 脚本成功加载模型到 NPU
- 输出显示正在使用 NPU 设备
- 生成回复并输出
- 无错误退出

### 步骤7：运行测试

```bash
cd {project_dir}/{model_name}/scripts

python3 ask_{model_name}.py
python3 ask_{model_name}.py --prompt "1+1等于几？"
python3 ask_{model_name}.py --device 7
python3 ask_{model_name}.py --model-path ../models
python3 ask_{model_name}.py --prompt "你好" --device 15 --model-path /custom/path
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

### 2. 命令行格式标准化
```bash
python3 scripts/ask_{model_name}.py --prompt "问题" --device 卡号 [--model-path 路径]
```

### 3. 所有参数有默认值
脚本可直接运行：`python3 ask_{model_name}.py`

### 4. 支持NPU卡号指定
`python3 ask_{model_name}.py --device 15`

### 5. 支持模型路径指定
`python3 ask_{model_name}.py --model-path ../models`

### 6. 离线运行能力
从本地models目录加载，不依赖网络

### 7. 检查验证脚本运行
- 检查 `scripts/ask_{model_name}.py` 是否生成
- 验证脚本能否成功运行
- 确保无错误退出