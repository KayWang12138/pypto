#!/usr/bin/env python3
"""
根据模板生成大语言模型问答脚本
用法: python3 generate_ask_script.py --model-name "模型名" --project-dir "/项目根目录"
示例: python3 generate_ask_script.py --model-name "nanbeige" --project-dir "/data/llm/Nanbeige4.1-3B"
"""

import argparse
import os

parser = argparse.ArgumentParser(description="生成大语言模型问答脚本")
parser.add_argument("--model-name", required=True, help="模型简称 (如: nanbeige)")
parser.add_argument(
    "--project-dir", required=True, help="项目根目录 (如: /data/llm/Nanbeige4.1-3B)"
)
args = parser.parse_args()

# 使用字符串拼接构建内容
content = f'''#!/usr/bin/env python3
"""
{args.model_name} 单次问答脚本
用法: python3 ask_{args.model_name}.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
"""

import argparse
import os
import sys
import torch

parser = argparse.ArgumentParser(description="{args.model_name} 问答脚本")
parser.add_argument("--prompt", default="你好，请介绍一下你自己", help="问题")
parser.add_argument("--device", default=15, help="NPU卡号")
parser.add_argument("--model-path", default=None, help="模型路径")
args = parser.parse_args()

# 自动推导模型路径（脚本目录的父目录下的models）
if args.model_path:
    MODEL_PATH = args.model_path
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(os.path.dirname(script_dir), "models")

# 检查模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"错误: 模型路径不存在: {{MODEL_PATH}}")
    sys.exit(1)

# 设置NPU
import torch_npu
torch.npu.set_device(args.device)

# 加载模型（离线）
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"加载模型到 NPU:{{args.device}}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={{"": f"npu:{{args.device}}"}},
    local_files_only=True,
)
model.eval()

# 生成回复
messages = [{{"role": "user", "content": args.prompt}}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(f"npu:{{args.device}}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
'''

# 输出到 scripts 子目录
scripts_dir = os.path.join(args.project_dir, "scripts")
os.makedirs(scripts_dir, exist_ok=True)

output_file = os.path.join(scripts_dir, f"ask_{args.model_name}.py")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

print(f"脚本已生成: {output_file}")
print(f"\n使用方法:")
print(f"  cd {scripts_dir}")
print(f"  python3 ask_{args.model_name}.py")
print(f"  python3 ask_{args.model_name}.py --prompt '你的问题'")
print(f"  python3 ask_{args.model_name}.py --device 7")
print(f"  python3 ask_{args.model_name}.py --model-path /custom/path")
