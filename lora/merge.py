"""
LoRA 权重合并脚本
- 将微调后的 LoRA adapter 合并到基础模型
- 输出完整可用的模型权重
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="lora/checkpoints/checkpoint-XXX")
    parser.add_argument("--output_path", type=str, default="lora/merged_model/")
    args = parser.parse_args()

    print(f"[Merge] 基础模型: {args.base_model}")
    print(f"[Merge] LoRA 权重: {args.lora_path}")
    print(f"[Merge] 输出路径: {args.output_path}")

    # ---- 量化配置（同训练时） ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ---- 加载基础模型 ----
    print("[Merge] 加载基础模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ---- 加载 LoRA adapter ----
    print("[Merge] 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    # ---- 合并权重 ----
    print("[Merge] 合并权重中...")
    model = model.merge_and_unload()

    # ---- 保存 ----
    print(f"[Merge] 保存到 {args.output_path}...")
    model.save_pretrained(args.output_path)

    # 同时保存 processor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.output_path)

    print(f"[Merge] ✅ 完成！合并后的模型已保存到: {args.output_path}")


if __name__ == "__main__":
    main()
