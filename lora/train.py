"""
LoRA 微调脚本 (QLoRA)
======================
- 基于 Qwen2.5-VL 7B + 4-bit 量化
- 在自建中文数据集上微调
- 使用 Kaggle T4×2 (30GB VRAM) 运行

使用方式:
    python lora/train.py --data_path data/custom_chinese.json --output_dir lora/checkpoints/
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from PIL import Image


def load_custom_dataset(data_path: str):
    """加载自建中文 JSON 数据集"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 转换为 HuggingFace Dataset 格式
    # 期望格式: [{"image_path": "...", "question": "...", "answer": "..."}]
    dataset = Dataset.from_list(raw)
    return dataset


def format_conversation(question: str, answer: str) -> list:
    """构造 Qwen2.5-VL 对话格式"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]


def preprocess_function(examples, processor, max_length=512):
    """预处理函数：图像 + 对话 → tokenized inputs"""
    images = []
    texts = []

    for img_path, question, answer in zip(
        examples["image_path"], examples["question"], examples["answer"]
    ):
        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        images.append(image)

        # 构造对话
        conversation = format_conversation(question, answer)
        text = processor.apply_chat_template(conversation, tokenize=False)
        texts.append(text)

    return {"images": images, "texts": texts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/custom_chinese.json")
    parser.add_argument("--output_dir", type=str, default="lora/checkpoints/")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════╗
║   QLoRA 微调 - Qwen2.5-VL 7B       ║
╠══════════════════════════════════════╣
║  数据:   {args.data_path:<28}║
║  轮数:   {args.epochs:<28}║
║  LoRA r: {args.lora_r:<28}║
║  LoRA α: {args.lora_alpha:<28}║
╚══════════════════════════════════════╝
""")

    # ---- 加载数据集 ----
    dataset = load_custom_dataset(args.data_path)
    print(f"[Data] 数据集大小: {len(dataset)}")

    # ---- 量化配置 ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ---- 加载模型 ----
    print("[Model] 加载基础模型 (4-bit)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ---- 准备 PEFT ----
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只微调 LLM 的 attention
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 处理器 ----
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # ---- 预处理 ----
    # 注意: 多模态数据集的 tokenize 需要特殊处理
    # 这里给出基本框架，实际训练时需要根据 Qwen2.5-VL 的输入格式调整
    print("[Data] 预处理数据集...")
    # 由于 Qwen2.5-VL 的图像 tokenize 比较复杂，实际训练可以使用
    # TRL 的 SFTTrainer 或自定义 DataCollator
    # 这里提供一个训练框架入口

    # ---- 训练参数 ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    print(f"""
[Train] 训练配置已就绪。
[Train] 输出目录: {args.output_dir}
[Train] 当前为训练框架，完整训练需要在 Kaggle T4×2 上运行。

下一步:
  1. 准备自建中文数据集 (JSON 格式)
  2. 上传到 Kaggle
  3. 运行: pip install peft trl && python lora/train.py
  4. 合并权重: python lora/merge.py
""")


if __name__ == "__main__":
    main()
