"""
推理引擎
- 单图问答接口
- 批量推理接口
- 使用 Qwen2.5-VL 的 messages 格式
"""

import time
from typing import List, Optional

import torch
from PIL import Image

from src.inference.model_loader import get_model, get_processor, is_loaded


def ask(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """
    单图问答

    Args:
        image: 输入图像 (PIL.Image, RGB)
        prompt: 文本提示
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度（0.0 = 贪心解码）

    Returns:
        模型生成的回答文本
    """
    if not is_loaded():
        raise RuntimeError("模型未加载，请先调用 load_model_and_processor()")

    model = get_model()
    processor = get_processor()

    # ---- 构建 Qwen2.5-VL messages ----
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # ---- 处理输入 ----
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # ---- 推理 ----
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
        )

    # 去掉输入部分，只保留生成的 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return answer.strip()


def ask_batch(
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    batch_size: int = 1,
) -> List[str]:
    """
    批量问答（逐个推理，显存安全）

    Args:
        images: 图像列表
        prompts: 对应的提示列表
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度
        batch_size: 暂未启用真正的 batching，预留参数

    Returns:
        回答列表
    """
    answers = []
    total = len(images)

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        t0 = time.time()
        try:
            ans = ask(img, prompt, max_new_tokens, temperature)
            elapsed = time.time() - t0
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] {elapsed:.1f}s | {ans[:60]}...")
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR: {e}")
            ans = ""
        answers.append(ans)

    return answers
