"""
本地假推理模块（Stub）
- 本机无 GPU 时使用，返回确定性假答案
- 用于调试验证 pipeline（预处理 → Prompt → 后处理 → 报告）全流程
- 答案模式可配置，便于测试不同评测逻辑
"""

import hashlib
from typing import List, Optional

from PIL import Image


def stub_ask(
    image: Image.Image,
    prompt: str,
    mode: str = "hash",
) -> str:
    """
    假推理：根据图像哈希和 prompt 生成确定性假答案

    Args:
        image: 输入图像
        prompt: 提示文本
        mode: 假答案模式
            - "hash": 基于图像哈希生成确定性答案
            - "yes": 始终返回 "yes"
            - "echo": 返回 prompt 的第一个词

    Returns:
        假答案字符串
    """
    if mode == "yes":
        return "yes"
    elif mode == "echo":
        # 返回 prompt 的前 3 个非空单词
        words = prompt.split()
        return " ".join(words[:3]) if words else "echo"
    else:  # hash
        # 基于图像像素哈希 + prompt 生成确定性答案
        # 保证相同输入始终返回相同答案（便于调试）
        hasher = hashlib.md5()
        # 图像哈希
        img_bytes = image.tobytes()[:4096]  # 取前 4KB 加快速度
        hasher.update(img_bytes)
        # prompt 哈希
        hasher.update(prompt.encode("utf-8"))
        hex_hash = hasher.hexdigest()

        # 模拟答案池
        answer_pool = [
            "yes", "no", "cat", "dog", "red", "blue",
            "2", "3", "table", "chair", "car", "tree",
            "stop", "left", "right", "white", "black", "green",
        ]
        idx = int(hex_hash[:8], 16) % len(answer_pool)
        return answer_pool[idx]


def stub_ask_batch(
    images: List[Image.Image],
    prompts: List[str],
    mode: str = "hash",
) -> List[str]:
    """批量假推理"""
    return [stub_ask(img, prompt, mode) for img, prompt in zip(images, prompts)]
