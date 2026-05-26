"""
分辨率适配模块
- Qwen2.5-VL 要求输入图像宽高均为 28 的倍数
- 使用 pad（填充）策略保持原始宽高比
- 控制最大像素数以适配显存
"""

import math
from typing import Tuple

from PIL import Image


# Qwen2.5-VL patch_size = 14，但视觉编码器要求输入为 28 的倍数
_FACTOR = 28


def resize_to_vl(
    image: Image.Image,
    max_pixels: int = 16384,
) -> Image.Image:
    """
    将图像适配到 Qwen2.5-VL 的输入要求

    策略：
    1. 如果图像像素数超过 max_pixels，先等比缩放
    2. 将宽高 pad 到 28 的倍数（右下角填充黑色）

    Args:
        image: 输入图像 (PIL.Image, RGB)
        max_pixels: 最大像素数上限（默认 16384）

    Returns:
        适配后的图像
    """
    w, h = image.size
    total_pixels = w * h

    # Step 1: 如果超过最大像素数，等比缩放
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 LANCZOS 高质量重采样
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    # Step 2: Pad 到 28 的倍数
    new_w = _round_up(w, _FACTOR)
    new_h = _round_up(h, _FACTOR)

    if new_w != w or new_h != h:
        padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        padded.paste(image, (0, 0))
        image = padded

    return image


def get_adjusted_size(
    w: int,
    h: int,
    max_pixels: int = 16384,
) -> Tuple[int, int]:
    """
    仅计算适配后的尺寸（不实际处理图像），用于预估

    Returns:
        (new_width, new_height)
    """
    total_pixels = w * h
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        w = int(w * scale)
        h = int(h * scale)

    w = _round_up(w, _FACTOR)
    h = _round_up(h, _FACTOR)
    return w, h


def _round_up(value: int, factor: int) -> int:
    """向上取整到 factor 的倍数"""
    return int(math.ceil(value / factor) * factor)
