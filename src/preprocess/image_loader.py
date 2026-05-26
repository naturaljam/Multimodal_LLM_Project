"""
图像加载模块
- 从本地路径加载
- 从 URL 加载
- 从 HuggingFace datasets 的 PIL.Image 字段直接使用
- 统一返回 PIL.Image（RGB）
"""

from io import BytesIO
from pathlib import Path
from typing import Union, Optional

import requests
from PIL import Image


def load_image(
    source: Union[str, Path, Image.Image, bytes],
    timeout: int = 10,
) -> Image.Image:
    """
    统一图像加载入口

    Args:
        source: 图像来源，支持：
            - PIL.Image 对象 → 直接返回
            - 本地文件路径 (str/Path)
            - HTTP/HTTPS URL
            - bytes 原始数据
        timeout: URL 下载超时（秒）

    Returns:
        PIL.Image (RGB 模式)
    """
    # 已经是 PIL.Image
    if isinstance(source, Image.Image):
        return _ensure_rgb(source)

    # bytes 原始数据
    if isinstance(source, bytes):
        img = Image.open(BytesIO(source))
        return _ensure_rgb(img)

    # Path 对象
    if isinstance(source, Path):
        source = str(source)

    # 字符串：URL 或本地路径
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return _load_from_url(source, timeout)
        else:
            return _load_from_disk(source)

    raise TypeError(f"不支持的图像来源类型: {type(source)}")


def _load_from_disk(path: str) -> Image.Image:
    """从本地磁盘加载"""
    img = Image.open(path)
    return _ensure_rgb(img)


def _load_from_url(url: str, timeout: int = 10) -> Image.Image:
    """从 URL 下载并加载"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    return _ensure_rgb(img)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """确保图像为 RGB 模式"""
    if img.mode == "RGBA":
        # 创建白色背景，合成 RGBA → RGB
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img
