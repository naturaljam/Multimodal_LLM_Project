"""
OCR 预提取模块
- 增强实验阶段使用
- 从图像中提取文字信息，拼入 Prompt 提升 TextVQA 性能
- 默认 disabled，由 config.yaml 中 preprocess.ocr.enabled 控制
"""

from typing import List, Optional

from PIL import Image


# 延迟导入，避免本地无 GPU 时 easyocr 导入失败
_reader: Optional[object] = None


def _get_reader(lang_list: Optional[List[str]] = None):
    """延迟初始化 EasyOCR Reader（单例）"""
    global _reader
    if _reader is None:
        import easyocr
        if lang_list is None:
            lang_list = ["ch_sim", "en"]
        _reader = easyocr.Reader(lang_list, gpu=True)
    return _reader


def extract_texts(
    image: Image.Image,
    lang_list: Optional[List[str]] = None,
) -> List[str]:
    """
    从图像中提取文字行（OCR）

    Args:
        image: 输入图像 (PIL.Image, RGB)
        lang_list: 语言列表，默认 ["ch_sim", "en"]

    Returns:
        检测到的文字列表，按从上到下、从左到右排列
        e.g. ["STOP", "停", "限速 60"]
    """
    reader = _get_reader(lang_list)
    # EasyOCR 接受 numpy array
    import numpy as np
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0)  # detail=0 只返回文字
    return results


def format_ocr_texts(texts: List[str], max_chars: int = 500) -> str:
    """
    将 OCR 结果格式化为可拼入 Prompt 的字符串

    Args:
        texts: OCR 提取的文字列表
        max_chars: 最大字符数，超出则截断

    Returns:
        格式化后的字符串，例如：
        "图像中检测到的文字：\n- STOP\n- 限速 60\n- 出口"
    """
    if not texts:
        return ""

    lines = ["图像中检测到的文字："]
    char_count = 0
    for t in texts:
        t = t.strip()
        if not t:
            continue
        line = f"- {t}"
        char_count += len(line) + 1
        if char_count > max_chars:
            lines.append("- ...（后续文字已截断）")
            break
        lines.append(line)

    return "\n".join(lines)
