"""
Prompt 模板库
- 预设多套 Prompt 模板，对比"通用 vs 场景化"的效果
- 支持中英文模板
"""

from typing import Dict


# ============================================================
# Qwen2.5-VL 对话格式说明
# Qwen2.5-VL 使用 messages 格式：[{"role": "user", "content": [...]}]
# content 是列表，可包含 {"type": "image", "image": ...} 和 {"type": "text", "text": ...}
# 此处只管理纯文本 Prompt，图像插入由 inference/engine.py 处理
# ============================================================


PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    # ============================================================
    # 通用模板：简洁、适用于各类 VQA 场景
    # ============================================================
    "general": {
        "label": "通用模板",
        "text": "请根据图像内容回答以下问题。直接给出答案，不要解释。\n问题：{question}",
    },
    "general_en": {
        "label": "General (English)",
        "text": 'Answer the question based on the image. Give only the answer, no explanation.\nQuestion: {question}',
    },

    # ============================================================
    # 场景化模板：针对图文问答的特定场景
    # ============================================================
    "scene": {
        "label": "场景化模板",
        "text": (
            "你是一个智能图文问答助手。请仔细观察图像中的物体、场景和文字信息，"
            "回答用户的问题。\n"
            "要求：\n"
            "1. 如果问题涉及图像中的文字，请仔细识别并引用\n"
            "2. 答案应简洁、准确，不超过 5 个词\n"
            "3. 如果图像中没有相关信息，请回答\"无法判断\"\n\n"
            "问题：{question}"
        ),
    },

    # ============================================================
    # TextVQA 专用模板：强调场景文字识别
    # ============================================================
    "textvqa": {
        "label": "TextVQA 专用模板",
        "text": (
            "请仔细阅读图像中的所有文字（包括标志、标签、招牌等），"
            "然后回答以下问题。答案应基于图像中的文字内容。\n"
            "问题：{question}\n"
            "答案："
        ),
    },

    # ============================================================
    # 中文场景专用模板
    # ============================================================
    "chinese": {
        "label": "中文场景模板",
        "text": (
            "请仔细观察图像，理解其中的中文内容（如有），"
            "用中文回答以下问题。答案请简洁明了。\n"
            "问题：{question}"
        ),
    },

    # ============================================================
    # 带 OCR 信息注入的模板（增强实验用）
    # ============================================================
    "with_ocr": {
        "label": "OCR 增强模板",
        "text": (
            "{ocr_hint}"
            "请结合图像和上述文字信息回答以下问题。直接给出答案。\n"
            "问题：{question}"
        ),
    },

    # ============================================================
    # Few-shot 模板
    # ============================================================
    "fewshot": {
        "label": "Few-shot 模板",
        "text": (
            "以下是一些图像问答的示例：\n"
            "{examples}\n\n"
            "现在，请根据新图像回答以下问题。直接给出答案。\n"
            "问题：{question}"
        ),
    },
}


def get_template(name: str) -> Dict[str, str]:
    """
    获取指定名称的模板

    Args:
        name: 模板名（general / scene / textvqa / chinese / with_ocr / fewshot）

    Returns:
        {"label": "...", "text": "..."}
    """
    if name not in PROMPT_TEMPLATES:
        available = list(PROMPT_TEMPLATES.keys())
        raise ValueError(f"未知模板: {name}，可用: {available}")
    return PROMPT_TEMPLATES[name]


def list_templates() -> list:
    """列出所有可用模板"""
    return [
        {"name": k, "label": v["label"]}
        for k, v in PROMPT_TEMPLATES.items()
    ]
