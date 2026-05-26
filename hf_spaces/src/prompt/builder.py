"""
Prompt 组装模块
- 将模板、OCR 结果、Few-shot 示例拼装为最终 Prompt
- 对外暴露统一接口 build_prompt()
"""

from typing import List, Optional

from src.prompt.templates import get_template
from src.prompt.fewshot_manager import FewShotManager
from src.preprocess.ocr_extractor import format_ocr_texts
from src.utils.config import config


def build_prompt(
    question: str,
    ocr_texts: Optional[List[str]] = None,
    template_name: Optional[str] = None,
    fewshot_manager: Optional[FewShotManager] = None,
) -> str:
    """
    组装最终 Prompt

    Args:
        question: 用户提问
        ocr_texts: OCR 预提取的文字列表（可选）
        template_name: 模板名，None 则用 config 中的默认值
        fewshot_manager: Few-shot 管理器（可选）

    Returns:
        完整的 Prompt 文本字符串
    """
    template_name = template_name or config.prompt_default_template

    # ---- 确定实际使用的模板 ----
    # 如果有 OCR 结果且 OCR 启用 → 用 with_ocr 模板
    if ocr_texts and config.ocr_enabled:
        template = get_template("with_ocr")
        ocr_hint = format_ocr_texts(ocr_texts)
        prompt = template["text"].format(
            ocr_hint=ocr_hint + "\n\n" if ocr_hint else "",
            question=question,
        )
        return prompt

    # 如果启用 Few-shot
    if config.fewshot_enabled and fewshot_manager:
        template = get_template("fewshot")
        examples_str = fewshot_manager.format_examples(config.fewshot_max_examples)
        prompt = template["text"].format(
            examples=examples_str or "（无可用示例）",
            question=question,
        )
        return prompt

    # 默认：用指定模板
    template = get_template(template_name)
    prompt = template["text"].format(question=question)
    return prompt


def build_baseline_prompt(question: str) -> str:
    """快速构建 baseline 通用 Prompt（最简接口）"""
    return build_prompt(question, template_name="general")


def build_textvqa_prompt(question: str) -> str:
    """快速构建 TextVQA 专用 Prompt"""
    return build_prompt(question, template_name="textvqa")


def build_chinese_prompt(question: str) -> str:
    """快速构建中文场景 Prompt"""
    return build_prompt(question, template_name="chinese")
