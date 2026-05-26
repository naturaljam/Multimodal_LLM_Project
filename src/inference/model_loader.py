"""
模型加载模块 (4-bit 量化)
- 加载 Qwen2.5-VL 7B + bitsandbytes NF4 量化
- 适配 Kaggle T4 15GB 显存
- 单例模式，全局共享一个模型实例
"""

import torch
from typing import Optional, Tuple

from PIL import Image


_model: Optional[object] = None
_processor: Optional[object] = None


def load_model_and_processor(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    use_quantization: bool = True,
    device_map: str = "auto",
):
    """
    加载 Qwen2.5-VL 模型 + 处理器

    Args:
        model_name: HuggingFace 模型名
        use_quantization: 是否使用 4-bit 量化
        device_map: 设备分配策略

    Returns:
        (model, processor)
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[ModelLoader] 正在加载模型: {model_name}")

    # ---- 量化配置 ----
    if use_quantization:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        print("[ModelLoader] 已启用 4-bit NF4 量化")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

    # ---- 处理器 ----
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 计算显存占用（近似）
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[ModelLoader] GPU 显存已占用: {allocated:.2f} GB")

    _model = model
    _processor = processor
    return model, processor


def get_model():
    """获取已加载的模型（单例）"""
    global _model
    if _model is None:
        raise RuntimeError("模型尚未加载，请先调用 load_model_and_processor()")
    return _model


def get_processor():
    """获取已加载的处理器（单例）"""
    global _processor
    if _processor is None:
        raise RuntimeError("处理器尚未加载，请先调用 load_model_and_processor()")
    return _processor


def is_loaded() -> bool:
    """检查模型是否已加载"""
    return _model is not None and _processor is not None
