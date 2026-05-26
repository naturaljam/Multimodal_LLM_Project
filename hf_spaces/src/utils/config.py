"""
全局配置管理模块
- 加载 config.yaml
- 自动检测运行平台（本地 / Kaggle / Colab / HuggingFace Space）
- 提供全局单例 Config，所有模块通过 `from src.utils.config import config` 引用
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# 平台检测
# ============================================================

def _detect_platform() -> str:
    """自动检测当前运行环境"""
    # Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        return "huggingface_space"

    # Kaggle
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle"):
        return "kaggle"

    # Google Colab
    try:
        import google.colab  # type: ignore
        return "colab"
    except ImportError:
        pass

    return "local"


# ============================================================
# Config 单例
# ============================================================

class Config:
    """全局配置单例"""

    _instance: Optional["Config"] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.platform: str = _detect_platform()
        self._data: Dict[str, Any] = {}

        # 找到 config.yaml 路径
        self.config_path = self._find_config()
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

    def _find_config(self) -> str:
        """查找 config.yaml，适配不同平台的目录结构"""
        # Kaggle: 项目解压后可能在 /kaggle/working/
        search_paths = [
            Path(__file__).parent.parent.parent / "config.yaml",  # 本地: project/config.yaml
            Path("/kaggle/working/config.yaml"),                   # Kaggle
            Path("/content/config.yaml"),                          # Colab
        ]
        for p in search_paths:
            if p.exists():
                return str(p)
        # 最后回退到相对路径
        return str(Path(__file__).parent.parent.parent / "config.yaml")

    # ---- 便捷属性 ----

    @property
    def is_local(self) -> bool:
        return self.platform == "local"

    @property
    def is_kaggle(self) -> bool:
        return self.platform == "kaggle"

    @property
    def is_colab(self) -> bool:
        return self.platform == "colab"

    @property
    def is_cloud(self) -> bool:
        """是否在云端 GPU 环境（Kaggle / Colab）"""
        return self.platform in ("kaggle", "colab")

    # ---- 模型配置 ----

    @property
    def model_name(self) -> str:
        return self._data.get("model", {}).get("name", "Qwen/Qwen2.5-VL-7B-Instruct")

    @property
    def model_quantization(self) -> str:
        return self._data.get("model", {}).get("quantization", "nf4")

    @property
    def model_max_new_tokens(self) -> int:
        return self._data.get("model", {}).get("max_new_tokens", 128)

    @property
    def model_temperature(self) -> float:
        return self._data.get("model", {}).get("temperature", 0.0)

    # ---- 预处理配置 ----

    @property
    def resize_strategy(self) -> str:
        return self._data.get("preprocess", {}).get("resize", {}).get("strategy", "pad")

    @property
    def resize_max_pixels(self) -> int:
        return self._data.get("preprocess", {}).get("resize", {}).get("max_pixels", 16384)

    @property
    def ocr_enabled(self) -> bool:
        return self._data.get("preprocess", {}).get("ocr", {}).get("enabled", False)

    @property
    def ocr_backend(self) -> str:
        return self._data.get("preprocess", {}).get("ocr", {}).get("backend", "easyocr")

    @property
    def ocr_lang_list(self) -> list:
        return self._data.get("preprocess", {}).get("ocr", {}).get("lang_list", ["ch_sim", "en"])

    # ---- Prompt 配置 ----

    @property
    def prompt_default_template(self) -> str:
        return self._data.get("prompt", {}).get("default_template", "general")

    @property
    def fewshot_enabled(self) -> bool:
        return self._data.get("prompt", {}).get("fewshot", {}).get("enabled", False)

    @property
    def fewshot_max_examples(self) -> int:
        return self._data.get("prompt", {}).get("fewshot", {}).get("max_examples", 3)

    @property
    def fewshot_examples_file(self) -> str:
        return self._data.get("prompt", {}).get("fewshot", {}).get("examples_file", "data/fewshot_examples.json")

    # ---- 数据集配置 ----

    @property
    def vqa_v2_name(self) -> str:
        return self._data.get("datasets", {}).get("vqa_v2", {}).get("name", "HuggingFaceM4/VQAv2")

    @property
    def vqa_v2_subset_size(self) -> int:
        return self._data.get("datasets", {}).get("vqa_v2", {}).get("subset_size", 1000)

    @property
    def textvqa_name(self) -> str:
        return self._data.get("datasets", {}).get("textvqa", {}).get("name", "lmms-lab/textvqa")

    @property
    def textvqa_subset_size(self) -> int:
        return self._data.get("datasets", {}).get("textvqa", {}).get("subset_size", 500)

    @property
    def custom_chinese_path(self) -> str:
        return self._data.get("datasets", {}).get("custom_chinese", {}).get("path", "data/custom_chinese.json")

    @property
    def custom_chinese_subset_size(self) -> int:
        return self._data.get("datasets", {}).get("custom_chinese", {}).get("subset_size", 100)

    # ---- 输出配置 ----

    @property
    def results_dir(self) -> str:
        return self._data.get("output", {}).get("results_dir", "experiments/results/")

    @property
    def output_formats(self) -> list:
        return self._data.get("output", {}).get("formats", ["json", "csv"])

    # ---- 通用方法 ----

    def get(self, *keys: str, default: Any = None) -> Any:
        """安全获取嵌套配置项，如 config.get('model', 'name')"""
        value = self._data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def __repr__(self) -> str:
        return f"Config(platform={self.platform}, model={self.model_name})"


# ============================================================
# 全局单例
# ============================================================

config = Config()
