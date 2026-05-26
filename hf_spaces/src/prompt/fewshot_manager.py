"""
Few-shot 示例管理器
- 从 JSON 文件加载图文问答示例
- 管理示例的选配和格式化
- 控制示例数量上限
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class FewShotManager:
    """
    Few-shot 示例管理器

    示例 JSON 格式：
    [
        {
            "image_path": "examples/001.jpg",
            "question": "图中有什么动物？",
            "answer": "猫"
        },
        ...
    ]
    """

    def __init__(self, examples_file: Optional[str] = None):
        self._examples: List[Dict[str, Any]] = []
        if examples_file:
            self.load(examples_file)

    def load(self, examples_file: str):
        """加载示例文件"""
        path = Path(examples_file)
        if not path.exists():
            # 尝试相对于 config.yaml 所在目录
            alt_path = Path(__file__).parent.parent.parent / examples_file
            if alt_path.exists():
                path = alt_path
            else:
                print(f"[FewShotManager] 警告: 示例文件不存在 {examples_file}")
                return

        with open(path, "r", encoding="utf-8") as f:
            self._examples = json.load(f)
        print(f"[FewShotManager] 已加载 {len(self._examples)} 个示例")

    @property
    def count(self) -> int:
        return len(self._examples)

    def get_examples(self, max_examples: int = 3) -> List[Dict[str, Any]]:
        """获取前 N 个示例"""
        return self._examples[:max_examples]

    def format_examples(
        self,
        max_examples: int = 3,
    ) -> str:
        """
        将示例格式化为可拼入 Prompt 的文本

        Returns:
            格式化文本，如：
            "示例 1:\n问题: 图中有什么动物？\n答案: 猫\n\n示例 2:\n..."
        """
        selected = self._examples[:max_examples]
        if not selected:
            return ""

        lines = []
        for i, ex in enumerate(selected, 1):
            lines.append(
                f"示例 {i}:\n"
                f"问题: {ex['question']}\n"
                f"答案: {ex['answer']}"
            )
        return "\n\n".join(lines)

    # 注意：Few-shot 中的图像需要在 engine.py 中单独处理
    # Qwen2.5-VL 支持多图输入，可以将示例图像拼接在提示中
    def get_example_images(self, max_examples: int = 3) -> List[str]:
        """获取示例中的图像路径列表"""
        return [
            ex.get("image_path", "")
            for ex in self._examples[:max_examples]
            if ex.get("image_path")
        ]
