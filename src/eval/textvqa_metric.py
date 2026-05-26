"""
TextVQA 评测指标
- 采用严格字符串匹配（Exact Match）作为主要指标
- 每条样本有 10 个人工标注答案
- 只要预测答案与任一标注答案完全匹配即算正确
"""

from typing import List


def _normalize(text: str) -> str:
    """文本标准化"""
    text = text.strip().lower()
    # 去除末尾句号
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def compute_textvqa_accuracy(
    predicted: str,
    ground_truth_answers: List[str],
) -> float:
    """
    计算单条 TextVQA 准确率（严格匹配）

    Args:
        predicted: 模型预测答案
        ground_truth_answers: 10 个人工标注答案列表

    Returns:
        1.0（匹配）或 0.0（不匹配）
    """
    if not ground_truth_answers:
        return 0.0

    pred_norm = _normalize(predicted)

    for ans in ground_truth_answers:
        if _normalize(ans) == pred_norm:
            return 1.0

    return 0.0


def compute_textvqa_dataset_accuracy(
    predictions: List[str],
    ground_truth_list: List[List[str]],
) -> dict:
    """
    计算整个 TextVQA 数据集的准确率

    Args:
        predictions: 模型预测答案列表
        ground_truth_list: 对应的人工标注答案列表

    Returns:
        {
            "accuracy": float,       # 严格匹配准确率
            "total": int,            # 总样本数
            "correct": int,          # 正确样本数
        }
    """
    total = len(predictions)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    correct = 0
    for pred, gt_answers in zip(predictions, ground_truth_list):
        if compute_textvqa_accuracy(pred, gt_answers) >= 1.0:
            correct += 1

    accuracy = correct / total

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
    }
