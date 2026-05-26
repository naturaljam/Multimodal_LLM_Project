"""
VQA-v2 评测指标
- 采用官方鲁棒准确率 (Robust Accuracy)
- 公式: Accuracy = min(count(预测答案, 10个人类答案) / 3, 1)
- 每条样本有 10 个人工标注答案
"""

from typing import List


def _normalize(text: str) -> str:
    """答案标准化：小写、去除首尾空格、去除末尾句号"""
    text = text.strip().lower()
    # 去除末尾的句号
    if text.endswith("."):
        text = text[:-1].strip()
    # 去除冠词 "a ", "an ", "the "（仅当答案是单个词时常见）
    # 保留更完整的处理
    return text


def compute_vqa_accuracy(
    predicted: str,
    ground_truth_answers: List[str],
) -> float:
    """
    计算单条 VQA-v2 鲁棒准确率

    Args:
        predicted: 模型预测答案
        ground_truth_answers: 10 个人工标注答案列表

    Returns:
        accuracy ∈ [0, 1]
    """
    if not ground_truth_answers:
        return 0.0

    pred_norm = _normalize(predicted)

    # 统计预测答案在人类答案中出现的次数
    match_count = 0
    for ans in ground_truth_answers:
        if _normalize(ans) == pred_norm:
            match_count += 1

    # 鲁棒准确率：至少 3 人给出相同答案才算满分
    accuracy = min(match_count / 3.0, 1.0)
    return accuracy


def compute_vqa_dataset_accuracy(
    predictions: List[str],
    ground_truth_list: List[List[str]],
) -> dict:
    """
    计算整个 VQA-v2 数据集的准确率

    Args:
        predictions: 模型预测答案列表
        ground_truth_list: 对应的人工标注答案列表（每条 10 个答案）

    Returns:
        {
            "accuracy": float,       # 整体准确率
            "total": int,            # 总样本数
            "correct_3of3": int,     # 至少 3 人一致的样本数
        }
    """
    total = len(predictions)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "correct_3of3": 0}

    scores = []
    for pred, gt_answers in zip(predictions, ground_truth_list):
        scores.append(compute_vqa_accuracy(pred, gt_answers))

    accuracy = sum(scores) / total
    correct_3of3 = sum(1 for s in scores if s >= 1.0)

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct_3of3": correct_3of3,
    }
