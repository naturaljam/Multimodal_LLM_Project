"""
自建中文数据集评测模块
- 人工打分 1-5 分
- 支持批量评分记录与统计分析
- JSON 格式存储打分记录
"""

import json
from typing import Dict, List


# 评分标准
SCORING_RUBRIC = {
    1: "完全错误——回答与图像内容无关，或严重误解",
    2: "部分相关——回答涉及图像但核心信息错误",
    3: "基本正确——回答大意正确但缺少关键细节",
    4: "准确完整——回答正确且包含充分细节",
    5: "优秀——回答准确、简洁，且超出预期（如识别了细微文字）",
}


def score_answer(
    predicted: str,
    reference_answers: List[str],
    question: str,
    score: int,
    scorer: str = "human",
) -> dict:
    """
    记录单条中文问答的人工评分

    Args:
        predicted: 模型预测答案
        reference_answers: 参考答案列表
        question: 原始问题
        score: 人工打分 (1-5)
        scorer: 评分人标识

    Returns:
        评分记录 dict
    """
    if score not in SCORING_RUBRIC:
        raise ValueError(f"分数必须在 1-5 之间，收到: {score}")

    return {
        "question": question,
        "predicted": predicted,
        "reference_answers": reference_answers,
        "score": score,
        "scorer": scorer,
    }


def compute_scores_stats(
    records: List[dict],
) -> dict:
    """
    计算人工打分的统计信息

    Args:
        records: 评分记录列表

    Returns:
        {
            "avg_score": float,         # 平均分
            "median_score": float,      # 中位数
            "score_distribution": dict,  # {1: count, 2: count, ...}
            "total": int,               # 总样本数
            "score_4plus_ratio": float,  # 4分及以上占比
        }
    """
    if not records:
        return {
            "avg_score": 0.0,
            "median_score": 0.0,
            "score_distribution": {i: 0 for i in range(1, 6)},
            "total": 0,
            "score_4plus_ratio": 0.0,
        }

    scores = [r["score"] for r in records]

    avg_score = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n % 2 == 1:
        median_score = sorted_scores[n // 2]
    else:
        median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    distribution = {i: 0 for i in range(1, 6)}
    for s in scores:
        distribution[s] += 1

    score_4plus = sum(1 for s in scores if s >= 4)
    score_4plus_ratio = score_4plus / n

    return {
        "avg_score": round(avg_score, 2),
        "median_score": round(median_score, 2),
        "score_distribution": distribution,
        "total": n,
        "score_4plus_ratio": round(score_4plus_ratio, 4),
    }


def save_records(records: List[dict], path: str):
    """保存评分记录到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_records(path: str) -> List[dict]:
    """加载评分记录"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
