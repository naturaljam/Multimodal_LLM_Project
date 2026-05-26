"""
评测结果报告模块
- 汇总各数据集评测结果
- 输出格式：终端表格 / JSON / CSV
- 支持多次实验（不同 Prompt 模板）的对比报告
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_dir(path: str):
    """确保输出目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_table(results: Dict[str, Any]) -> str:
    """
    将结果格式化为终端可打印的表格

    Args:
        results: {
            "dataset_name": {"accuracy": 0.xxx, "total": 1000, ...},
            ...
        }

    Returns:
        表格字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("                    评测结果报告")
    lines.append("=" * 60)
    lines.append(f"{'数据集':<20} {'准确率':>10} {'样本数':>8} {'备注':>15}")
    lines.append("-" * 60)

    for name, metrics in results.items():
        acc = metrics.get("accuracy", "N/A")
        if isinstance(acc, float):
            acc_str = f"{acc:.2%}"
        else:
            acc_str = str(acc)
        total = metrics.get("total", "N/A")
        extra = ""
        if "correct" in metrics:
            extra = f"正确: {metrics['correct']}"
        elif "correct_3of3" in metrics:
            extra = f"≥3票: {metrics['correct_3of3']}"
        lines.append(f"{name:<20} {acc_str:>10} {str(total):>8} {extra:>15}")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_json(results: Dict[str, Any], path: str):
    """保存结果为 JSON"""
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Reporter] JSON 已保存: {path}")


def save_csv(results: Dict[str, Any], path: str):
    """保存结果为 CSV"""
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "accuracy", "total", "extra"])
        for name, metrics in results.items():
            acc = metrics.get("accuracy", "")
            total = metrics.get("total", "")
            extra = metrics.get("correct", metrics.get("correct_3of3", ""))
            writer.writerow([name, acc, total, extra])
    print(f"[Reporter] CSV 已保存: {path}")


def report(
    results: Dict[str, Any],
    output_dir: str = "experiments/results/",
    formats: Optional[List[str]] = None,
    tag: Optional[str] = None,
):
    """
    汇总输出评测报告

    Args:
        results: 各数据集评测结果字典
        output_dir: 输出目录
        formats: 输出格式列表 ["json", "csv"]
        tag: 实验标签（用于文件命名）
    """
    if formats is None:
        formats = ["json", "csv"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_" if tag else ""
    base_name = f"{prefix}{timestamp}"

    # 终端表格
    print("\n" + format_table(results))

    # JSON
    if "json" in formats:
        json_path = os.path.join(output_dir, f"{base_name}.json")
        save_json(results, json_path)

    # CSV
    if "csv" in formats:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        save_csv(results, csv_path)


def report_comparison(
    experiments: List[Dict[str, Any]],
    output_dir: str = "experiments/results/",
):
    """
    多实验对比报告

    Args:
        experiments: [
            {"tag": "baseline", "results": {...}},
            {"tag": "ocr_enhanced", "results": {...}},
        ]
        output_dir: 输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")

    report_data = {
        "timestamp": timestamp,
        "experiments": experiments,
    }

    _ensure_dir(output_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"                    对比报告")
    print(f"{'='*60}")
    print(f"{'实验':<25} {'VQA-v2':>10} {'TextVQA':>10} {'中文集':>10}")
    print(f"{'-'*60}")

    for exp in experiments:
        tag = exp.get("tag", "unknown")
        res = exp.get("results", {})
        vqa_acc = res.get("VQA-v2", {}).get("accuracy", "N/A")
        tvqa_acc = res.get("TextVQA", {}).get("accuracy", "N/A")
        ch_acc = res.get("自建中文集", {}).get("avg_score", "N/A")

        vqa_str = f"{vqa_acc:.2%}" if isinstance(vqa_acc, float) else str(vqa_acc)
        tvqa_str = f"{tvqa_acc:.2%}" if isinstance(tvqa_acc, float) else str(tvqa_acc)
        ch_str = f"{ch_acc:.2f}" if isinstance(ch_acc, float) else str(ch_acc)

        print(f"{tag:<25} {vqa_str:>10} {tvqa_str:>10} {ch_str:>10}")

    print(f"{'='*60}")
    print(f"[Reporter] 对比报告已保存: {json_path}")
