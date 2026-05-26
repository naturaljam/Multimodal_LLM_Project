"""
评测流水线主脚本
===================
串联所有模块：数据集加载 → 预处理 → Prompt构建 → 推理 → 后处理 → 指标计算 → 报告

使用方式:
    # 本地 stub 模式（无 GPU，用假推理验证 pipeline）
    python experiments/scripts/run_eval.py --stub

    # Kaggle/Colab 真实推理
    python experiments/scripts/run_eval.py

    # 指定模板
    python experiments/scripts/run_eval.py --template textvqa

    # 启用 OCR 预提取
    python experiments/scripts/run_eval.py --ocr

    # 只跑指定数据集
    python experiments/scripts/run_eval.py --datasets vqa_v2,textvqa
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

# 确保项目根目录在 sys.path 中（适配 Kaggle/Colab 解压后的路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm

from src.utils.config import config
from src.preprocess.image_loader import load_image
from src.preprocess.resize_adapter import resize_to_vl
from src.preprocess.ocr_extractor import extract_texts
from src.prompt.builder import build_prompt
from src.eval.vqa_metric import compute_vqa_dataset_accuracy
from src.eval.textvqa_metric import compute_textvqa_dataset_accuracy
from src.eval.chinese_metric import score_answer, compute_scores_stats, save_records
from src.eval.reporter import report


# ============================================================
# 数据集加载
# ============================================================

def _generate_stub_samples(dataset_type: str, subset_size: int) -> list:
    """stub 模式：生成合成样本用于验证 pipeline"""
    from PIL import Image

    samples = []
    for i in range(min(subset_size, 20)):  # stub 模式只生成少量样本
        # 创建简单的纯色测试图像
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 128, 128)][i % 5]
        img = Image.new("RGB", (448, 448), color)

        if dataset_type == "custom":
            samples.append({
                "image": img,
                "question": f"测试问题 {i+1}：这是什么颜色？",
                "reference_answers": [f"测试答案{i+1}"],
            })
        else:
            samples.append({
                "image": img,
                "question": f"Test question {i+1}: What color is this?",
                "answers": [{"answer": f"test answer {i+1}"} for _ in range(10)],
            })
    print(f"[Data] Stub 模式：生成 {len(samples)} 条合成样本")
    return samples


def load_vqa_v2(subset_size: int, use_stub: bool = False):
    """从 HuggingFace datasets 加载 VQA-v2 验证集子集"""
    if use_stub:
        return _generate_stub_samples("vqa_v2", subset_size)

    from datasets import load_dataset
    print(f"[Data] 加载 VQA-v2 子集 (n={subset_size})...")
    ds = load_dataset(config.vqa_v2_name, split="validation", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= subset_size:
            break
        samples.append(item)
    print(f"[Data] 实际加载 {len(samples)} 条")
    return samples


def load_textvqa(subset_size: int, use_stub: bool = False):
    """从 HuggingFace datasets 加载 TextVQA 验证集子集"""
    if use_stub:
        return _generate_stub_samples("textvqa", subset_size)

    from datasets import load_dataset
    print(f"[Data] 加载 TextVQA 子集 (n={subset_size})...")
    ds = load_dataset(config.textvqa_name, split="validation", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= subset_size:
            break
        samples.append(item)
    print(f"[Data] 实际加载 {len(samples)} 条")
    return samples


def load_custom_chinese(subset_size: int, use_stub: bool = False):
    """加载自建中文数据集"""
    if use_stub:
        return _generate_stub_samples("custom", subset_size)

    import json
    path = config.custom_chinese_path
    if not os.path.exists(path):
        print(f"[Data] 警告: 自建中文集不存在 ({path})，跳过")
        return []

    with open(path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    samples = all_samples[:min(len(all_samples), subset_size)]
    print(f"[Data] 加载自建中文集 {len(samples)} 条")
    return samples


# ============================================================
# 单条样本处理
# ============================================================

def process_sample(
    sample: dict,
    dataset_type: str,
    template_name: str,
    use_stub: bool,
    use_ocr: bool,
) -> Dict[str, Any]:
    """
    处理单条样本：预处理 → Prompt → 推理 → 返回结果

    Args:
        sample: 数据集中的单条样本
        dataset_type: "vqa_v2" | "textvqa" | "custom"
        template_name: Prompt 模板名
        use_stub: 是否使用假推理
        use_ocr: 是否启用 OCR 预提取

    Returns:
        {"predicted": str, "ground_truth": list, "question": str, "image": PIL.Image}
    """
    # ---- 加载图像 ----
    if "image" in sample:
        image = sample["image"]  # HuggingFace datasets 的 PIL.Image 字段
        if not isinstance(image, __import__("PIL").Image.Image):
            image = load_image(image)
    elif "image_path" in sample:
        image = load_image(sample["image_path"])
    else:
        raise KeyError(f"样本缺少图像字段: {list(sample.keys())}")

    # ---- 分辨率适配 ----
    image = resize_to_vl(image, config.resize_max_pixels)

    # ---- OCR 预提取 ----
    ocr_texts = None
    if use_ocr and config.ocr_enabled:
        ocr_texts = extract_texts(image, config.ocr_lang_list)

    # ---- 获取问题 ----
    question = sample.get("question", sample.get("input", ""))

    # ---- 构建 Prompt ----
    prompt = build_prompt(question, ocr_texts, template_name)

    # ---- 推理 ----
    if use_stub:
        from src.inference.stub import stub_ask
        answer = stub_ask(image, prompt, mode="hash")
    else:
        from src.inference.engine import ask
        answer = ask(image, prompt, config.model_max_new_tokens, config.model_temperature)

    # ---- 获取标准答案 ----
    # VQA-v2: answers 字段是 list[dict] → 提取 answer 文本
    # TextVQA: 也是类似结构
    gt_answers = []
    if "answers" in sample:
        raw_answers = sample["answers"]
        if isinstance(raw_answers, list):
            gt_answers = [
                a["answer"] if isinstance(a, dict) else str(a)
                for a in raw_answers
            ]
    elif "reference_answers" in sample:
        gt_answers = sample["reference_answers"]
    elif "answer" in sample:
        gt_answers = [sample["answer"]]

    return {
        "predicted": answer,
        "ground_truth": gt_answers,
        "question": question,
    }


# ============================================================
# 数据集评测
# ============================================================

def evaluate_dataset(
    samples: List[dict],
    dataset_type: str,
    template_name: str,
    use_stub: bool,
    use_ocr: bool,
) -> Dict[str, Any]:
    """
    对单个数据集进行全量评测

    Returns:
        评测指标字典
    """
    print(f"\n{'='*50}")
    print(f"[Eval] 评测数据集: {dataset_type} ({len(samples)} 条)")
    print(f"       模板: {template_name} | Stub: {use_stub} | OCR: {use_ocr}")
    print(f"{'='*50}")

    predictions = []
    ground_truths = []

    t_start = time.time()

    for sample in tqdm(samples, desc=f"[{dataset_type}]"):
        try:
            result = process_sample(sample, dataset_type, template_name, use_stub, use_ocr)
        except Exception as e:
            print(f"\n[ERROR] 样本处理失败: {e}")
            result = {"predicted": "", "ground_truth": [], "question": sample.get("question", "")}

        predictions.append(result["predicted"])
        ground_truths.append(result["ground_truth"])

    elapsed = time.time() - t_start
    avg_time = elapsed / len(samples) if samples else 0
    print(f"[Eval] 完成，耗时 {elapsed:.1f}s，平均 {avg_time:.2f}s/条")

    # ---- 根据数据集类型计算指标 ----
    valid_gts = [gt for gt in ground_truths if gt]  # 过滤空标准答案

    if dataset_type == "vqa_v2":
        metrics = compute_vqa_dataset_accuracy(predictions, ground_truths)
    elif dataset_type == "textvqa":
        metrics = compute_textvqa_dataset_accuracy(predictions, ground_truths)
    elif dataset_type == "custom":
        # 自建中文集：先输出预测供人工评分，也记录基础信息
        # 实际评分需人工操作，这里只做基础统计
        metrics = {
            "total": len(predictions),
            "note": "待人工评分",
            "predictions": predictions,
            "questions": [s.get("question", "") for s in samples],
        }
    else:
        metrics = {"total": len(predictions), "note": f"未知数据集类型: {dataset_type}"}

    # ---- 打印部分预测样例 ----
    print(f"\n[Sample Predictions] ({dataset_type}):")
    for i in range(min(5, len(predictions))):
        q = samples[i].get("question", "?")
        pred = predictions[i][:80]
        gt = ", ".join(str(g) for g in ground_truths[i][:3])
        print(f"  Q: {q[:60]}")
        print(f"  Pred: {pred}")
        print(f"  GT:   {gt}")
        print()

    return metrics


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VLM 智能图文问答助手 - 评测流水线")
    parser.add_argument("--stub", action="store_true",
                        help="使用假推理模式（本地无 GPU 调试）")
    parser.add_argument("--template", type=str, default=None,
                        help="Prompt 模板名 (general/textvqa/scene/chinese)")
    parser.add_argument("--ocr", action="store_true",
                        help="启用 OCR 预提取")
    parser.add_argument("--datasets", type=str, default=None,
                        help="指定数据集，逗号分隔 (vqa_v2,textvqa,custom)，默认全部")
    parser.add_argument("--no-load-model", action="store_true",
                        help="不加载模型（仅适用于 --stub 模式或后续扩展）")
    args = parser.parse_args()

    # ---- 确定运行参数 ----
    use_stub = args.stub or config.is_local
    template_name = args.template or config.prompt_default_template
    use_ocr = args.ocr

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       VLM 智能图文问答助手 - 评测流水线                 ║
╠══════════════════════════════════════════════════════════╣
║  平台:     {config.platform:<44}║
║  模型:     {config.model_name[:44]:<44}║
║  Stub:     {str(use_stub):<44}║
║  模板:     {template_name:<44}║
║  OCR:      {str(use_ocr):<44}║
╚══════════════════════════════════════════════════════════╝
""")

    # ---- 加载模型（非 stub 模式） ----
    if not use_stub and not args.no_load_model:
        from src.inference.model_loader import load_model_and_processor
        load_model_and_processor(
            model_name=config.model_name,
            use_quantization=(config.model_quantization == "nf4"),
        )

    # ---- 确定要评测的数据集 ----
    dataset_names = ["vqa_v2", "textvqa", "custom"]
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    # ---- 逐个数据集评测 ----
    all_results = {}
    dataset_loaders = {
        "vqa_v2": (load_vqa_v2, config.vqa_v2_subset_size),
        "textvqa": (load_textvqa, config.textvqa_subset_size),
        "custom": (load_custom_chinese, config.custom_chinese_subset_size),
    }

    for ds_name in dataset_names:
        if ds_name not in dataset_loaders:
            print(f"[Skip] 未知数据集: {ds_name}")
            continue

        loader_fn, size = dataset_loaders[ds_name]
        try:
            samples = loader_fn(size, use_stub=use_stub)
        except Exception as e:
            print(f"[Skip] 加载失败 ({ds_name}): {e}")
            continue

        if not samples:
            continue

        metrics = evaluate_dataset(samples, ds_name, template_name, use_stub, use_ocr)
        all_results[ds_name] = metrics

    # ---- 输出报告 ----
    if all_results:
        tag = f"{template_name}" + ("_ocr" if use_ocr else "") + ("_stub" if use_stub else "")
        report(
            results=all_results,
            output_dir=config.results_dir,
            formats=config.output_formats,
            tag=tag,
        )
    else:
        print("[Eval] 没有数据集被成功评测。")


if __name__ == "__main__":
    main()
