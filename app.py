"""
Gradio Web UI（Kaggle 本地推理版）
==================================
- 在 Kaggle T4 上直接用 4-bit 量化模型推理
- 不依赖 HF Inference API
- 自动检测 Kaggle 环境，生成公网链接

启动:
    python app.py --share
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from PIL import Image

from src.utils.config import config
from src.preprocess.image_loader import load_image
from src.preprocess.resize_adapter import resize_to_vl
from src.prompt.builder import build_prompt
from src.prompt.templates import list_templates

# ============================================================
# 推理函数
# ============================================================

if config.is_kaggle or config.is_colab:
    # 云端：用本地 4-bit 模型
    from src.inference.engine import ask
    print("[App] 模式: 本地 GPU 推理")
else:
    # 本机无 GPU：stub
    from src.inference.stub import stub_ask as ask
    print("[App] 模式: Stub 假推理")


def respond(image, question, template_name):
    if image is None:
        return "请先上传一张图片。", ""
    if not question or not question.strip():
        return "请输入你的问题。", ""

    try:
        img = load_image(image) if not isinstance(image, Image.Image) else image
        img = resize_to_vl(img, config.resize_max_pixels)
        prompt_text = build_prompt(question.strip(), template_name=template_name)
        answer = ask(img, prompt_text)
        return answer, prompt_text
    except Exception as e:
        return f"❌ 错误: {e}", ""


def main():
    templates = list_templates()

    print(f"""
╔══════════════════════════════════════╗
║   VLM 智能图文问答助手 - Web UI    ║
╠══════════════════════════════════════╣
║  平台:  {config.platform:<28}║
║  模型:  Qwen2.5-VL 7B (4-bit)     ║
╚══════════════════════════════════════╝
""")

    with gr.Blocks(title="VLM 智能图文问答助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ VLM 智能图文问答助手
        基于 **Qwen2.5-VL 7B (4-bit 量化)** | Kaggle T4
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📷 上传图片")
                template_dropdown = gr.Dropdown(
                    choices=[t["name"] for t in templates],
                    value="general",
                    label="📝 Prompt 模板",
                )
                question_input = gr.Textbox(
                    label="❓ 你的问题",
                    placeholder="例如：图中有什么物体？文字内容是什么？",
                    lines=2,
                )
                submit_btn = gr.Button("🚀 提问", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label="💬 回答", lines=5, interactive=False
                )
                prompt_output = gr.Textbox(
                    label="🔍 实际 Prompt（调试）", lines=6, interactive=False
                )

        gr.Examples(
            examples=[
                ["图中有什么动物？", "general"],
                ["请识别图像中的所有文字", "textvqa"],
                ["描述这张图片的内容", "scene"],
            ],
            inputs=[question_input, template_dropdown],
            label="💡 快速示例（请先上传图片）",
        )

        submit_btn.click(
            fn=respond,
            inputs=[image_input, question_input, template_dropdown],
            outputs=[answer_output, prompt_output],
        )

    # Kaggle 环境自动 share
    share = config.is_kaggle or config.is_colab
    demo.launch(server_name="0.0.0.0", share=share)


if __name__ == "__main__":
    main()
