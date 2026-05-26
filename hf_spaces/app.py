"""
Gradio Web UI (HF Serverless Inference API)
============================================
- 通过 HuggingFace 免费 serverless API 调用 Qwen2.5-VL 7B
- 不需要 Inference Providers 权限
- 部署到 HF Spaces（免费）

启动方式:
    python app.py
"""

import os, sys, io, base64, json, time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from PIL import Image
from huggingface_hub import InferenceClient

from src.utils.config import config
from src.preprocess.image_loader import load_image
from src.preprocess.resize_adapter import resize_to_vl
from src.prompt.builder import build_prompt

# ============================================================
# 推理客户端（延迟初始化）
# ============================================================

_client = None

def _get_client():
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN", None)
        # 不指定 provider → 走免费 serverless API
        _client = InferenceClient(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            token=token,
        )
        print("[HF API] 已连接 Serverless Inference API")
    return _client


def _call_model(image: Image.Image, prompt: str) -> str:
    """通过 Serverless API 调用 Qwen2.5-VL"""
    client = _get_client()

    # PIL → base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                messages=messages,
                max_tokens=config.model_max_new_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "loading" in msg.lower() or "503" in msg:
                wait = (attempt + 1) * 20
                print(f"[HF API] 模型加载中，{wait}s 后重试...")
                time.sleep(wait)
            else:
                if attempt < 2:
                    time.sleep(5)
                else:
                    return f"❌ 错误: {msg[:300]}"

    return "❌ 模型启动超时，请稍后重试"


# ============================================================
# Gradio UI
# ============================================================

def respond(image, question, template_name):
    if image is None:
        return "请上传图片", ""
    if not question or not question.strip():
        return "请输入问题", ""

    try:
        img = load_image(image) if not isinstance(image, Image.Image) else image
        img = resize_to_vl(img, config.resize_max_pixels)
        prompt_text = build_prompt(question.strip(), template_name=template_name)
        answer = _call_model(img, prompt_text)
        return answer, prompt_text
    except Exception as e:
        return f"❌ {e}", ""


def main():
    from src.prompt.templates import list_templates
    templates = list_templates()

    with gr.Blocks(title="VLM 智能图文问答助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ VLM 智能图文问答助手
        基于 **Qwen2.5-VL 7B** | 上传图片，输入问题，获取答案
        > 首次推理需等待模型唤醒（约 1-2 分钟）
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📷 上传图片")
                template_dropdown = gr.Dropdown(
                    choices=[t["name"] for t in templates],
                    value="general", label="📝 Prompt 模板"
                )
                question_input = gr.Textbox(
                    label="❓ 问题", placeholder="例如：图中有什么？", lines=2
                )
                submit_btn = gr.Button("🚀 提问", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="💬 回答", lines=5, interactive=False)
                prompt_output = gr.Textbox(label="🔍 实际 Prompt", lines=6, interactive=False)

        submit_btn.click(
            fn=respond,
            inputs=[image_input, question_input, template_dropdown],
            outputs=[answer_output, prompt_output],
        )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
