"""
PaddleOCR-VL 藏文 OCR Demo
HuggingFace Space / Gradio 可视化演示
"""

import gradio as gr
import paddle
from PIL import Image
from paddleformers import AutoModelForCausalLM, AutoTokenizer
from paddleformers.processing_utils import ProcessorMixin
from paddlenlp.transformers import AutoModelForCausalLM as PDModel
from paddlenlp.transformers import AutoTokenizer as PDTokenizer


# ============ 模型配置 ============
# 方案1: HuggingFace Hub 模型（推荐，部署到 Space 时用）
# MODEL_PATH = "你的用户名/PaddleOCR-VL-1.5-Tibetan-OCR"

# 方案2: 本地路径（本地测试用）
# MODEL_PATH = "/home/ubuntu2204/xf/output/"

MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5-base"  # 临时基座，替换成你的微调模型
LORA_PATH = None  # 如果用合并后的模型则设为 None

# ============ 全局变量 ============
model = None
tokenizer = None
processor = None


def load_model():
    """加载模型（首次推理时调用）"""
    global model, tokenizer, processor

    if model is not None:
        return

    print("Loading model...")
    tokenizer = PDTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = PDModel.from_pretrained(MODEL_PATH, trust_remote_code=True, dtype="float16")
    model.eval()
    print("Model loaded!")


def ocr_predict(image):
    """
    对上传图片进行 OCR 识别
    """
    if image is None:
        return "请上传一张图片"

    load_model()

    # 确保是 RGB
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    # 构造输入
    prompt = "<image>OCR"

    # 使用模型的 chat 方法推理
    try:
        from paddleformers.processors import PaddleOCRVLProcessor
        processor = PaddleOCRVLProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        messages = [{"role": "user", "content": prompt}]
        inputs = processor(messages=messages, images=[img], return_tensors="pd")

        with paddle.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
    except Exception as e:
        # 备用方案：用 paddleformers CLI 风格推理
        try:
            messages = [{"role": "user", "content": prompt}]
            response = model.chat(images=[img], messages=messages, tokenizer=tokenizer, max_new_tokens=512)
            result = response if isinstance(response, str) else str(response)
        except Exception as e2:
            result = f"推理出错: {str(e2)}\n备用方案出错: {str(e)}"

    return result.strip()


# ============ 示例图片 ============
# 部署时替换为你的实际示例图片路径或 URL
EXAMPLES = [
    # ["examples/tibetan_1.jpg"],
    # ["examples/tibetan_2.jpg"],
    # ["examples/tibetan_3.jpg"],
]


# ============ Gradio 界面 ============
with gr.Blocks(
    title="藏文 OCR 演示 - PaddleOCR-VL",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 藏文 OCR 演示系统
        
        基于 **PaddleOCR-VL-1.5** 大模型 + LoRA 微调的藏文光学字符识别系统。
        
        ### 使用方法
        1. 上传一张包含藏文的图片
        2. 点击「识别」按钮
        3. 查看识别结果
        
        ### 支持场景
        - 藏文古籍手稿
        - 藏文自然场景文字
        - 藏文 + 中文混合文本
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="上传图片",
                type="pil",
                height=400,
            )
            submit_btn = gr.Button("识别", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="识别结果",
                lines=15,
                placeholder="识别结果将显示在这里...",
                show_copy_button=True,
            )
            clear_btn = gr.Button("清空")

    if EXAMPLES:
        gr.Markdown("### 示例图片")
        gr.Examples(
            examples=EXAMPLES,
            inputs=input_image,
            outputs=output_text,
            fn=ocr_predict,
            label="点击示例图片快速体验",
        )

    gr.Markdown(
        """
        ---
        **技术栈**: PaddlePaddle + PaddleFormers + PaddleOCR-VL-1.5 + LoRA  
        **训练数据**: 藏文古籍 + 自然场景多语言数据  
        **基座模型**: [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-base)
        """
    )

    # 事件绑定
    submit_btn.click(fn=ocr_predict, inputs=input_image, outputs=output_text)
    clear_btn.click(fn=lambda: (None, ""), inputs=None, outputs=[input_image, output_text])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
