FROM python:3.10-slim

# 安装 PaddlePaddle GPU（CUDA 12.x）
RUN pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 安装其他依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

CMD ["python", "app.py"]
