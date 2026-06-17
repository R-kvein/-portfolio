FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
EXPOSE 8000

# 注意：vLLM 推理服务建议独立容器跑，这里只起 FastAPI 应用层
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
