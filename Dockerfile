FROM python:3.12-slim

# ─── Biến môi trường cho production ──────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4

# ffmpeg cần cho librosa decode mp3/m4a/...
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài torch bản CPU riêng (nhẹ hơn ~2GB so với bản mặc định kèm CUDA)
# torch>=2.6 bắt buộc cho transformers 5.x (dùng torch.accelerator)
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Cài phần dependencies còn lại (tách lớp để tận dụng Docker cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code + model vào image
COPY app.py .
COPY model/ ./model/

# Chạy bằng user không phải root (an toàn hơn cho production)
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Kiểm tra sống của container qua endpoint /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
