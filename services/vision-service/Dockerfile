# services/vision-service/Dockerfile
# Build context: services/vision-service/ (no libs/ dependency here)

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    paddlepaddle-gpu==3.3.0 \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN pip install --no-cache-dir \
    paddleocr==3.7.0 "paddlex[ocr-core]==3.7.*"

RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# paddleocr needs numpy < 2.0
RUN pip uninstall -y numpy 
RUN pip install --no-cache-dir numpy==1.26

COPY app ./app

ENV MODEL_DIR=/app/models
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--bind", "0.0.0.0:8000"]