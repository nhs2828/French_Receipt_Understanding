# services/kie-service/Dockerfile
# Build context MUST be repo root: docker build -f services/llm-service/Dockerfile .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# vision_client — installed as a proper editable package (has pyproject.toml now).
# No PYTHONPATH needed/wanted: PYTHONPATH=/libs would put the OUTER folder on
# sys.path too, which Python treats as an empty namespace package and it can
# shadow the real installed package, causing "cannot import name X (unknown location)".
COPY libs/vision_client /libs/vision_client
RUN pip install --no-cache-dir -e /libs/vision_client

# requirements.txt next — detectron2's build needs torch already installed
COPY services/llm-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/llm-service/app ./app

ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
EXPOSE 8000


#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--bind", "0.0.0.0:8000"]
