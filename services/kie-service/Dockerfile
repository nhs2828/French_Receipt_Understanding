# services/kie-service/Dockerfile
# Build context MUST be repo root: docker build -f services/kie-service/Dockerfile .

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
COPY services/kie-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# detectron2 from git, pinned commit — separate step, after torch exists.
# No CC/CXX/ARCHFLAGS needed here (that's macOS-only); gcc handles it on Linux.
RUN pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31'

COPY services/kie-service/app ./app

ENV MODEL_DIR=/app/models
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
