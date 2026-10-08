FROM vllm/vllm-openai:v0.31.0

WORKDIR /app
COPY serve.sh /app/serve.sh
RUN chmod +x /app/serve.sh

ENV MODEL_DIR=/app/models/qwen2.5-7b-awq \
    SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
    QUANTIZATION=awq \
    DTYPE=float16 \
    MAX_MODEL_LEN=4096 \
    MAX_NUM_SEQS=8 \
    GPU_MEM_UTIL=0.6 \
    PORT=8000 \
    HF_HUB_OFFLINE=1

EXPOSE 8000
ENTRYPOINT ["/app/serve.sh"]