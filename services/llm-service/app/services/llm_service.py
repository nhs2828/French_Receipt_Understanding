from pathlib import Path
import time
from dataclasses import dataclass
import numpy as np
from PIL import Image
import asyncio
import httpx
import json
from openai import AsyncOpenAI
from app.core.config import get_settings
from .prompt import SYSTEM_PROMPT, RECEIPT_SCHEMA

ROOT_DIR = Path(__file__).parents[2]

@dataclass
class SegmentationResult:
    """
    Internal handoff object — consumed by preprocessing/ops.py next.
    """
    original_image: np.ndarray      # the original image, in RGB format, as a numpy array
    mask_polygons: list[np.ndarray]     # one polygon per detected receipt, in image coords
    confidences: list[float]
    original_size: tuple[int, int]  # (width, height), needed later for coordinate mapping


class LLMClient():
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30.0, max_concurrency: int = 8):
        self._base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._model = model
        self._model_loaded = None
        self._startup_timeout = 180
        self._load_time_ms: float | None = None

    def load(self) -> None:
        start = time.perf_counter()
        base_url = self._base_url.rstrip("/")   # e.g. http://vllm:8000/v1
        root_url = base_url.removesuffix("/v1") # e.g. http://vllm:8000
        deadline = time.monotonic() + self._startup_timeout
        last_error = None

        with httpx.Client(timeout=5.0) as http:
            # Wait until the server is up
            while True:
                try:
                    if http.get(f"{root_url}/health").status_code == 200:
                        break
                except httpx.HTTPError as e:
                    last_error = e
                if time.monotonic() > deadline:
                    raise RuntimeError(f"vLLM not ready after {self._startup_timeout}s: {last_error}")
                time.sleep(2)

            # Check the expected model is actually served
            resp = http.get(f"{base_url}/models")
            resp.raise_for_status()
            served = [m["id"] for m in resp.json()["data"]]
            if self._model not in served:
                raise RuntimeError(f"Model '{self._model}' not served by vLLM. Available: {served}")

            # Warm-up request
            warmup = http.post(
                f"{base_url}/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "[0] TOTAL 1.00"}],
                    "temperature": 0,
                    "max_tokens": 64,
                    "structured_outputs": {"json": RECEIPT_SCHEMA},
                },
                timeout=60.0,
            )
            warmup.raise_for_status()

        self._model_loaded = True
        self._load_time_ms = round((time.perf_counter() - start) * 1000, 2)

    async def extract(self, words: list[str], boxes: list[list[int]]) -> dict:
        lines = LLMClient.process_raw_ocr_output(words, boxes)
        if not lines:
            return {}
        ocr_text = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines))

        async with self._sem:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract the receipt fields from this OCR text:\n\n{ocr_text}"},
                ],
                extra_body={"structured_outputs": {"json": RECEIPT_SCHEMA}},
                temperature=0.0,
                max_tokens=256,
            )
        return json.loads(resp.choices[0].message.content)

    async def aclose(self):
        await self._client.close()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded is not None

    @property
    def load_time_ms(self) -> float | None:
        return self._load_time_ms

    @staticmethod
    def process_raw_ocr_output(words_raw, boxes_raw, y_thresh=0.5):
        """
        Merge OCR words into text lines using their bounding boxes.

        A word joins the current line if its vertical center is close to the
        line's average center (within y_thresh * the smaller box height).
        Returns a list of line strings in reading order (top to bottom,
        left to right).
        """
        items = []
        for word, box in zip(words_raw, boxes_raw):
            word = word.strip()
            if not word:
                continue
            x1, y1, x2, y2 = box
            items.append({
                "text": word,
                "x1": x1,
                "cy": (y1 + y2) / 2,
                "h": max(y2 - y1, 1),
            })

        # Top to bottom by vertical center
        items.sort(key=lambda i: i["cy"])

        lines = []
        for item in items:
            if lines:
                line = lines[-1]
                line_cy = sum(i["cy"] for i in line) / len(line)
                line_h = sum(i["h"] for i in line) / len(line)
                if abs(item["cy"] - line_cy) <= y_thresh * min(item["h"], line_h):
                    line.append(item)
                    continue
            lines.append([item])

        # Put the text from left to right inside each line
        return [
            " ".join(i["text"] for i in sorted(line, key=lambda i: i["x1"]))
            for line in lines
        ]