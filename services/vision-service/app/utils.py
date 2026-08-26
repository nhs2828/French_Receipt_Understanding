import numpy as np
import cv2
from PIL import Image
import contextvars
from concurrent.futures import ThreadPoolExecutor

def ndarray_to_jpg(img_np, img_path):
    img = Image.fromarray(img_np)
    img.save(img_path)


def draw_polys(img_np, polys):
    img = np.ascontiguousarray(img_np)
    if len(polys[0]) == 8:
        polys = [np.array(p).reshape(4, 2) for p in polys]
    polys = [np.array(p, dtype=np.int32) for p in polys]
    cv2.polylines(img, polys, isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


async def run_with_context(loop, executor: ThreadPoolExecutor, func, *args):
    """Runs func(*args) inside the executor, preserving the current contextvars (request_id)"""
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(executor, lambda: ctx.run(func, *args))