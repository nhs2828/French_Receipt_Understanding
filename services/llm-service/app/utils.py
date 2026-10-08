import contextvars
from concurrent.futures import ThreadPoolExecutor

async def run_with_context(loop, executor: ThreadPoolExecutor, func, *args):
    """Runs func(*args) inside the executor, preserving the current contextvars (request_id)"""
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(executor, lambda: ctx.run(func, *args))