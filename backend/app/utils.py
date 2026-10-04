"""
Shared utility functions.
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def timed(func):
    """Decorator that logs the execution time of an async function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("%s executed in %.3f s", func.__name__, elapsed)
        return result
    return wrapper
