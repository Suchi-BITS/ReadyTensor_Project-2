# utils/resilience.py
import time
import random
import logging
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)

def retry_with_backoff(
    fn: Callable,
    exceptions: Tuple[Type[Exception], ...],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True,
):
    """ Generic retry wrapper with exponential backoff """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except exceptions as e:
            if attempt >= max_retries:
                logger.error(f"Retry failed after {attempt} attempts: {e}")
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay *= random.uniform(0.8, 1.2)
            logger.warning(
                f"Retry {attempt}/{max_retries} failed: {e}. Retrying in {delay:.2f}s"
            )
            time.sleep(delay)
