import time
import concurrent.futures
import pytest


def long_running_task():
    time.sleep(2)
    return "done"


def test_timeout_is_triggered():
    """
    A long-running task should raise TimeoutError
    when execution exceeds the timeout limit.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(long_running_task)

        with pytest.raises(concurrent.futures.TimeoutError):
            future.result(timeout=0.5)
