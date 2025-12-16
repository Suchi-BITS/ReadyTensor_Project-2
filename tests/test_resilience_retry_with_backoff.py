import pytest
import time

from utils.resilience import retry_with_backoff


def test_retry_succeeds_after_transient_failure():
    """
    Retry should succeed if the function recovers
    within the allowed retry limit.
    """
    calls = {"count": 0}

    def flaky_function():
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("temporary failure")
        return "success"

    result = retry_with_backoff(
        fn=flaky_function,
        exceptions=(TimeoutError,),
        max_retries=3,
        base_delay=0.01
    )

    assert result == "success"
    assert calls["count"] == 2


def test_retry_fails_after_max_retries():
    """
    Retry should raise the original exception
    after max retries are exhausted.
    """
    def always_fail():
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError):
        retry_with_backoff(
            fn=always_fail,
            exceptions=(RuntimeError,),
            max_retries=2,
            base_delay=0.01
        )
