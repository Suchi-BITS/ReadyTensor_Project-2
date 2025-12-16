import pytest

from integrations.main import process_query


def test_process_query_handles_supervisor_failure(mocker):
    """
    process_query should gracefully handle supervisor crashes
    and return a user-friendly error response.
    """
    mocker.patch(
        "integrations.main.run_supervisor",
        side_effect=RuntimeError("supervisor crashed")
    )

    result = process_query(
        user_query="show cost",
        csv_path="data/data.csv",
        conversation_history=[],
        session_id="test-session"
    )

    assert result.get("error") is True
    assert "error" in result.get("response", "").lower()


def test_process_query_turn_limit_exceeded(mocker):
    """
    process_query should stop execution when turn limit is exceeded.
    """
    mocker.patch(
        "integrations.main.init_state",
        return_value={"turn_number": 25}
    )

    result = process_query(
        user_query="any query",
        csv_path="data/data.csv",
        conversation_history=[],
        session_id="test-session"
    )

    assert result.get("error") is True
    assert "limit" in result.get("response", "").lower()
