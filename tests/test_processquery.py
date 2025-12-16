import pytest
from integrations.main import process_query


@pytest.fixture
def sample_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("service,cost\nEC2,100\n")
    return str(csv)


def test_process_query_basic_flow(sample_csv, monkeypatch):
    monkeypatch.setattr(
        "integrations.main.validate_environment",
        lambda: True
    )

    result = process_query(
        user_query="total cost",
        csv_path=sample_csv,
        conversation_history=[],
        session_id="test"
    )

    assert isinstance(result, dict)
    assert "response" in result


def test_process_query_turn_limit_exceeded(sample_csv, monkeypatch):
    monkeypatch.setattr(
        "integrations.main.init_state",
        lambda *_, **__: {"turn_number": 999}
    )
    monkeypatch.setattr(
        "integrations.main.validate_environment",
        lambda: True
    )

    result = process_query(
        user_query="any",
        csv_path=sample_csv,
        conversation_history=[],
        session_id="test"
    )

    assert result["error"] is True
    assert "limit" in result["response"].lower()
