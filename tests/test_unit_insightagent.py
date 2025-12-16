import os
from unittest.mock import MagicMock
import pytest

from agents.insightAgent import generate_insights


def test_generate_insights_invalid_path():
    output = generate_insights(
        user_query="test",
        csv_path="missing.csv"
    )

    assert output["error"] is True
    assert "csv" in output["summary"].lower()


def test_generate_insights_python_parse_failure(monkeypatch, tmp_path):
    """
    Python parse failure should NOT crash the pipeline.
    System should gracefully fall back.
    """
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2")

    monkeypatch.setattr(os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        "agents.insightAgent.run_python",
        lambda _: "not a dict"
    )

    output = generate_insights(
        user_query="test",
        csv_path=str(csv)
    )

    assert output["error"] is False
    assert isinstance(output["summary"], str)
    assert len(output["summary"]) > 0



def test_generate_insights_python_only(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2")

    monkeypatch.setattr(os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        "agents.insightAgent.run_python",
        lambda _: "{'rows': 1, 'total_cost': 10}"
    )
    monkeypatch.setattr(
        "agents.insightAgent.openai_client",
        None
    )

    output = generate_insights(
        user_query="test",
        csv_path=str(csv)
    )

    assert output["error"] is False
    assert "python analysis" in output["summary"].lower()


def test_generate_insights_with_openai(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2")

    monkeypatch.setattr(os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        "agents.insightAgent.run_python",
        lambda _: "{'rows': 1, 'total_cost': 10}"
    )

    fake_resp = MagicMock()
    fake_resp.choices = [
        MagicMock(message=MagicMock(content="mocked openai insight"))
    ]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    monkeypatch.setattr(
        "agents.insightAgent.openai_client",
        fake_client
    )

    output = generate_insights(
        user_query="test",
        csv_path=str(csv)
    )

    assert output["error"] is False
    assert "mocked openai insight" in output["summary"].lower()
