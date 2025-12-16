import os
import pytest

from agents.insightAgent import generate_insights


def test_integration_python_only(monkeypatch, tmp_path):
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
        user_query="total cost",
        csv_path=str(csv)
    )

    assert isinstance(output, dict)
    assert output["error"] is False
    assert "python analysis" in output["summary"].lower()
