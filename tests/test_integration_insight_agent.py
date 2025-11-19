# tests/integration/test_integration_insight_agent.py

import pytest
from unittest.mock import MagicMock
from agents.insightAgent import generate_insights


def test_integration_generate_insights_with_real_csv_mock_llm(mocker, tmp_path):
    csv_file = tmp_path / "integration.csv"
    csv_file.write_text(
        "date,cost,resource\n"
        "2024-01-01,10,A\n"
        "2024-02-01,15,B\n"
    )

    mocker.patch("os.environ.get", return_value="dummy")
    mocker.patch("os.path.exists", return_value=True)

    # Allow real pandas REPL execution
    def fake_repl(code):
        return """{'summary': {'rows': 2, 'columns': ['date', 'cost', 'resource']},
                   'trend': [{'date': '2024-01', 'cost': 10},
                             {'date': '2024-02', 'cost': 15}]}"""

    mocker.patch("agents.insightAgent.python_repl.run", side_effect=fake_repl)

    mocker.patch("utils.prompt_loader.load_prompt_from_hub", return_value="prompt")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = "integration test insight"
    mocker.patch("agents.insightAgent.ChatGroq", return_value=fake_llm)

    output = generate_insights.run(str(csv_file))

    assert "integration test insight" in output
    assert "rows" in output
    assert "Python Analysis" in output
    assert "Trend" in output


def test_integration_python_only_no_llm(mocker, tmp_path):
    csv_file = tmp_path / "integration2.csv"
    csv_file.write_text("date,cost\n2024-01-01,20")

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.getenv", return_value=None)

    mocker.patch(
        "agents.insightAgent.python_repl.run",
        return_value="{'summary': {'rows': 1, 'columns': ['date', 'cost']}, 'trend': None}"
    )

    output = generate_insights.run(str(csv_file))

    assert "Python Analysis" in output
    assert "Trend" in output
    assert "rows" in output
