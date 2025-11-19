# tests/unit/test_unit_insightAgent.py

import pytest
from unittest.mock import MagicMock

# correct import (snake_case, not insightAgent)
from agents.insightAgent import generate_insights, python_repl
from schema.state import AgentState


# ---------------------------
# UNIT TESTS: AgentState
# ---------------------------

def test_agent_state_update_valid():
    state = AgentState({
        "original_query": "test",
        "session_id": "abc",
        "memory_context": None,
        "intent": None,
        "category": None,
        "subagent": None,
        "confidence": 0.5,
        "csv_path": None,
        "dataframe_path": None,
        "chart_path": None,
        "insight_details": None,
        "tip": None,
        "response": None,
    })

    updated = state.update(intent="billing")
    assert state["intent"] == "billing"
    assert updated["intent"] == "billing"


def test_agent_state_reject_invalid_key():
    state = AgentState({
        "original_query": "q",
        "session_id": "abc",
        "memory_context": None,
        "intent": None,
        "category": None,
        "subagent": None,
        "confidence": 0.1,
        "csv_path": None,
        "dataframe_path": None,
        "chart_path": None,
        "insight_details": None,
        "tip": None,
        "response": None,
    })

    with pytest.raises(KeyError):
        state.update(unknown_key="x")


# ---------------------------
# UNIT TESTS: generate_insights()
# ---------------------------

def test_generate_insights_invalid_path(mocker):
    mocker.patch("os.path.exists", return_value=False)
    # MUST use .run() because generate_insights is a StructuredTool
    output = generate_insights.run("bad.csv")
    assert "CSV file not found" in output


def test_generate_insights_repl_parse_failure(mocker, tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("date,cost\n2024-01-01,10")

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("agents.insightAgent.python_repl.run", return_value="not a dict")

    output = generate_insights.run(str(csv_file))
    assert "Failed to parse" in output


def test_generate_insights_python_only_missing_llm_key(mocker, tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("date,cost\n2024-01-01,10")

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "agents.insightAgent.python_repl.run",
        return_value="{'summary': {'rows': 1, 'columns': ['date', 'cost']}, 'trend': None}"
    )

    mocker.patch("os.getenv", return_value=None)

    output = generate_insights.run(str(csv_file))

    assert "Python Analysis" in output
    assert "rows" in output


def test_generate_insights_with_mocked_llm(mocker, tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("date,cost\n2024-01-01,10")

    mocker.patch("os.path.exists", return_value=True)

    mocker.patch(
        "agents.insightAgent.python_repl.run",
        return_value="{'summary': {'rows': 1, 'columns': ['date', 'cost']}, 'trend': None}"
    )

    mocker.patch("os.getenv", return_value="dummy_key")

    mocker.patch("utils.prompt_loader.load_prompt_from_hub", return_value="fake prompt")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = "mocked LLM output"
    mocker.patch("agents.insightAgent.ChatGroq", return_value=fake_llm)

    output = generate_insights.run(str(csv_file))
    assert "mocked LLM output" in output
