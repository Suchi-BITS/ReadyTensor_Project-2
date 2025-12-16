import pytest

from agents.agentic_tools.text2sql import call_gpt_generate_sql


def test_text2sql_retry_on_llm_failure(mocker):
    """
    Text2SQL should retry LLM calls and recover
    from transient failures.
    """
    mock_client = mocker.patch(
        "agents.agentic_tools.text2sql.openai_client"
    )

    # Fail twice, then succeed
    mock_client.chat.completions.create.side_effect = [
        TimeoutError("rate limit"),
        TimeoutError("rate limit"),
        {
            "choices": [
                {"message": {"content": "SELECT 1"}}
            ]
        }
    ]

    sql = call_gpt_generate_sql(
        user_query="test query",
        schema=[],
        table_name="dummy_table"
    )

    assert "select" in sql.lower()
