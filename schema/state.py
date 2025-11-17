# schema/state.py

def init_state(original_query: str, session_id: str = "default-session") -> dict:
    """
    Initialize the FinOps Agent state for LangGraph execution.
    """
    return {
        "original_query": original_query,       # ✅ REQUIRED
        "session_id": session_id,
        "memory_context": "No previous context.",
        "intent": None,
        "category": None,
        "subagent": None,
        "confidence": 0.0,
        "csv_path": None,
        "dataframe_path": None,
        "chart_path": None,
        "insight_details": None,
        "tip": None,
        "response": None,
    }


def update_state(state: dict, **updates) -> dict:
    """
    Update existing state with new key-value pairs.
    """
    state.update(updates)
    return state


class AgentState(dict):
    """Simple subclass to keep compatibility."""
    pass
