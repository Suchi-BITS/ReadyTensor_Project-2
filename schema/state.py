# schema/state.py

from typing import Any, Dict
import uuid


# -------- Utility Validation Helpers -------- #

def _validate_string(value: Any, field_name: str) -> str:
    """Validate and sanitize string inputs."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    # Guardrail: Basic sanitization to avoid command injection / script injection
    forbidden = [";", "&&", "||", "`", "$(", "<script", "</script>"]
    if any(f in value.lower() for f in forbidden):
        raise ValueError(f"{field_name} contains forbidden characters for security reasons.")

    # Strip excessive whitespace
    return value.strip()


def _validate_uuid_or_str(value: Any, field_name: str) -> str:
    """Ensure session_id is a valid string or UUID-like."""
    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, str):
        value = value.strip()
        try:
            uuid.UUID(value)
            return value
        except Exception:
            # It's ok if it's not a strict UUID, allow simple session names
            return value

    raise TypeError(f"{field_name} must be a UUID or string.")


# -------- Core State Functions -------- #

def init_state(original_query: str, session_id: str = "default-session") -> Dict[str, Any]:
    """
    Initialize the FinOps Agent state for LangGraph execution with validation,
    sanitization, and security guardrails.
    """

    # Validate inputs
    original_query = _validate_string(original_query, "original_query")
    session_id = _validate_uuid_or_str(session_id, "session_id")

    # Guardrail: Original query cannot be empty
    if not original_query:
        raise ValueError("original_query cannot be empty.")

    return {
        "original_query": original_query,
        "session_id": session_id,

        # Default system fields
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


def update_state(state: Dict[str, Any], **updates) -> Dict[str, Any]:
    """
    Safely update the agent state, enforcing structure, validation, and guarding against
    injection or unexpected keys.
    """
    if not isinstance(state, dict):
        raise TypeError("State must be a dictionary.")

    allowed_keys = set(state.keys())

    for key, value in updates.items():

        # Prevent adding unknown keys (guardrail)
        if key not in allowed_keys:
            raise KeyError(f"Invalid state key: '{key}' is not allowed.")

        # Validation for known string fields
        if key in ["intent", "category", "subagent", "memory_context", "tip"]:
            if value is not None:
                value = _validate_string(value, key)

        # Validation for paths (basic guardrails)
        if key.endswith("_path") and value is not None:
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string path.")
            if any(f in value for f in ["..", "//", "\\"]):
                raise ValueError(f"Unsafe path detected in '{key}'.")

        # Confidence: must be a float 0–1
        if key == "confidence":
            if not isinstance(value, (int, float)):
                raise TypeError("confidence must be a number.")
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError("confidence must be between 0.0 and 1.0.")

        state[key] = value

    return state


class AgentState(dict):
    """A safer state object with enforced update rules."""

    def update(self, *args, **kwargs):
        # Force updates to use update_state logic
        return update_state(self, **kwargs)
