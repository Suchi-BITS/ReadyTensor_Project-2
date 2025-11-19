from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime


# -------- Utility Validation Helpers -------- #

def validate_string(value: Any, field_name: str) -> str:
    """Validate and sanitize string inputs."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    # Guardrail: Basic sanitization to avoid command injection / script injection
    forbidden = [";", "&&", "||", "`", "$(", "<script", "</script>"]
    if any(f in value.lower() for f in forbidden):
        raise ValueError(f"{field_name} contains forbidden characters for security reasons.")

    # Strip excessive whitespace
    return value.strip()


def validate_uuid_or_str(value: Any, field_name: str) -> str:
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


def validate_conversation_history(history: Any) -> List[Dict[str, Any]]:
    """Validate conversation history structure"""
    if history is None:
        return []
    
    if not isinstance(history, list):
        raise TypeError("conversation_history must be a list")
    
    validated_history = []
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise TypeError(f"History entry {idx} must be a dictionary")
        
        required_keys = ["role", "content", "timestamp"]
        missing_keys = [k for k in required_keys if k not in entry]
        if missing_keys:
            raise ValueError(f"History entry {idx} missing keys: {missing_keys}")
        
        if entry["role"] not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role in history entry {idx}: {entry['role']}")
        
        validated_history.append(entry)
    
    return validated_history


# -------- Memory Management Functions -------- #

def create_memory_entry(role: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a standardized memory entry
    
    Args:
        role: 'user', 'assistant', or 'system'
        content: The message content
        metadata: Optional metadata (intent, subagent, etc.)
    
    Returns:
        Dictionary with timestamp and structured data
    """
    if role not in ["user", "assistant", "system"]:
        raise ValueError(f"Invalid role: {role}")
    
    entry = {
        "role": role,
        "content": validate_string(content, "content"),
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    return entry


def format_memory_context(history: List[Dict[str, Any]], max_turns: int = 5) -> str:
    """
    Format conversation history into a readable context string
    
    Args:
        history: List of conversation entries
        max_turns: Maximum number of recent turns to include
    
    Returns:
        Formatted string for LLM context
    """
    if not history:
        return "No previous conversation context."
    
    # Take only recent turns
    recent_history = history[-max_turns * 2:]  # 2 messages per turn (user + assistant)
    
    context_lines = ["Previous Conversation:"]
    
    for entry in recent_history:
        role = entry["role"].capitalize()
        content = entry["content"]
        timestamp = entry.get("timestamp", "")
        
        # Truncate very long messages
        if len(content) > 500:
            content = content[:497] + "..."
        
        context_lines.append(f"{role}: {content}")
    
    return "\n".join(context_lines)


def extract_entities_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract remembered entities from conversation history
    
    Args:
        history: Conversation history
    
    Returns:
        Dictionary of extracted entities (filters, columns, etc.)
    """
    entities = {
        "mentioned_services": set(),
        "mentioned_regions": set(),
        "mentioned_dates": set(),
        "last_query_type": None,
        "last_filters": None
    }
    
    for entry in history:
        metadata = entry.get("metadata", {})
        
        # Extract from metadata
        if "extracted_entities" in metadata:
            extracted = metadata["extracted_entities"]
            if isinstance(extracted, dict):
                entities["last_filters"] = extracted
        
        if "intent" in metadata:
            entities["last_query_type"] = metadata["intent"]
    
    return entities


def summarize_conversation(history: List[Dict[str, Any]]) -> str:
    """
    Create a summary of the conversation for long-term memory
    
    Args:
        history: Full conversation history
    
    Returns:
        Summary string
    """
    if not history:
        return "No conversation to summarize."
    
    user_queries = [e["content"] for e in history if e["role"] == "user"]
    
    summary_parts = [
        f"Total exchanges: {len(history) // 2}",
        f"Topics discussed: {', '.join(user_queries[:3])}",
        f"Session duration: {len(history)} messages"
    ]
    
    return " | ".join(summary_parts)


# -------- Core State Functions -------- #

def init_state(
    original_query: str, 
    session_id: str = "default-session",
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Initialize the FinOps Agent state for LangGraph execution with validation,
    sanitization, security guardrails, and memory support.
    
    Args:
        original_query: Current user query
        session_id: Unique session identifier
        conversation_history: Previous conversation turns
    
    Returns:
        Initialized state dictionary
    """

    # Validate inputs
    original_query = validate_string(original_query, "original_query")
    session_id = validate_uuid_or_str(session_id, "session_id")
    conversation_history = validate_conversation_history(conversation_history)

    # Guardrail: Original query cannot be empty
    if not original_query:
        raise ValueError("original_query cannot be empty.")

    # Format memory context from history
    memory_context = format_memory_context(conversation_history, max_turns=5)
    
    # Extract entities from history for context
    remembered_entities = extract_entities_from_history(conversation_history)

    return {
        # Current query
        "original_query": original_query,
        "session_id": session_id,

        # Memory fields
        "conversation_history": conversation_history,
        "memory_context": memory_context,
        "remembered_entities": remembered_entities,
        "turn_number": len(conversation_history) // 2 + 1,

        # Default system fields
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
        
        # SQL fields
        "extracted_entities": None,
        "sql_query": None,
        "sql_result": None,
        
        # Error handling
        "error": False,
        "error_message": None
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
                value = validate_string(value, key)

        # Validation for conversation history
        if key == "conversation_history":
            value = validate_conversation_history(value)

        # Validation for paths (basic guardrails)
        if key.endswith("_path") and value is not None:
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string path.")
            if any(f in value for f in ["..", "//", "\\\\"]):
                raise ValueError(f"Unsafe path detected in '{key}'.")

        # Confidence: must be a float 0-1
        if key == "confidence":
            if not isinstance(value, (int, float)):
                raise TypeError("confidence must be a number.")
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError("confidence must be between 0.0 and 1.0.")

        state[key] = value

    return state


def add_to_conversation_history(
    state: Dict[str, Any],
    role: str,
    content: str,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Add a new entry to conversation history
    
    Args:
        state: Current state
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Optional metadata
    
    Returns:
        Updated state with new history entry
    """
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    new_entry = create_memory_entry(role, content, metadata)
    state["conversation_history"].append(new_entry)
    
    # Update memory context
    state["memory_context"] = format_memory_context(state["conversation_history"])
    
    return state


class AgentState(dict):
    """A safer state object with enforced update rules and memory support."""

    def update(self, *args, **kwargs):
        # Force updates to use update_state logic
        return update_state(self, **kwargs)
    
    def add_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add entry to conversation history"""
        return add_to_conversation_history(self, role, content, metadata)