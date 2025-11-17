# agents/small_talk.py
import os
from utils.prompt_loader import load_prompt_from_hub

# -------------------------------------------------------------------
# Local small-talk agent (no LLM or external API)
# -------------------------------------------------------------------

def handle_small_talk(user_query: str) -> str:
    """
    Handles greetings and casual interactions.
    Uses a lightweight rule-based system with optional prompt guidance.
    """

    query = user_query.lower().strip()
    response = None

    # Load optional small-talk prompt (optional)
    try:
        system_prompt = load_prompt_from_hub("small_talk")
    except Exception:
        system_prompt = (
            "You are a friendly FinOps assistant. Respond casually to greetings, thanks, "
            "and general conversation, but do not perform analysis here."
        )

    # Simple keyword-based responses
    if any(word in query for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        response = "👋 Hello! How can I help you with your FinOps data today?"
    elif "thank" in query:
        response = "You're very welcome! 😊 Happy to help!"
    elif "how are you" in query:
        response = "I'm doing great, thanks for asking! How can I assist you with your cloud costs?"
    elif "bye" in query or "exit" in query:
        response = "Goodbye! 👋 Hope to see you again soon."
    else:
        # Fallback response
        response = "🙂 I’m here! You can ask me about trends, costs, or usage details anytime."

    return response


# -------------------------------------------------------------------
# Test script
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("💬 Small Talk Agent (Local Test)\n" + "-" * 40)
    test_inputs = [
        "Hello there!",
        "Thanks!",
        "How are you doing?",
        "Bye for now",
        "Can you help me?"
    ]
    for text in test_inputs:
        print(f"You: {text}")
        print(f"Bot: {handle_small_talk(text)}\n")
