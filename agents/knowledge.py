import os
from dotenv import load_dotenv
load_dotenv()

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


# ============================================================
#            LOAD ALL FINOPS KNOWLEDGE DOCUMENTS
# ============================================================
def load_all_finops_docs() -> str:
    """
    Reads and concatenates ALL .txt files in /data folder.
    Provides a unified FinOps knowledge base for the LLM.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    combined_text = ""

    if not os.path.exists(data_dir):
        return "No data folder found."

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            try:
                with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
                    combined_text += f"\n\n--- FILE: {file_name} ---\n\n"
                    combined_text += f.read()
            except:
                pass

    return combined_text or "No text files found in data folder."


# ============================================================
#               KNOWLEDGE SUMMARY AGENT
# ============================================================
@tool
def get_knowledge_summary(
    query: str,
    memory_context: str = "",
    remembered_entities: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Provides memory-aware FinOps knowledge responses.
    Uses stored knowledge files + conversation history.
    """

    try:
        # Load system prompt
        try:
            prompt = load_prompt_from_hub("knowledge_agent")
        except:
            prompt = (
                "You are a FinOps Knowledge Agent. Provide clear, correct domain insights. "
                "Include examples and best practices when helpful."
            )

        # Load knowledge base
        finops_knowledge = load_all_finops_docs()

        # Construct context
        context_parts = [f"Current query: {query}"]

        if memory_context:
            context_parts.append(f"\nConversation context:\n{memory_context}")

        if remembered_entities:
            context_parts.append("\nRemembered Entities:")
            for k, v in remembered_entities.items():
                context_parts.append(f"- {k}: {v}")

        if conversation_history:
            last_topics = [
                entry["content"][:50]
                for entry in conversation_history[-6:]
                if entry["role"] == "user"
            ]
            if last_topics:
                context_parts.append(f"\nRecent topics: {' | '.join(last_topics)}")

        final_context = "\n".join(context_parts)

        # Create OpenAI client
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )

        messages = [
            SystemMessage(
                content=(
                    f"{prompt}\n\n"
                    "You have access to an internal FinOps knowledge base below. "
                    "Use memory context and conversation continuity.\n\n"
                    "=== FINOPS KNOWLEDGE BASE ===\n"
                    f"{finops_knowledge}\n"
                )
            ),
            HumanMessage(content=final_context),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in get_knowledge_summary: {e}")
        return f"Failed to retrieve knowledge summary: {e}"


# ============================================================
#                FINOPS TIP AGENT
# ============================================================
@tool
def get_finops_tip(
    topic: str = "",
    memory_context: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Returns a short contextual FinOps optimization tip.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.4,
        )

        base_prompt = (
            "You are a FinOps coach. Provide a short, practical cloud cost optimization tip. "
            "Use memory/context when applicable. Keep it under 3 sentences."
        )

        tip_context = f"Topic: {topic or 'General FinOps'}"

        if memory_context:
            tip_context += f"\nConversation context:\n{memory_context[:300]}"

        if conversation_history:
            last_msgs = [
                entry["content"] for entry in conversation_history[-4:]
                if entry["role"] == "user"
            ]
            if last_msgs:
                tip_context += f"\nRecent discussion: {' -> '.join(last_msgs)}"

        messages = [
            SystemMessage(content=base_prompt),
            HumanMessage(content=tip_context),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in get_finops_tip: {e}")
        return f"Failed to generate FinOps tip: {e}"



