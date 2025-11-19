import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


def load_all_finops_docs() -> str:
    """
    Reads and concatenates ALL .txt files inside the /data folder.
    Returns one large string containing all domain knowledge.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    combined_text = ""

    if not os.path.exists(data_dir):
        return "No data folder found."

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_text += f"\n\n--- FILE: {file_name} ---\n\n"
                    combined_text += f.read()
            except:
                pass

    return combined_text if combined_text else "No text files found in data folder."


@tool
def get_knowledge_summary(
    query: str,
    memory_context: str = "",
    remembered_entities: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Retrieve FinOps knowledge with full conversation memory awareness
    
    Args:
        query: Current user query
        memory_context: Formatted previous conversation
        remembered_entities: Previously extracted entities
        conversation_history: Full conversation history for deeper context
    
    Returns:
        Memory-aware knowledge summary
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")

        # Load system prompt
        prompt = load_prompt_from_hub("knowledge_agent")

        # Load FinOps knowledge base
        finops_knowledge = load_all_finops_docs()

        # Build memory-enriched context
        context_parts = [f"Current query: {query}"]
        
        if memory_context:
            context_parts.append(f"\nRecent conversation:\n{memory_context}")
        
        if remembered_entities:
            context_parts.append("\nRemembered context:")
            if remembered_entities.get("mentioned_services"):
                context_parts.append(f"- Services discussed: {', '.join(remembered_entities['mentioned_services'])}")
            if remembered_entities.get("last_query_type"):
                context_parts.append(f"- Previous query type: {remembered_entities['last_query_type']}")
        
        if conversation_history:
            # Extract topics from conversation
            topics = []
            for entry in conversation_history[-6:]:
                if entry["role"] == "user":
                    topics.append(entry["content"][:50])
            
            if topics:
                context_parts.append(f"\nDiscussion topics: {' | '.join(topics)}")

        enhanced_context = "\n".join(context_parts)

        # LLM setup with memory-aware system prompt
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

        messages = [
            SystemMessage(
                content=(
                    f"{prompt}\n"
                    "Below is the FinOps knowledge base:\n"
                    f"{finops_knowledge}\n\n"
                    "IMPORTANT: You have access to previous conversation context. "
                    "Reference earlier discussions when relevant to maintain conversational continuity. "
                    "If the user refers to something mentioned earlier (like 'that cost spike' or 'the previous analysis'), "
                    "use the conversation context to understand what they mean."
                )
            ),
            HumanMessage(content=enhanced_context),
        ]

        response = llm.invoke(messages)
        answer = response.content.strip()

        logger.info("get_knowledge_summary executed with memory context")
        return answer

    except Exception as e:
        logger.error(f"Error in get_knowledge_summary: {e}")
        return f"Failed to retrieve knowledge summary: {e}"


@tool
def get_finops_tip(
    topic: str = "",
    memory_context: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Returns a contextual FinOps tip based on conversation history
    
    Args:
        topic: Specific topic for the tip
        memory_context: Previous conversation context
        conversation_history: Full conversation history
    
    Returns:
        Memory-aware FinOps tip
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

        # Build context-aware prompt
        base_prompt = (
            "You are a FinOps coach with conversation memory. "
            "Provide a short, practical cost-optimization tip relevant to the current discussion. "
            "Keep it under three sentences."
        )
        
        # Enhance with conversation context
        tip_context = f"Topic: {topic if topic else 'General FinOps tip'}"
        
        if memory_context:
            tip_context += f"\n\nConversation context:\n{memory_context[:300]}"
        
        if conversation_history:
            recent_topics = []
            for entry in conversation_history[-4:]:
                if entry["role"] == "user":
                    recent_topics.append(entry["content"])
            
            if recent_topics:
                tip_context += f"\n\nRecent discussion: {' -> '.join(recent_topics)}"

        messages = [
            SystemMessage(content=base_prompt),
            HumanMessage(content=tip_context),
        ]

        response = llm.invoke(messages)
        tip = response.content.strip()

        logger.info("get_finops_tip executed with memory context")
        return tip

    except Exception as e:
        logger.error(f"Error in get_finops_tip: {e}")
        return f"Failed to generate FinOps tip: {e}"