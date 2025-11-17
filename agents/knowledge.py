# agents/knowledge.py

import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
logger = setup_execution_logger()


# -------------------------------------------------------------------
# Helper: Load all .txt files from /data
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Knowledge Summary Tool
# -------------------------------------------------------------------
@tool
def get_knowledge_summary(query: str, memory_context: str = "") -> str:
    """
    Retrieve a hallucination-resistant FinOps knowledge summary using
    the combined knowledge from all .txt files in the data directory.
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")

        # Load system prompt
        prompt = load_prompt_from_hub("knowledge_agent")

        # Load ALL FinOps text files from data/ directory
        finops_knowledge = load_all_finops_docs()

        # LLM setup
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

        messages = [
            SystemMessage(
                content=(
                    f"{prompt}\n"
                    "Below is the FinOps knowledge base extracted from multiple documents:\n"
                    f"{finops_knowledge}\n"
                    "Use ONLY this information to answer queries."
                )
            ),
            HumanMessage(
                content=f"User query: {query}\nMemory context: {memory_context}"
            ),
        ]

        response = llm.invoke(messages)
        answer = response.content.strip()

        logger.info("get_knowledge_summary executed successfully.")
        return answer

    except Exception as e:
        logger.error(f"Error in get_knowledge_summary: {e}")
        return f"Failed to retrieve knowledge summary: {e}"


# -------------------------------------------------------------------
# FinOps Tip Tool
# -------------------------------------------------------------------
@tool
def get_finops_tip(topic: str = "") -> str:
    """
    Returns a short FinOps optimization or cost-saving tip.
    """
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

        base_prompt = (
            "You are a FinOps coach. Provide a short, practical cost-optimization tip. "
            "Keep it under three sentences and relevant to cloud cost management."
        )

        messages = [
            SystemMessage(content=base_prompt),
            HumanMessage(content=f"Topic: {topic if topic else 'General FinOps tip'}"),
        ]

        response = llm.invoke(messages)
        tip = response.content.strip()

        logger.info("get_finops_tip executed successfully.")
        return tip

    except Exception as e:
        logger.error(f"Error in get_finops_tip: {e}")
        return f"Failed to generate FinOps tip: {e}"


# -------------------------------------------------------------------
# Manual Test
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["GROQ_API_KEY"] = "<YOUR_GROQ_KEY>"  # local testing

    print("\n--- Testing Knowledge Summary ---")
    print(
        get_knowledge_summary.invoke(
            {
                "query": "Explain predictive FinOps in simple terms.",
                "memory_context": "No memory.",
            }
        )
    )

    print("\n--- Testing FinOps Tip ---")
    print(get_finops_tip.invoke({"topic": "Kubernetes cost optimization"}))
