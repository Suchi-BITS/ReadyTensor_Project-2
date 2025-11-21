import os, sys
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from vecx.vectorx import VectorX
from utils.prompt_loader import load_prompt_from_hub
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

INDEX_NAME = "query_intent_examples"

# -------------------- Pydantic Models --------------------
class EntityDetail(BaseModel):
    name: str = Field(description="Name of extracted FOCUS entity.")
    confidence: float = Field(ge=0.0, le=1.0)

class ColumnDetail(BaseModel):
    name: str = Field(description="Name of extracted column.")
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractedQueryDetails(BaseModel):
    entities: List[EntityDetail] = Field(default_factory=list)
    columns_to_select: List[ColumnDetail] = Field(default_factory=list)
    reasoning: str

class ExtractionError(BaseModel):
    error_message: str
    details: Optional[str] = None
    query: str


# -------------------- API Keys --------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
VECTORX_API_TOKEN = st.secrets.get("VECTORX_API_TOKEN") or os.getenv("VECTORX_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment or secrets.")

vx = VectorX(token=VECTORX_API_TOKEN)

# -------------------- Few-shot Retrieval --------------------
def get_extraction_examples(user_query: str, k: int = 3) -> str:
    vectordb = vx.get_index(INDEX_NAME)
    if not vectordb:
        return ""

    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=OPENAI_API_KEY
        )

        query_embedding = embedding_model.embed_query(user_query)

        results = vectordb.query(
            vector=query_embedding,
            top_k=k,
            ef=128,
            include_vectors=False
        )

        if not results:
            return ""

        examples = []
        for result in results:
            meta = result.get("meta", {})
            if not meta:
                continue

            example = {
                "query": meta.get("user_query", ""),
                "entities": meta.get("entity", "").split(", ") if meta.get("entity") else [],
                "columns": meta.get("column", "").split(", ") if meta.get("column") else [],
                "sql": meta.get("sql_query", "")
            }
            examples.append(example)

        formatted = ["Here are similar example queries with expected entities & columns:\n"]
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"- Query: {ex['query']}")
            formatted.append(f"- Entities: {', '.join(ex['entities']) or 'None'}")
            formatted.append(f"- Columns: {', '.join(ex['columns']) or 'None'}")
            formatted.append(f"- SQL: {ex['sql']}\n")

        return "\n".join(formatted)

    except Exception as e:
        print(f"Error retrieving few-shot examples: {e}")
        return ""

# -------------------- JSON Parser --------------------
def parse_json_response(text: str) -> dict:
    """Extract a JSON object even if wrapped inside markdown."""
    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise

# -------------------- Main Tool --------------------
@tool
def extract_entities_columns_filters(user_query: str, few_shot_str: str) -> ExtractedQueryDetails | ExtractionError:
    """
    Extract FinOps entities and columns from user query using OpenAI LLM.
    """

    print(f"[EntityExtraction] User query: {user_query}")

    try:
        focus_spec = json.loads(load_prompt_from_hub("schema_context"))

        prompt_template = load_prompt_from_hub(
            "entity_extraction",
            focus_spec_str=focus_spec,
            few_shot_str=few_shot_str or "No examples available."
        )

        full_prompt = f"""
{prompt_template}

USER QUERY:
{user_query}

You MUST respond with a strict JSON object:
{{
  "entities": [{{"name": "entity", "confidence": 0.95}} ],
  "columns_to_select": [{{"name": "column", "confidence": 0.90}} ],
  "reasoning": "Explain your reasoning here"
}}

Return ONLY valid JSON.
"""

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
        )

        print("[EntityExtraction] Calling OpenAI...")

        response = llm.invoke([
            SystemMessage(content="Extract FinOps entities & columns. Output ONLY valid JSON."),
            HumanMessage(content=full_prompt)
        ])

        response_text = response.content.strip()
        print("[EntityExtraction] Raw LLM output:", response_text[:120], "...")

        parsed = parse_json_response(response_text)

        result = ExtractedQueryDetails(**parsed)

        print(f"[EntityExtraction] Extracted {len(result.entities)} entities & {len(result.columns_to_select)} columns")

        return result

    except json.JSONDecodeError as e:
        err_msg = f"Failed to parse JSON: {str(e)}"
        return ExtractionError(error_message=err_msg, details=response_text[:400], query=user_query)

    except Exception as e:
        return ExtractionError(
            error_message="Unexpected error during entity extraction",
            details=str(e),
            query=user_query,
        )

# -------------------- Manual Test --------------------
if __name__ == "__main__":
    test_query = "what are the top cost drivers in compute?"
    few_shot = get_extraction_examples(test_query)

    response = extract_entities_columns_filters.invoke({
        "user_query": test_query,
        "few_shot_str": few_shot
    })

    print("\n=== RESULT ===")
    print(response)
