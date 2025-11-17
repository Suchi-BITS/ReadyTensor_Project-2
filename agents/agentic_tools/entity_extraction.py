import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from vecx.vectorx import VectorX
from utils.prompt_loader import load_prompt_from_hub
import streamlit as st
from groq import Groq

INDEX_NAME = "query_intent_examples"

# --- Pydantic Models for Structured Output ---
class EntityDetail(BaseModel):
    name: str = Field(description="The name of the FOCUS entities identified from the schema.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0) for this entity identification.")

class ColumnDetail(BaseModel):
    name: str = Field(description="The name of the FOCUS columns to be selected, as found in the schema.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0) for this column selection.")

class ExtractedQueryDetails(BaseModel):
    entities: List[EntityDetail] = Field(default_factory=list, description="A list of relevant FOCUS entities identified from the user query and schema.")
    columns_to_select: List[ColumnDetail] = Field(default_factory=list, description="A list of FOCUS columns that should be selected based on the user query.")
    reasoning: str

class ExtractionError(BaseModel):
    error_message: str
    details: Optional[str] = None
    query: str

class ExtractLog(BaseModel):
    entities: List[str]
    columns: List[str]

# --- API Keys ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
VECTORX_API_TOKEN = st.secrets.get("VECTORX_API_TOKEN") or os.getenv("VECTORX_API_TOKEN")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(" GROQ_API_KEY not found in secrets or environment")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize VectorX
vx = VectorX(token=VECTORX_API_TOKEN)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_extraction_examples(user_query: str, k: int = 3) -> str:
    """
    Retrieve similar query examples from the VectorX DB built from the JSON,
    including the expected SQL query in the formatted few-shot string.
    """
    vectordb = vx.get_index(INDEX_NAME)
    if not vectordb:
        print("Vector store not available. Cannot retrieve similar examples.")
        return ""
    
    try:
        # Generate embedding for the query
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=OPENAI_API_KEY
        )
        
        query_embedding = embedding_model.embed_query(user_query)
        
        # Query VectorX index
        results = vectordb.query(
            vector=query_embedding,
            top_k=k,
            ef=128,
            include_vectors=False
        )
        
        if not results:
            print("No similar examples found.")
            return ""
        
        examples = []
        for result in results:
            if "meta" not in result:
                continue
                
            meta = result["meta"]
            
            # Extract data from metadata
            example = {
                "query": meta.get("user_query", ""),
                "entities": meta.get("entity", ""),
                "columns": meta.get("column", ""),
                "sql": meta.get("sql_query", "")
            }
            
            # Convert comma-separated strings back to lists
            if isinstance(example["entities"], str) and example["entities"]:
                example["entities"] = [e.strip() for e in example["entities"].split(", ") if e.strip()]
            elif not example["entities"]:
                example["entities"] = []
            
            if isinstance(example["columns"], str) and example["columns"]:
                example["columns"] = [c.strip() for c in example["columns"].split(", ") if c.strip()]
            elif not example["columns"]:
                example["columns"] = []
            
            # Ensure entities/columns are lists
            if not isinstance(example["entities"], list):
                example["entities"] = [str(example["entities"])] if example["entities"] else []
            if not isinstance(example["columns"], list):
                example["columns"] = [str(example["columns"])] if example["columns"] else []
            
            examples.append(example)
        
        # Format the examples
        formatted = ["Here are some similar query examples with their expected entities, columns, and SQL:"]
        for i, example in enumerate(examples):
            query = example.get("query", "")
            entities = example.get("entities", [])
            columns = example.get("columns", [])
            sql = example.get("sql", "")
            
            formatted.append(f"Example {i+1}:")
            formatted.append(f"- Query: \"{query}\"")
            formatted.append(f"- Expected Entities: {', '.join(entities) if entities else 'None'}")
            formatted.append(f"- Expected Columns: {', '.join(columns) if columns else 'None'}")
            formatted.append(f"- Expected SQL: {sql}")
            formatted.append("")
        
        formatted.append("Use these examples as additional guidance for your analysis.")
        return "\n".join(formatted)
        
    except Exception as e:
        print(f"Error retrieving similar examples: {e}")
        return ""


def get_full_focus_spec_str() -> str:
    try:
        return json.loads(load_prompt_from_hub("schema_context"))
    except Exception as e:
        raise KeyError(f'Failed to load schema context from prompt hub: {str(e)}')


def parse_groq_json_response(text: str) -> dict:
    """
    Parse JSON from Groq response, handling markdown code blocks and other formatting.
    """
    text = text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Try to parse the JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # If parsing fails, try to extract JSON object from text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise e


@tool
def extract_entities_columns_filters(user_query: str, few_shot_str: str) -> ExtractedQueryDetails | ExtractionError:
    """
    Analyzes a structured FinOps delegation query to extract relevant entities and columns
    based on the FOCUS specification and examples using Groq LLM.

    Args:
        user_query: Structured delegation containing user query, memory context, objective,
                   suggested columns, filters, group by clauses, FinOps context, and data scope
        few_shot_str: Retrieved few-shot examples from vector database for guidance
        
    Returns:
        ExtractedQueryDetails: Pydantic model with entities, columns, and reasoning on success
        ExtractionError: Pydantic model with error details on failure
    """
    print(f'User Query in entity extraction: {user_query}')
    
    try:
        focus_spec_str = get_full_focus_spec_str()
        prompt_content = load_prompt_from_hub(
            "entity_extraction",
            focus_spec_str=focus_spec_str,
            few_shot_str=few_shot_str if few_shot_str else "No few shot available."
        )
        
        # Create the full prompt with JSON schema instruction
        full_prompt = f"""{prompt_content}

USER QUERY:
{user_query}

You MUST respond with a valid JSON object matching this exact schema:
{{
  "entities": [
    {{"name": "entity_name", "confidence": 0.95}}
  ],
  "columns_to_select": [
    {{"name": "column_name", "confidence": 0.90}}
  ],
  "reasoning": "Explain your reasoning here"
}}

Return ONLY the JSON object, no additional text or markdown formatting.
"""
        
        # Call Groq API
        print("[EntityExtraction] Calling Groq for entity extraction...")
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing FinOps queries and extracting entities and columns based on FOCUS specification. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        print(f"[EntityExtraction] Raw response: {response_text[:200]}...")
        
        # Parse the JSON response
        response_json = parse_groq_json_response(response_text)
        
        # Validate and convert to Pydantic model
        response_model = ExtractedQueryDetails(**response_json)
        
        print(f"[EntityExtraction]  Extracted {len(response_model.entities)} entities and {len(response_model.columns_to_select)} columns")
        return response_model
        
    except json.JSONDecodeError as e:
        error_details = f"Failed to parse JSON from Groq response: {str(e)}\nResponse: {response_text[:500]}"
        print(f"[EntityExtraction] JSON Parse Error: {error_details}")
        return ExtractionError(
            error_message="Failed to parse JSON response from LLM",
            details=error_details,
            query=user_query
        )
    except Exception as e:
        error_details = f"Unexpected error in entity extraction: {str(e)}"
        print(f"[EntityExtraction] Error: {error_details}")
        import traceback
        traceback.print_exc()
        return ExtractionError(
            error_message="Failed to extract structured query details from LLM response",
            details=error_details,
            query=user_query
        )


if __name__ == "__main__":
    # Test the entity extraction
    user_query = "what are the top 10 cost drivers?"
    
    print("Testing Entity Extraction with Groq\n" + "="*70)
    print(f"Query: {user_query}\n")
    
    # Get few-shot examples
    few_shot_str = get_extraction_examples(user_query)
    print(f"Few-shot examples retrieved: {len(few_shot_str) > 0}\n")
    
    # Extract entities
    response = extract_entities_columns_filters.invoke({
        "user_query": user_query,
        "few_shot_str": few_shot_str
    })
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    
    if isinstance(response, ExtractedQueryDetails):
        print(f"\n Extraction successful!\n")
        print(f"Entities: {[e.name for e in response.entities]}")
        print(f"Columns: {[c.name for c in response.columns_to_select]}")
        print(f"\nReasoning: {response.reasoning}")
    else:
        print(f"\n Extraction failed!")
        print(f"Error: {response.error_message}")
        if response.details:
            print(f"Details: {response.details}")