import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
import datetime
from typing import List, Optional, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import uuid 
from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger
import streamlit as st
logger = setup_execution_logger()
import pandas as pd
from agents.agentic_tools.entity_extraction import ExtractedQueryDetails

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
SCHEMA_PATH = os.path.join(BASE_DIR, "schema", "schema-context.json")

# os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
# llm = ChatAnthropic(model = "claude-sonnet-4-20250514")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model = "gpt-5-2025-08-07")
class SQLQueryDetails(BaseModel):
    sql_query: str = Field(..., description="The generated SQL query.")
    selected_columns: List[str] = Field(..., description="Columns selected.")
    selected_entities: List[str] = Field(...,description="Entities selected")
    confidence: float = Field(..., description="Reflection on how certain you are about the generated sql_query")
    reasoning: str = Field(..., description="Step by step reasoning of your sql_query generation")
    execution_status: str = Field(..., description="Execution status: success or error")
    execution_time:float = Field(...,description="Time taken to execute SQL query in database")
    execution_error: Optional[str] = Field(None, description="Error message if execution failed")
    row_count: int = Field(0, description="Number of rows returned")
    execution_result: Optional[List[dict]] = Field(default_factory=list, description="Sample of execution result")
    dataframe_path: Optional[str] = Field(None, description="Path to the CSV file with results")
    sql_valid: bool = Field(True, description="Whether the SQL is valid")
    sql_validation_error: Optional[str] = Field(None, description="Error from SQL validation")
    transpiled_sql: Optional[str] = Field(None, description="Transpiled SQL if requested")

def get_schema_for_columns(column_name):
    """
    Searches through the JSON data structure to find and return 
    the complete column information for a given column name.
    Loads data only once using function attribute for caching.
    
    Args:
        column_name: The name of the column to search for   
    
    Returns:
        dict: The complete column information if found, None otherwise
    """
    # Load data only once using function attribute
    if not hasattr(get_schema_for_columns, '_cached_data'):
        get_schema_for_columns._cached_data = json.loads(load_prompt_from_hub("schema-context"))
    
    data = get_schema_for_columns._cached_data
    
    def search_recursive(obj):
        # Handle case where obj is a list
        if isinstance(obj, list):
            for item in obj:
                result = search_recursive(item)
                if result is not None:
                    return result
        
        # Handle case where obj is a dictionary
        elif isinstance(obj, dict):
            # Check if this dict has a 'columns' key
            if 'columns' in obj and isinstance(obj['columns'], list):
                for column in obj['columns']:
                    if isinstance(column, dict) and column.get('name') == column_name:
                        return column
            
            # Recursively search through all values in the dictionary
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    result = search_recursive(value)
                    if result is not None:
                        return result
        
        return None
    
    return json.dumps(search_recursive(data),indent = 2)


def build_sql_prompt(extracted: ExtractedQueryDetails, few_shot_str: str):
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    columns = [col.name for col in extracted.columns_to_select]
    # columns_sql = [col.lower().replace(" ", "") for col in columns]
    schema_context = get_schema_for_columns(columns)
    # logger.info(f'SCHEMA CONTEXT : \n{schema_context}\n')
    entity_info = [
        f"{ent.name} (confidence: {ent.confidence:.2f})"
        for ent in extracted.entities
    ]
    column_info = [
        f"{col.name} (confidence: {col.confidence:.2f})"
        for col in extracted.columns_to_select
    ]

    logger.info(
        "[3] Extraction Results\n"
        "├─ Entities:\n"
        + "\n".join(f"│   - {e}" for e in entity_info) + "\n"
        "├─ Columns:\n"
        + "\n".join(f"│   - {c}" for c in column_info) + "\n"
        f"└─ Reasoning: {extracted.reasoning}"
    )
    
    prompt_template = load_prompt_from_hub(
        "text2sql",
        columns=columns,
        today_date=today_date,
        schema_context=schema_context,
        few_shot_str=few_shot_str
    )
    return prompt_template

if __name__ == "__main__":
    from agents.agentic_tools.entity_extraction import extract_entities_columns_filters, ExtractedQueryDetails,get_extraction_examples

    # Define a sample user query
    user_query = "What is the total production cost in April?"

    # Retrieve few-shot examples for the query
    few_shot_str = get_extraction_examples(user_query)

    # Extract entities and columns using the entity extraction tool
    extraction_response = extract_entities_columns_filters.invoke({
        "user_query": user_query,
        "few_shot_str": few_shot_str
    })

    # Check if the extraction was successful
    if isinstance(extraction_response, ExtractedQueryDetails):
        print("\n=== Entity and Column Extraction Results ===")
        print(f"Entities: {[entity.name for entity in extraction_response.entities]}")
        print(f"Columns: {[column.name for column in extraction_response.columns_to_select]}")
        print(f"Reasoning: {extraction_response.reasoning}")

        # Generate the SQL prompt using the extracted details
        sql_prompt = build_sql_prompt(extraction_response, few_shot_str=few_shot_str)

        print("\n=== Generated SQL Prompt ===")
        print(sql_prompt)
        # structured_llm = tex.with_structured_output(SQLGen)
        # messages = [{'role' : 'system','content' : prompt},{'role':'user','content' : formatted_query }]
        # sql_result = structured_llm.invoke(messages)