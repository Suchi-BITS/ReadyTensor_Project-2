# agents/supervisor.py
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from utils.logger_setup import setup_execution_logger
import pandas as pd
import sqlite3
from langchain_openai import ChatOpenAI
import streamlit as st
import os
import re

from agents.intent_router import classify_intent
from agents.small_talk import handle_small_talk
from agents.data_fetcher import fetch_data
from agents.insightAgent import generate_insights
from agents.visualizerAgent import visualize_data
from agents.knowledge import get_knowledge_summary

from agents.agentic_tools.entity_extraction import (
    extract_entities_columns_filters, 
    get_extraction_examples,
    ExtractionError
)

logger = setup_execution_logger()

# Security Constants
MAX_SQL_LENGTH = 2000
MAX_RESULT_ROWS = 10000
MAX_DATAFRAME_SIZE_MB = 500
BLOCKED_SQL_PATTERNS = [
    r'(?i)\bDROP\s+TABLE\b',
    r'(?i)\bDROP\s+DATABASE\b',
    r'(?i)\bDELETE\s+FROM\b',
    r'(?i)\bTRUNCATE\b',
    r'(?i)\bALTER\s+TABLE\b',
    r'(?i)\bCREATE\s+TABLE\b',
    r'(?i)\bINSERT\s+INTO\b',
    r'(?i)\bUPDATE\s+SET\b',
    r'(?i)\bEXEC\b',
    r'(?i)\bEXECUTE\b',
    r'(?i);\s*DROP\b',
    r'(?i);\s*DELETE\b',
    r'--',  # SQL comments
    r'/\*',  # Block comments
]

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class AgentState(TypedDict, total=False):
    """Shared state dictionary that persists across all LangGraph nodes."""
    original_query: str
    session_id: str
    memory_context: str
    intent: str
    category: str
    subagent: str
    confidence: float
    csv_path: str
    dataframe_path: str
    chart_path: str
    insight_details: Any
    tip: str
    response: str
    extracted_entities: Any
    sql_query: str
    sql_result: Any
    error: bool
    error_message: str

def validate_sql_query(sql_query: str) -> str:
    """
    Validate SQL query for security risks
    
    Args:
        sql_query: Generated SQL query string
        
    Returns:
        Validated SQL query
        
    Raises:
        SecurityError: If SQL contains dangerous patterns
        ValidationError: If SQL is invalid
    """
    if not sql_query or not isinstance(sql_query, str):
        raise ValidationError("SQL query must be a non-empty string")
    
    sql_query = sql_query.strip()
    
    # Check length
    if len(sql_query) > MAX_SQL_LENGTH:
        raise ValidationError(f"SQL query exceeds maximum length of {MAX_SQL_LENGTH} characters")
    
    # Check for dangerous SQL patterns
    for pattern in BLOCKED_SQL_PATTERNS:
        if re.search(pattern, sql_query):
            logger.warning(f"Blocked dangerous SQL pattern: {pattern}")
            raise SecurityError("SQL query contains potentially dangerous operations and has been blocked")
    
    # Ensure query starts with SELECT
    if not sql_query.upper().strip().startswith('SELECT'):
        raise SecurityError("Only SELECT queries are allowed")
    
    # Check for multiple statements
    if sql_query.count(';') > 1:
        raise SecurityError("Multiple SQL statements are not allowed")
    
    logger.info("SQL query validation passed")
    return sql_query

def validate_dataframe(df: pd.DataFrame, operation: str = "unknown") -> pd.DataFrame:
    """
    Validate pandas DataFrame for size and content
    
    Args:
        df: DataFrame to validate
        operation: Name of operation for logging
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError: If DataFrame fails validation
    """
    if df is None:
        raise ValidationError(f"DataFrame is None in {operation}")
    
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)} in {operation}")
    
    if df.empty:
        logger.warning(f"DataFrame is empty in {operation}")
        return df
    
    # Check DataFrame size
    df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if df_size_mb > MAX_DATAFRAME_SIZE_MB:
        raise ValidationError(
            f"DataFrame size ({df_size_mb:.2f}MB) exceeds maximum of {MAX_DATAFRAME_SIZE_MB}MB in {operation}"
        )
    
    # Check row count
    if len(df) > MAX_RESULT_ROWS:
        logger.warning(f"DataFrame has {len(df)} rows, truncating to {MAX_RESULT_ROWS}")
        df = df.head(MAX_RESULT_ROWS)
    
    logger.info(f"DataFrame validation passed for {operation}: {len(df)} rows, {df_size_mb:.2f}MB")
    return df

def validate_state(state: AgentState, required_fields: list = None) -> AgentState:
    """
    Validate agent state structure and required fields
    
    Args:
        state: Current agent state
        required_fields: List of required field names
        
    Returns:
        Validated state
        
    Raises:
        ValidationError: If state is invalid
    """
    if not isinstance(state, dict):
        raise ValidationError(f"State must be a dictionary, got {type(state)}")
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in state]
        if missing_fields:
            raise ValidationError(f"Missing required state fields: {', '.join(missing_fields)}")
    
    return state

def safe_execute_sql(sql_query: str, conn: sqlite3.Connection, timeout: int = 30) -> pd.DataFrame:
    """
    Safely execute SQL query with timeout and error handling
    
    Args:
        sql_query: SQL query to execute
        conn: SQLite connection
        timeout: Query timeout in seconds
        
    Returns:
        Result DataFrame
        
    Raises:
        ValidationError: If query execution fails
    """
    try:
        # Set timeout
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        
        # Execute query
        result_df = pd.read_sql_query(sql_query, conn)
        
        # Validate result
        result_df = validate_dataframe(result_df, "SQL execution")
        
        return result_df
        
    except pd.io.sql.DatabaseError as e:
        logger.error(f"Database error executing SQL: {e}")
        raise ValidationError(f"Database error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise ValidationError(f"Query execution failed: {str(e)}")

def classify_node(state: AgentState) -> AgentState:
    """Classify user intent with error handling"""
    try:
        validate_state(state, required_fields=['original_query'])
        
        intent_info = classify_intent(state["original_query"])
        
        if not isinstance(intent_info, dict):
            raise ValidationError("Intent classification returned invalid format")
        
        logger.info(f"Classifier intent={intent_info.get('intent')}, subagent={intent_info.get('subagent')}")
        
        result = {
            **state,
            "intent": intent_info.get("intent", "unknown"),
            "subagent": intent_info.get("subagent", "unknown"),
            "confidence": intent_info.get("confidence", 0.0)
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in classify_node: {e}")
        return {
            **state,
            "intent": "error",
            "subagent": "none",
            "error": True,
            "error_message": f"Classification error: {str(e)}"
        }

def small_talk_node(state: AgentState) -> AgentState:
    """Handle small talk with error handling"""
    try:
        validate_state(state, required_fields=['original_query'])
        
        response = handle_small_talk(state["original_query"])
        
        if not response or not isinstance(response, str):
            response = "I understand you want to chat. How can I help you with your FinOps data?"
        
        result = {**state, "response": response}
        return result
        
    except Exception as e:
        logger.error(f"Error in small_talk_node: {e}")
        return {
            **state,
            "response": "I apologize, but I encountered an error. How can I help you with your cloud spending data?",
            "error": True
        }

def data_fetcher_node(state: AgentState) -> AgentState:
    """
    DataFetcher node with comprehensive validation and security
    """
    conn = None
    try:
        validate_state(state, required_fields=['original_query', 'csv_path'])
        
        query = state["original_query"]
        csv_path = state["csv_path"]
        
        logger.info(f"DataFetcher Processing query: {query}")
        
        # Clear previous SQL state
        state.pop("sql_query", None)
        state.pop("sql_result", None)
        state.pop("error", None)
        state.pop("error_message", None)

        # STEP 1: Entity Extraction with error handling
        logger.info("DataFetcher Step 1: Extracting entities")
        extracted_entities = None
        extracted_columns = None
        
        try:
            few_shot_str = get_extraction_examples(query, k=3)
            extraction_result = extract_entities_columns_filters.invoke({
                "user_query": query,
                "few_shot_str": few_shot_str
            })
            
            if not isinstance(extraction_result, ExtractionError):
                extracted_entities = [e.name for e in extraction_result.entities]
                extracted_columns = [c.name for c in extraction_result.columns_to_select]
                
                logger.info(f"Extracted entities: {extracted_entities}")
                logger.info(f"Extracted columns: {extracted_columns}")
                
                state["extracted_entities"] = {
                    "entities": extracted_entities,
                    "columns": extracted_columns,
                    "reasoning": extraction_result.reasoning
                }
            else:
                logger.warning(f"Entity extraction failed: {extraction_result.error_message}")
                
        except Exception as e:
            logger.warning(f"Entity extraction error: {e}. Continuing without it.")

        # STEP 2: Load CSV with validation
        logger.info("DataFetcher Step 2: Loading CSV into SQLite")
        
        if not os.path.exists(csv_path):
            raise ValidationError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            df = validate_dataframe(df, "CSV loading")
        except Exception as e:
            raise ValidationError(f"Failed to load CSV: {str(e)}")
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        df.to_sql("finops_data", conn, index=False, if_exists="replace")
        
        logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

        # Build schema description
        schema_desc = f"""
Table: finops_data
Columns: {', '.join(df.columns.tolist())}
Row count: {len(df):,}
Column types:
{df.dtypes.to_string()}
"""

        # STEP 3: Generate SQL with Groq
        logger.info("DataFetcher Step 3: Generating SQL")
        
        entity_context = ""
        if extracted_entities or extracted_columns:
            entity_context = f"""
EXTRACTED ENTITIES AND COLUMNS:
- Entities identified: {', '.join(extracted_entities) if extracted_entities else 'None'}
- Columns to select: {', '.join(extracted_columns) if extracted_columns else 'None'}

Use these extracted entities and columns to guide your SQL generation.
"""

        sql_prompt = f"""
You are a SQL expert working with SQLite.
Generate a valid SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_desc}
{entity_context}
USER QUERY: {query}

Rules:
1. Table name is 'finops_data'
2. Use SQLite syntax only
3. Return ONLY the SQL query with no explanations or markdown
4. Use SUM for cost aggregations
5. Use the extracted columns when available
6. Use GROUP BY and ORDER BY as needed
7. Do NOT use DROP, DELETE, UPDATE, INSERT, or ALTER
8. Only SELECT queries are allowed
"""

        # Get API key with validation
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValidationError("GROQ_API_KEY not found in environment or Streamlit secrets")

        # Import here to avoid circular dependencies
        from groq import Groq
        
        # Create Groq client
        client = Groq(api_key=api_key)
        
        # Call Groq API with error handling
        try:
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator for FinOps data."},
                    {"role": "user", "content": sql_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            
            sql_query = chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise ValidationError(f"Failed to generate SQL query: {str(e)}")

        logger.info(f"Raw SQL from Groq: {sql_query}")

        # Cleanup SQL output
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        # Validate SQL query for security
        try:
            sql_query = validate_sql_query(sql_query)
        except (SecurityError, ValidationError) as e:
            logger.error(f"SQL validation failed: {e}")
            raise

        logger.info(f"Validated SQL: {sql_query}")

        # STEP 4: Execute SQL safely
        logger.info("DataFetcher Step 4: Executing SQL")
        
        try:
            result_df = safe_execute_sql(sql_query, conn, timeout=30)
        except Exception as e:
            raise ValidationError(f"SQL execution failed: {str(e)}")

        # Format results
        if result_df.empty:
            response_text = "Query executed successfully but returned no results."
            result_dict = None
        else:
            result_dict = result_df.to_dict("records")
            
            # Limit result display for large results
            display_limit = 100
            display_results = result_dict[:display_limit]
            
            entity_info = ""
            if extracted_entities or extracted_columns:
                entity_info = f"""
Entity Extraction Results:
- Entities: {', '.join(extracted_entities) if extracted_entities else 'None'}
- Columns: {', '.join(extracted_columns) if extracted_columns else 'None'}

"""
            
            result_preview = str(display_results)[:1000]  # Limit text size
            
            response_text = f"""
Query Results

{entity_info}SQL Query Executed:
{sql_query}

Results Preview (showing first {min(len(result_dict), display_limit)} rows):
{result_preview}

Total Row Count: {len(result_df)}
"""

        logger.info("DataFetcher Query executed successfully")
        
        return {
            **state,
            "sql_query": sql_query,
            "sql_result": result_dict,
            "response": response_text,
            "error": False
        }

    except ValidationError as e:
        logger.error(f"Validation error in data_fetcher_node: {e}")
        return {
            **state,
            "response": f"Validation Error: {str(e)}",
            "error": True,
            "error_message": str(e)
        }
    
    except SecurityError as e:
        logger.error(f"Security error in data_fetcher_node: {e}")
        return {
            **state,
            "response": f"Security Error: {str(e)}",
            "error": True,
            "error_message": str(e)
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in data_fetcher_node: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "response": f"Error processing query: An unexpected error occurred",
            "error": True,
            "error_message": str(e)
        }
    
    finally:
        if conn is not None:
            try:
                conn.close()
                logger.info("Database connection closed")
            except:
                pass

def insights_node(state: AgentState) -> AgentState:
    """Generate insights with error handling"""
    try:
        validate_state(state, required_fields=['csv_path'])
        
        insights = generate_insights(state["csv_path"])
        
        if not insights:
            raise ValidationError("Insights generation returned no results")
        
        result = {
            **state,
            "insight_details": insights.details,
            "response": insights.summary,
            "dataframe_path": insights.dataframe_path,
            "error": False
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in insights_node: {e}")
        return {
            **state,
            "response": "Unable to generate insights. Please try a different query.",
            "error": True,
            "error_message": str(e)
        }

def visualize_node(state: AgentState) -> AgentState:
    """Create visualization with error handling"""
    try:
        if not state.get("dataframe_path"):
            return {
                **state,
                "response": state.get("response", "No data available for visualization.")
            }
        
        # Validate dataframe path
        df_path = state["dataframe_path"]
        if not os.path.exists(df_path):
            logger.warning(f"Dataframe path does not exist: {df_path}")
            return {**state}
        
        chart = visualize_data(df_path)
        
        if not chart or not chart.chart_path:
            logger.warning("Visualization failed to generate chart")
            return {**state}
        
        current_response = state.get("response", "")
        response_text = f"{current_response}\n\nVisualization created: {chart.chart_path}"
        
        result = {
            **state,
            "chart_path": chart.chart_path,
            "response": response_text,
            "error": False
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in visualize_node: {e}")
        # Don't fail pipeline for visualization errors
        return {**state}

def knowledge_node(state: AgentState) -> AgentState:
    """Add knowledge summary with error handling"""
    try:
        validate_state(state, required_fields=['original_query'])
        
        current_response = state.get("response", "")

        summary = get_knowledge_summary.invoke({
            "query": state["original_query"],
            "memory_context": state.get("memory_context", "")
        })
        
        if summary and isinstance(summary, str):
            response_text = f"{current_response}\n\n{summary}"
        else:
            response_text = current_response

        return {
            **state,
            "tip": None,
            "response": response_text,
            "error": False
        }

    except Exception as e:
        logger.error(f"Error in knowledge_node: {e}")
        # Don't fail pipeline for knowledge errors
        return {**state}

def build_supervisor_graph():
    """Build and compile the supervisor graph with error handling"""
    try:
        graph = StateGraph(AgentState)

        # Add all nodes
        graph.add_node("classify", classify_node)
        graph.add_node("small_talk", small_talk_node)
        graph.add_node("data_fetcher", data_fetcher_node)
        graph.add_node("insights", insights_node)
        graph.add_node("visualizer", visualize_node)
        graph.add_node("knowledge", knowledge_node)

        # Set entry point
        graph.set_entry_point("classify")

        # Define routing function with validation
        def route_decision(state: AgentState) -> str:
            intent = state.get("intent", "unknown")
            subagent = state.get("subagent", "unknown")
            has_error = state.get("error", False)
            
            logger.info(f"Router Routing - intent: {intent}, subagent: {subagent}, error: {has_error}")
            
            # If there was an error in classification, end gracefully
            if has_error or intent == "error":
                return "end"
            
            if intent == "small_talk":
                return "small_talk"
            
            if intent == "finops_query":
                if subagent == "data_fetcher":
                    return "data_fetcher"
                if subagent == "insight_agent":
                    return "insights"
            
            # Default to END if no match
            logger.warning(f"No route matched for intent={intent}, subagent={subagent}")
            return "end"

        # Add conditional edges from classify
        graph.add_conditional_edges(
            "classify",
            route_decision,
            {
                "small_talk": "small_talk",
                "data_fetcher": "data_fetcher",
                "insights": "insights",
                "end": END,
            },
        )

        # Add terminal edges
        graph.add_edge("small_talk", END)
        graph.add_edge("data_fetcher", "knowledge")
        graph.add_edge("insights", "visualizer")
        graph.add_edge("visualizer", "knowledge")
        graph.add_edge("knowledge", END)

        # Compile and return
        compiled_graph = graph.compile()
        logger.info("Supervisor graph compiled successfully")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"Error building supervisor graph: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_supervisor(state: dict, csv_path: str):
    """
    Execute supervisor pipeline with comprehensive error handling
    """
    try:
        # Validate inputs
        if not state or not isinstance(state, dict):
            raise ValidationError("State must be a non-empty dictionary")
        
        query = state.get("original_query")
        if not query:
            raise ValidationError("Missing 'original_query' in state")
        
        if not csv_path:
            raise ValidationError("CSV path is required")

        logger.info(f"Supervisor Executing for query: {query}")

        # Prepare initial state with validation
        initial_state: AgentState = {
            "original_query": query,
            "session_id": state.get("session_id", "default"),
            "memory_context": state.get("memory_context", "No previous context."),
            "csv_path": csv_path,
            "intent": None,
            "category": None,
            "subagent": None,
            "confidence": 0.0,
            "dataframe_path": None,
            "chart_path": None,
            "insight_details": None,
            "tip": None,
            "response": None,
            "extracted_entities": None,
            "sql_query": None,
            "sql_result": None,
            "error": False,
            "error_message": None
        }

        # Build and invoke graph
        finops_supervisor_app = build_supervisor_graph()
        result_state = finops_supervisor_app.invoke(initial_state)

        # Validate result state
        if not isinstance(result_state, dict):
            raise ValidationError(f"Invalid result state type: {type(result_state)}")

        logger.info("Supervisor executed successfully")
        
        # Extract response with fallback
        final_response = result_state.get("response")
        if not final_response or final_response == "None" or final_response is None:
            if result_state.get("error"):
                final_response = result_state.get("error_message", "An error occurred during processing")
            else:
                final_response = "Pipeline completed but no response was generated"
        
        return {
            "response": str(final_response),
            "chart_path": result_state.get("chart_path"),
            "error": result_state.get("error", False)
        }

    except ValidationError as e:
        logger.error(f"Validation error in run_supervisor: {e}")
        return {
            "response": f"Validation Error: {str(e)}",
            "chart_path": None,
            "error": True
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in run_supervisor: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": "An unexpected error occurred. Please try again or contact support",
            "chart_path": None,
            "error": True
        }