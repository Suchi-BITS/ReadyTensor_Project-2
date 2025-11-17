# agents/supervisor.py
from datetime import datetime
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from utils.logger_setup import setup_execution_logger
import pandas as pd
import sqlite3
from langchain_openai import ChatOpenAI
import streamlit as st
import os

from agents.intent_router import classify_intent
from agents.small_talk import handle_small_talk
from agents.data_fetcher import fetch_data
from agents.insightAgent import generate_insights
from agents.visualizerAgent import visualize_data
from agents.knowledge import get_knowledge_summary

# Import your advanced tools
from agents.agentic_tools.entity_extraction import (
    extract_entities_columns_filters, 
    get_extraction_examples,
    ExtractionError
)
#from agents.agentic_tools.text2sql import build_sql_prompt

logger = setup_execution_logger()


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
    # New fields for SQL workflow
    extracted_entities: Any
    sql_query: str
    sql_result: Any


def classify_node(state: AgentState) -> AgentState:
    try:
        intent_info = classify_intent(state["original_query"])
        logger.info(f"[Classifier] intent={intent_info.get('intent')}, subagent={intent_info.get('subagent')}")
        
        result = {
            **state,
            "intent": intent_info.get("intent"),
            "subagent": intent_info.get("subagent")
        }
        return result
    except Exception as e:
        logger.error(f"Error in classify_node: {e}")
        raise


def small_talk_node(state: AgentState) -> AgentState:
    try:
        response = handle_small_talk(state["original_query"])
        result = {**state, "response": response}
        return result
    except Exception as e:
        logger.error(f"Error in small_talk_node: {e}")
        raise


def data_fetcher_node(state: AgentState) -> AgentState:
    """
    DataFetcher node with Entity Extraction:
    1. Extract entities and columns from user query
    2. Loads CSV into SQLite
    3. Generates SQL using Groq with extracted entities
    4. Executes SQL and returns results
    """
    conn = None
    try:
        import pandas as pd
        import sqlite3
        import os
        import json
        from groq import Groq

        query = state["original_query"]
        csv_path = state["csv_path"]
        logger.info(f"[DataFetcher] Processing query: {query}")
        
        # Clear any previous SQL state
        state.pop("sql_query", None)
        state.pop("sql_result", None)

        # --- STEP 1: Entity Extraction ---
        logger.info("[DataFetcher] Step 1: Extracting entities and columns...")
        extracted_entities = None
        extracted_columns = None
        
        try:
            # Get few-shot examples
            few_shot_str = get_extraction_examples(query, k=3)
            
            # Extract entities
            extraction_result = extract_entities_columns_filters.invoke({
                "user_query": query,
                "few_shot_str": few_shot_str
            })
            
            if isinstance(extraction_result, ExtractionError):
                logger.warning(f"[DataFetcher] Entity extraction failed: {extraction_result.error_message}")
            else:
                extracted_entities = [e.name for e in extraction_result.entities]
                extracted_columns = [c.name for c in extraction_result.columns_to_select]
                logger.info(f"[DataFetcher] Extracted entities: {extracted_entities}")
                logger.info(f"[DataFetcher] Extracted columns: {extracted_columns}")
                
                # Store in state
                state["extracted_entities"] = {
                    "entities": extracted_entities,
                    "columns": extracted_columns,
                    "reasoning": extraction_result.reasoning
                }
        except Exception as e:
            logger.warning(f"[DataFetcher] Entity extraction error: {e}. Continuing without it.")

        # --- STEP 2: Load CSV into in-memory SQLite ---
        logger.info("[DataFetcher] Step 2: Loading CSV into SQLite memory DB...")
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(":memory:")
        df.to_sql("finops_data", conn, index=False, if_exists="replace")
        logger.info(f"[DataFetcher] Loaded {len(df)} rows with columns: {df.columns.tolist()}")

        # --- Build schema description ---
        schema_desc = f"""
Table: finops_data
Columns: {', '.join(df.columns.tolist())}
Row count: {len(df):,}
Column types:
{df.dtypes.to_string()}
"""

        # --- STEP 3: Build enhanced SQL generation prompt with extracted entities ---
        logger.info("[DataFetcher] Step 3: Generating SQL with Groq...")
        
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
3. Return ONLY the SQL query (no explanations)
4. Use SUM(BilledCost) or SUM(EffectiveCost) for cost totals
5. Use the extracted columns when available
6. Use GROUP BY and ORDER BY as needed
7. Avoid markdown formatting
8. Handle filters, trends, and grouping logically
"""

        # --- Initialize Groq LLM (fresh client each time) ---
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or Streamlit secrets")

        # Create fresh client for each request
        client = Groq(api_key=api_key)
        
        logger.info(f"[DataFetcher] Generating SQL for query: '{query}'")

        # --- Call Groq model ---
        logger.info("[DataFetcher] Generating SQL with Groq...")
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
        logger.info(f"[DataFetcher] Raw SQL from Groq: {sql_query}")

        # --- Cleanup SQL output ---
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        # --- Validate SQL query ---
        if not sql_query or len(sql_query) < 10 or "SELECT" not in sql_query.upper():
            logger.error(f"[DataFetcher] Invalid SQL generated: {sql_query}")
            raise ValueError(f"Invalid SQL query generated. Please rephrase your question.")

        logger.info(f"[DataFetcher] Final SQL: {sql_query}")

        # --- Execute SQL ---
        logger.info("[DataFetcher] Executing SQL...")
        result_df = pd.read_sql_query(sql_query, conn)

        # --- Format results ---
        if result_df.empty:
            response_text = "Query executed successfully but returned no results."
        else:
            result_dict = result_df.to_dict("records")
            
            # Add entity extraction info to response
            entity_info = ""
            if extracted_entities or extracted_columns:
                entity_info = f"""
**Entity Extraction Results:**
- Entities: {', '.join(extracted_entities) if extracted_entities else 'None'}
- Columns: {', '.join(extracted_columns) if extracted_columns else 'None'}

"""
            
            response_text = f"""
**Query Results**

{entity_info}**SQL Query Executed:**
```sql
{sql_query}
```

**Results:**
{json.dumps(result_dict, indent=2)}

**Row Count:** {len(result_df)}
"""

        logger.info("[DataFetcher] Query executed successfully")
        
        return {
            **state,
            "sql_query": sql_query,
            "sql_result": result_dict if not result_df.empty else None,
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error in data_fetcher_node: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "response": f"Error processing query: {str(e)}"
        }
    finally:
        # Ensure connection is always closed
        if conn is not None:
            try:
                conn.close()
                logger.info("[DataFetcher] Database connection closed")
            except:
                pass


def insights_node(state: AgentState) -> AgentState:
    try:
        insights = generate_insights(state["csv_path"])
        
        result = {
            **state,
            "insight_details": insights.details,
            "response": insights.summary,
            "dataframe_path": insights.dataframe_path
        }
        return result
    except Exception as e:
        logger.error(f"Error in insights_node: {e}")
        raise


def visualize_node(state: AgentState) -> AgentState:
    try:
        if not state.get("dataframe_path"):
            result = {**state, "response": state.get("response", "No data to visualize.")}
            return result
        
        chart = visualize_data(state["dataframe_path"])
        
        current_response = state.get("response", "")
        response_text = f"{current_response}\n\nVisualization created: {chart.chart_path}"
        
        result = {
            **state,
            "chart_path": chart.chart_path,
            "response": response_text
        }
        return result
    except Exception as e:
        logger.error(f"Error in visualize_node: {e}")
        # Don't fail the whole pipeline, just skip visualization
        return {**state}


from agents.knowledge import get_knowledge_summary

def knowledge_node(state: AgentState) -> AgentState:
    try:
        current_response = state.get("response", "")

        summary = get_knowledge_summary.invoke({
            "query": state["original_query"],
            "memory_context": state.get("memory_context", "")
        })

        response_text = f"{current_response}\n\n{summary}"

        return {
            **state,
            "tip": None,
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error in knowledge_node: {e}")
        return {**state}


def build_supervisor_graph():
    """Build and compile the supervisor graph"""
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

        # Define routing function
        def route_decision(state: AgentState) -> str:
            intent = state.get("intent")
            subagent = state.get("subagent")
            
            print(f"\n{'='*60}")
            print(f"[ROUTING DEBUG]")
            print(f"Query: {state.get('original_query')}")
            print(f"Intent: {intent}")
            print(f"Subagent: {subagent}")
            print(f"{'='*60}\n")
            
            logger.info(f"[Router] Routing - intent: {intent}, subagent: {subagent}")
            
            if intent == "small_talk":
                return "small_talk"
            if intent == "finops_query":
                if subagent == "data_fetcher":
                    return "data_fetcher"
                if subagent == "insight_agent":
                    return "insights"
            
            # Default to END if no match
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
    Executes the FinOps Supervisor pipeline using the given agent state and CSV path.
    Ensures full state persistence across all LangGraph nodes.
    """
    try:
        query = state.get("original_query")
        if not query:
            raise ValueError("Missing 'original_query' in state.")

        logger.info(f"[Supervisor] Executing for query: {query}")

        # Prepare initial state with defaults
        initial_state: AgentState = {
            "original_query": query,
            "session_id": state.get("session_id", ""),
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
        }

        # Build and invoke the graph
        finops_supervisor_app = build_supervisor_graph()
        result_state = finops_supervisor_app.invoke(initial_state)

        logger.info("Supervisor executed successfully.")
        
        final_response = result_state.get("response")
        if not final_response or final_response == "None" or final_response is None:
            final_response = "Pipeline completed but no response was generated."
        
        return {
            "response": final_response,
            "chart_path": result_state.get("chart_path"),
        }

    except Exception as e:
        logger.error(f"Error running supervisor: {e}")
        import traceback
        traceback.print_exc()
        return {"response": f"Error running supervisor: {e}", "chart_path": None}