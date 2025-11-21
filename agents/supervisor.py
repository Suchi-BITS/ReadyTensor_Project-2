# agents/supervisor.py
"""
Supervisor orchestrator:
- Receives state dict and csv_path (csv_path default: data/data.csv)
- Calls text2sql.generate_sql_and_execute to get SQL & results
- Calls insightAgent.generate_insights and visualizerAgent.visualize_from_csv_path as needed
- Returns final dict: { response, chart_path, error, intent, subagent, sql, ... }

This module avoids external dependencies (Groq/Deepseek). Uses gpt-4o-mini internally via text2sql/insight.
"""
import os
import traceback
from typing import Dict, Any, Optional
from utils.logger_setup import setup_execution_logger
from dotenv import load_dotenv
load_dotenv()
logger = setup_execution_logger()

# Local agent imports
from agents.agentic_tools.text2sql import generate_sql_and_execute
from agents.visualizerAgent import visualize_from_csv_path
from agents.insightAgent import generate_insights

# Default CSV location (confirmed)
DEFAULT_CSV = os.getenv("FINOPS_CSV_PATH", "data/data.csv")

def run_supervisor(state: Dict[str, Any], csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the supervisor pipeline for a single user query.
    Expected state contains at least {"original_query": "..."}

    Returns:
    {
      "response": str,
      "chart_path": Optional[str],
      "error": bool,
      "error_message": Optional[str],
      "sql": Optional[str],
      "intent": Optional[str],
      "subagent": Optional[str],
      "insight": Optional[dict]
    }
    """
    try:
        final_response = "" 
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")

        user_query = state.get("original_query") or state.get("query") or ""
        if not user_query or not isinstance(user_query, str):
            return {"response": "No query provided", "chart_path": None, "error": True, "error_message": "Missing original_query", "sql": None}

        csv_path = csv_path or DEFAULT_CSV
        if not os.path.exists(csv_path):
            return {"response": f"CSV not found at {csv_path}", "chart_path": None, "error": True, "error_message": "CSV missing", "sql": None}

        logger.info(f"Supervisor: processing query: {user_query}")

        # Step 1: Text2SQL -> generate SQL and execute
        t2s_out = generate_sql_and_execute(
            user_query=user_query,
            csv_path=csv_path,
            db_path=os.getenv("FINOPS_SQLITE_DB", "finops.db"),
            table_name=os.getenv("FINOPS_TABLE_NAME", "finops_data"),
            model=os.getenv("FINOPS_T2S_MODEL", "gpt-4o-mini")
        )

        if t2s_out.get("error"):
            logger.error("Text2SQL failed: %s", t2s_out.get("error_message"))
            return {"response": f"Validation Error: {t2s_out.get('error_message')}", "chart_path": None, "error": True, "error_message": t2s_out.get("error_message"), "sql": None}

        sql = t2s_out.get("sql")
        csv_result_path = t2s_out.get("csv_path")
        df = t2s_out.get("dataframe")
       
        # Step 2: Generate insights (always try)
        insight_result = generate_insights(user_query=user_query, csv_path=csv_result_path, df=df, schema_context=None)
        insight_text = insight_result.get("summary") or ""
        logger.info("Insights generated")

        # Step 3: Visualization - detect if user asked for plot; otherwise, but if they asked for "plot" etc.
        chart_path = None
        chart_info = None
        if any(w in user_query.lower() for w in ["plot", "chart", "graph", "visualize", "show", "draw"]):
            viz_out = visualize_from_csv_path(csv_result_path, user_query)
            if not viz_out.get("error"):
                chart_path = viz_out.get("chart_path")
                chart_info = viz_out.get("caption")
            else:
                logger.warning("Visualizer returned error: %s", viz_out.get("error_message"))

        # Build final response
        # Build final response
        response_lines = []
        response_lines.append(insight_text)

        # ADD DATAFRAME HERE
        df_preview = ""
        if df is not None and len(df) > 0:
            df_preview = df.to_csv(index=False)
            response_lines.append(f"\nSQL Result Preview:\n```\n{df_preview}\n```")
        if sql:
            response_lines.append(f"\nExecuted SQL:\n```\n{sql}\n```")

        if chart_path:
            response_lines.append(f"\nA visualization was generated: {chart_path}")
            final_response = "\n\n".join(response_lines)


        return {
            "response": final_response,
            "chart_path": chart_path,
            "error": False,
            "error_message": None,
            "sql": sql,
            "intent": "finops_query",
            "subagent": "data_fetcher",
            "insight": insight_result
        }

    except Exception as exc:
        logger.exception("Supervisor critical failure")
        tb = traceback.format_exc()
        print("SUPERVISOR ERROR:", str(exc))
        print("TRACEBACK:", tb)
        return {"response": "An unexpected error occurred. Please try again.", "chart_path": None, "error": True, "error_message": str(exc) + "\n" + tb, "sql": None}
