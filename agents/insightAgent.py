import os
import ast
import builtins
from typing import Any, Dict, Optional, List
import pandas as pd
from datetime import datetime

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


class LocalPythonREPL:
    """
    Minimal sandboxed execution environment with memory awareness
    """

    def __init__(self):
        safe_builtins = {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "range": range,
            "enumerate": enumerate,
        }

        self.globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "datetime": datetime,
        }

    def run(self, code: str, timeout_seconds: Optional[int] = None) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        local_ns: Dict[str, Any] = {}

        try:
            exec(code, self.globals, local_ns)

            if "result" not in local_ns:
                return "ERROR: No result variable produced by REPL code"

            result_obj = local_ns["result"]

            def _normalize(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, pd.Series):
                    return obj.to_list()
                if isinstance(obj, dict):
                    return {k: _normalize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_normalize(v) for v in obj]
                return obj

            normalized = _normalize(result_obj)
            return repr(normalized)

        except Exception as exc:
            logger.exception("Error executing REPL code")
            return f"ERROR: {exc}"


python_repl = LocalPythonREPL()


def _validate_csv_path(csv_path: Any) -> str:
    if not isinstance(csv_path, str):
        raise TypeError("csv_path must be a string")
    sanitized = csv_path.strip()
    if sanitized == "":
        raise ValueError("csv_path cannot be empty")
    forbidden = ["..", "$(", "`", "|", ";", "&", "\\\\"]
    for f in forbidden:
        if f in sanitized:
            raise ValueError("csv_path contains unsafe sequence")
    return sanitized


def _safe_parse_repl_output(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("REPL output must be a string")
    if raw.startswith("ERROR:"):
        raise ValueError(raw)
    try:
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed REPL output is not a dict")
        return parsed
    except Exception as exc:
        raise ValueError(f"Failed to parse REPL output: {exc}")


@tool
def generate_insights(
    csv_path: str,
    memory_context: Optional[str] = None,
    remembered_entities: Optional[Dict[str, Any]] = None,
    previous_insights: Optional[List[str]] = None
) -> str:
    """
    Analyze CSV data with memory awareness to provide contextual insights
    
    Args:
        csv_path: Path to CSV file
        memory_context: Previous conversation context
        remembered_entities: Previously extracted entities
        previous_insights: List of insights from previous turns
    
    Returns:
        Combined Python analysis and AI insight with memory context
    """

    try:
        # Validate input
        try:
            csv_path = _validate_csv_path(csv_path)
        except Exception as e:
            logger.error("Invalid csv_path provided: %s", e)
            return f"Invalid csv_path: {e}"

        if not os.path.exists(csv_path):
            msg = f"CSV file not found at: {csv_path}"
            logger.error(msg)
            return msg

        # Build analysis code
        repl_code = f"""
import pandas as pd

df = pd.read_csv(r'''{csv_path}''')

date_cols = [c for c in df.columns if 'date' in c.lower()]
cost_cols = [c for c in df.columns if 'cost' in c.lower() or 'amount' in c.lower()]

summary = {{
    "rows": int(len(df)),
    "columns": list(df.columns)
}}

trend = None
if date_cols and cost_cols:
    try:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        trend_df = df.groupby(df[date_cols[0]].dt.to_period('M'))[cost_cols[0]].sum().reset_index()
        trend_df[date_cols[0]] = trend_df[date_cols[0]].astype(str)
        trend = trend_df
    except Exception:
        trend = None

result = {{
    "summary": summary,
    "trend": trend.to_dict(orient="records") if trend is not None else None
}}
"""

        logger.info("Running local python analysis with memory context")
        raw_output = python_repl.run(repl_code)
        logger.debug("Raw REPL output: %s", raw_output)

        try:
            analysis = _safe_parse_repl_output(raw_output)
        except Exception as e:
            logger.error("REPL parse failed: %s", e)
            return f"Failed to parse python REPL output: {e}\nRaw output:\n{raw_output}"

        # Load prompt
        try:
            prompt_text = load_prompt_from_hub("insight_agent")
            logger.info("Loaded prompt from hub")
        except Exception as e:
            logger.warning("Could not load prompt from hub: %s. Using fallback.", e)
            prompt_text = (
                "You are a FinOps Insight Agent with conversation memory. "
                "Provide contextual insights based on current and previous analysis."
            )

        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not found. Returning Python analysis only.")
            python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\nTrend: {analysis.get('trend')}"
            return python_part

        # Build context with memory
        context_lines = [
            f"Rows: {analysis['summary'].get('rows')}",
            f"Columns: {', '.join(analysis['summary'].get('columns', [])[:20])}",
        ]
        
        if analysis.get("trend"):
            context_lines.append("Monthly trend (first 10 rows):")
            for row in analysis["trend"][:10]:
                context_lines.append(str(row))
        else:
            context_lines.append("No monthly trend available from the data.")
        
        # Add memory context
        if memory_context:
            context_lines.append("\nPrevious Conversation Context:")
            context_lines.append(memory_context[:500])
        
        # Add remembered entities
        if remembered_entities:
            context_lines.append("\nRemembered Context:")
            if remembered_entities.get("last_query_type"):
                context_lines.append(f"Previous query type: {remembered_entities['last_query_type']}")
            if remembered_entities.get("last_filters"):
                context_lines.append(f"Previous filters: {remembered_entities['last_filters']}")
        
        # Add previous insights
        if previous_insights:
            context_lines.append("\nPrevious Insights:")
            for insight in previous_insights[-3:]:
                context_lines.append(f"- {insight[:100]}")

        context = "\n".join(context_lines)

        # Call Groq LLM with memory-enriched context
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
            
            system_prompt = f"{prompt_text}\n\nIMPORTANT: Reference previous context and insights when relevant to provide coherent, contextual analysis."
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context),
            ]
            logger.info("Invoking Groq LLM for insights with memory context")
            response = llm.invoke(messages)
            ai_insight = getattr(response, "content", str(response)).strip()
            logger.info("Groq LLM returned insights")
        except Exception as e:
            logger.exception("LLM invocation failed")
            ai_insight = f"LLM insight generation failed: {e}"

        python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\n"
        python_part += f"Trend (sample): {analysis.get('trend')[:5] if analysis.get('trend') else 'None'}\n"

        final_output = f"{python_part}\nAI Insight (with memory context):\n{ai_insight}"
        return final_output

    except Exception as exc:
        logger.exception("Unexpected error in generate_insights")
        return f"Unexpected error in generate_insights: {exc}"

