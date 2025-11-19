# agents/insight_agent.py

import os
import ast
import builtins
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


# ---------------------------
# Simple local REPL wrapper
# ---------------------------
class LocalPythonREPL:
    """
    Minimal sandboxed execution environment that exposes a run(code: str) -> str method.
    The executed code should set a variable named `result` which will be returned.
    The return value is the repr() of the Python object result so callers can ast.literal_eval it.
    """

    def __init__(self):
        # Prebuild a safe globals mapping with pandas available
        safe_builtins = {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "range": range,
            "enumerate": enumerate,
        }

        # Provide minimal builtins and pandas in globals
        self.globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "datetime": datetime,
        }

    def run(self, code: str, timeout_seconds: Optional[int] = None) -> str:
        """
        Execute code string in a limited environment.
        Expect the code to set a variable named `result`.
        Returns repr(result). Errors are returned as a string prefixed with 'ERROR:'.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        # Local namespace for this run
        local_ns: Dict[str, Any] = {}

        try:
            # Execute the code. We do not provide eval or direct access to builtins beyond safe set.
            exec(code, self.globals, local_ns)

            if "result" not in local_ns:
                return "ERROR: No result variable produced by REPL code"

            result_obj = local_ns["result"]

            # If result contains pandas DataFrame objects, convert them to python primitives
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
            # Use repr so the caller can ast.literal_eval it safely
            return repr(normalized)

        except Exception as exc:
            logger.exception("Error executing REPL code")
            return f"ERROR: {exc}"


# Expose an instance which tests can patch
python_repl = LocalPythonREPL()


# ---------------------------
# Utilities
# ---------------------------

def _validate_csv_path(csv_path: Any) -> str:
    if not isinstance(csv_path, str):
        raise TypeError("csv_path must be a string")
    sanitized = csv_path.strip()
    # Basic guardrails
    if sanitized == "":
        raise ValueError("csv_path cannot be empty")
    forbidden = ["..", "$(", "`", "|", ";", "&", "\\\\"]  # backslash escaped for string
    for f in forbidden:
        if f in sanitized:
            raise ValueError("csv_path contains unsafe sequence")
    return sanitized


def _safe_parse_repl_output(raw: str) -> Dict[str, Any]:
    """
    Safely parse the REPL output which is expected to be repr of a python structure.
    Returns a dictionary or raises ValueError.
    """
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


# ---------------------------
# Main tool
# ---------------------------

@tool
def generate_insights(csv_path: str) -> str:
    """
    Analyze CSV data and produce combined Python analysis and an AI insight.
    This function validates input, runs a local python analysis (in a sandbox),
    and optionally calls a Groq LLM for enrichment.

    The local REPL code must produce a variable named `result` which is a dict
    with at least a `summary` key and an optional `trend` key.
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

        # Build analysis code that will run inside the REPL
        # The code will create a dict named `result`
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

        logger.info("Running local python analysis")
        raw_output = python_repl.run(repl_code)
        logger.debug("Raw REPL output: %s", raw_output)

        try:
            analysis = _safe_parse_repl_output(raw_output)
        except Exception as e:
            logger.error("REPL parse failed: %s", e)
            return f"Failed to parse python REPL output: {e}\nRaw output:\n{raw_output}"

        # Load prompt from hub
        try:
            prompt_text = load_prompt_from_hub("insight_agent")
            logger.info("Loaded prompt from hub")
        except Exception as e:
            logger.warning("Could not load prompt from hub: %s. Using fallback.", e)
            prompt_text = (
                "You are a FinOps Insight Agent. Given dataset summary and monthly trend produce "
                "a concise business focused response: one line overview, two key findings, "
                "one actionable recommendation. Plain text only."
            )

        # If no Groq key, return Python analysis only
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not found. Returning Python analysis only.")
            python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\nTrend: {analysis.get('trend')}"
            return python_part

        # Build context for LLM
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

        context = "\n".join(context_lines)

        # Call Groq LLM
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=context),
            ]
            logger.info("Invoking Groq LLM for insights")
            response = llm.invoke(messages)
            ai_insight = getattr(response, "content", str(response)).strip()
            logger.info("Groq LLM returned insights")
        except Exception as e:
            logger.exception("LLM invocation failed")
            ai_insight = f"LLM insight generation failed: {e}"

        python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\n"
        python_part += f"Trend (sample): {analysis.get('trend')[:5] if analysis.get('trend') else 'None'}\n"

        final_output = f"{python_part}\nAI Insight:\n{ai_insight}"
        return final_output

    except Exception as exc:
        logger.exception("Unexpected error in generate_insights")
        return f"Unexpected error in generate_insights: {exc}"


if __name__ == "__main__":
    print("Insight Agent loaded successfully")
