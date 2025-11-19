import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ast
import builtins
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Optional ML imports (may require scikit-learn installed in your environment)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
except Exception:
    IsolationForest = None
    LinearRegression = None

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


# -----------------------------
# Safe analytics utilities
# -----------------------------

def detect_anomalies_zscore(df: pd.DataFrame, column: str, z_thresh: float = 3.0):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": "no numeric data"}
    z = (series - series.mean()) / (series.std() if series.std() != 0 else 1)
    anomalies = series[np.abs(z) > z_thresh]
    return anomalies.to_dict()


def detect_anomalies_isolation(df: pd.DataFrame, column: str, contamination: float = 0.05):
    if IsolationForest is None:
        return {"error": "IsolationForest not available (scikit-learn missing)"}
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.shape[0] < 5:
        return {"error": "not enough data for isolation forest"}
    model = IsolationForest(contamination=contamination, random_state=42)
    X = series.values.reshape(-1, 1)
    flags = model.fit_predict(X)
    outliers = series[flags == -1]
    return outliers.to_dict()


def forecast_linear(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 3):
    if LinearRegression is None:
        return {"error": "LinearRegression not available (scikit-learn missing)"}
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("date_col or value_col not found")
    tmp = df[[date_col, value_col]].dropna()
    if tmp.empty:
        return {"error": "no data to forecast"}
    tmp = tmp.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    tmp = tmp.sort_values(date_col)
    tmp["t"] = range(len(tmp))
    X = tmp[["t"]].values
    y = pd.to_numeric(tmp[value_col], errors="coerce").values
    if len(X) < 2 or np.all(np.isnan(y)):
        return {"error": "insufficient numeric data to train"}
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(X), len(X) + periods).reshape(-1, 1)
    preds = model.predict(future_t).tolist()
    return {
        "model": "linear_regression",
        "periods": periods,
        "predictions": preds,
    }


def moving_average(df: pd.DataFrame, column: str, window: int = 3):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": "no numeric data"}
    ma = series.rolling(window).mean().dropna()
    return ma.tolist()


def correlation_matrix(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return {"error": "not enough numeric columns for correlation"}
    corr = numeric.corr()
    return corr.to_dict()


# -----------------------------
# Local Python REPL (sandbox)
# -----------------------------
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
            "abs": abs,
        }

        # Expose pandas and datetime and the safe analytics functions
        self.globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "datetime": datetime,
            # safe functions
            "detect_anomalies_zscore": detect_anomalies_zscore,
            "detect_anomalies_isolation": detect_anomalies_isolation,
            "forecast_linear": forecast_linear,
            "moving_average": moving_average,
            "correlation_matrix": correlation_matrix,
        }

    def run(self, code: str, timeout_seconds: Optional[int] = None) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        local_ns: Dict[str, Any] = {}

        try:
            # Execute in controlled globals
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
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            normalized = _normalize(result_obj)
            return repr(normalized)

        except Exception as exc:
            logger.exception("Error executing REPL code")
            return f"ERROR: {exc}"


python_repl = LocalPythonREPL()


# -----------------------------
# Helpers
# -----------------------------

def _validate_csv_path(csv_path: Any) -> str:
    if not isinstance(csv_path, str):
        raise TypeError("csv_path must be a string")
    sanitized = csv_path.strip()
    if sanitized == "":
        raise ValueError("csv_path cannot be empty")
    forbidden = ["..", "$(`", "`", "|", ";", "&", "\\"]
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


# -----------------------------
# Main tool
# -----------------------------
@tool
def generate_insights(
    csv_path: str,
    user_query: Optional[str] = None,
    memory_context: Optional[str] = None,
    remembered_entities: Optional[Dict[str, Any]] = None,
    previous_insights: Optional[List[str]] = None,
    hybrid_mode: bool = True,
) -> str:
    """
    Analyze CSV data with memory awareness to provide contextual insights

    Args:
        csv_path: Path to CSV file
        user_query: user's natural language question about the data
        memory_context: Previous conversation context
        remembered_entities: Previously extracted entities
        previous_insights: List of insights from previous turns
        hybrid_mode: If True, allow LLM to generate safe plans; otherwise use predefined logic

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

        # Quick structural read to build metadata (read limited rows to be efficient)
        try:
            df_preview = pd.read_csv(csv_path, nrows=5000)
        except Exception as e:
            logger.exception("Failed to read CSV for preview")
            return f"Failed to read CSV: {e}"

        columns = df_preview.columns.tolist()
        dtypes = df_preview.dtypes.astype(str).to_dict()
        numeric_cols = df_preview.select_dtypes(include=["number"]).columns.tolist()
        date_cols = [c for c in columns if "date" in c.lower()]

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

        # Build context for LLM
        context_lines = [
            f"Columns: {columns}",
            f"Dtypes sample: {dict(list(dtypes.items())[:10])}",
            f"Numeric columns: {numeric_cols}",
            f"Date candidate columns: {date_cols}",
        ]

        if memory_context:
            context_lines.append("\nPrevious Conversation Context:")
            context_lines.append(memory_context[:1000])

        if remembered_entities:
            context_lines.append("\nRemembered Entities:")
            context_lines.append(str(remembered_entities))

        if previous_insights:
            context_lines.append("\nPrevious Insights (last 3):")
            for insight in previous_insights[-3:]:
                context_lines.append(insight[:300])

        if user_query:
            context_lines.append(f"\nUser Query: {user_query}")

        llm_context = "\n".join(context_lines)

        # If GROQ API key not present, fallback to safe predefined analysis
        if not os.getenv("GROQ_API_KEY") or not hybrid_mode:
            logger.info("GROQ_API_KEY not found or hybrid_mode disabled. Using safe default analyses.")

            # Default safe analyses: summary, monthly trend if date+numeric present
            summary = {
                "rows": int(len(df_preview)),
                "columns": columns,
                "numeric_columns": numeric_cols,
            }

            trend = None
            if date_cols and numeric_cols:
                try:
                    full_df = pd.read_csv(csv_path)
                    full_df[date_cols[0]] = pd.to_datetime(full_df[date_cols[0]], errors="coerce")
                    trend_df = (
                        full_df.groupby(full_df[date_cols[0]].dt.to_period("M")).agg({numeric_cols[0]: "sum"}).reset_index()
                    )
                    trend_df[date_cols[0]] = trend_df[date_cols[0]].astype(str)
                    trend = trend_df.to_dict(orient="records")
                except Exception:
                    trend = None

            python_part = f"Python Analysis:\nSummary: {summary}\nTrend (sample): {trend[:5] if trend else 'None'}\n"
            final_output = python_part + "AI Insight: (GROQ not configured) Provide detailed insights when GROQ_API_KEY is available."
            return final_output

        # Build LLM plan prompt
        analysis_plan_prompt = f"""
You are an Insight Analysis Planner. You will generate SAFE python code that uses only the following allowed functions (already available in the sandbox):
- detect_anomalies_zscore(df, column)
- detect_anomalies_isolation(df, column)
- forecast_linear(df, date_col, value_col, periods)
- moving_average(df, column, window)
- correlation_matrix(df)

You may access the DataFrame as `df` (it will be available in the sandbox during execution) and you may call only the allowed functions above and standard pandas operations.

Dataset columns: {columns}
Numeric columns: {numeric_cols}
Date candidate columns: {date_cols}
User Query: "{user_query or ''}"

Return ONLY a python snippet that sets a variable named `result` to a JSON-serializable dict. Do NOT include any imports or calls to external systems. Do NOT use forbidden keywords like import, open, os, subprocess, eval, exec, or sys.
Example safe output:
result = {{"monthly_cost": trend.to_dict(orient='records'), "forecast": forecast}}
"""

        # Invoke LLM to get plan
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
            system_prompt = f"{prompt_text}\n\nIMPORTANT: Generate only safe python that assigns `result`. Always consider previous memory when relevant."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=analysis_plan_prompt + "\n\nContext:\n" + llm_context),
            ]

            logger.info("Requesting analysis plan from LLM")
            plan_response = llm.invoke(messages)
            plan_code = getattr(plan_response, "content", str(plan_response)).strip()
            logger.debug("LLM plan: %s", plan_code)

        except Exception as e:
            logger.exception("LLM invocation for plan failed")
            return f"LLM planning failed: {e}"

        # Basic safety checks on plan_code
        forbidden = ["import\n", "open(", "os.", "sys.", "subprocess", "eval(", "exec(", "pickle", "__import__"]
        lower_plan = plan_code.lower()
        for f in forbidden:
            if f in lower_plan:
                logger.warning("Rejected unsafe plan due to forbidden token: %s", f)
                return f"Rejected unsafe plan - contains forbidden token: {f}"

        # Ensure plan defines `result`
        if "result" not in plan_code:
            logger.warning("Plan does not define 'result'. Attempting to wrap in result variable.")
            plan_code = plan_code + "\n\n# Ensure result variable exists\nif 'result' not in locals():\n    result = {'note': 'No result produced by plan'}\n"

        # Compose REPL code: load full dataframe inside REPL and run plan
        repl_code = f"""
import pandas as pd

# load full dataset
try:
    df = pd.read_csv(r'''{csv_path}''')
except Exception as e:
    result = {{'error': f'Failed to load CSV in REPL: {e}'}}

# ensure date columns are parsed lazily if present
for c in {date_cols!r}:
    if c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            pass

# user/LLM provided plan
{plan_code}

"""

        logger.info("Executing plan in local REPL")
        raw_output = python_repl.run(repl_code)
        logger.debug("Raw REPL output: %s", raw_output)

        try:
            analysis = _safe_parse_repl_output(raw_output)
        except Exception as e:
            logger.error("REPL parse failed: %s", e)
            return f"Failed to parse python REPL output: {e}\nRaw output:\n{raw_output}"

        # Call LLM to generate human-readable insight based on analysis + memory
        try:
            insight_prompt = (
                f"You are an Insight Reporter. Given the following analysis output (as Python dict):\n{analysis}\n\n"
                f"Use the memory context and previous insights when relevant:\n{llm_context}\n\n"
                "Produce a concise, actionable analysis summary with interpretation, key metrics, recommended actions, and confidence level."
            )

            messages2 = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=insight_prompt),
            ]

            logger.info("Requesting final AI insight from LLM")
            response2 = llm.invoke(messages2)
            ai_insight = getattr(response2, "content", str(response2)).strip()

        except Exception as e:
            logger.exception("LLM insight generation failed")
            ai_insight = f"LLM insight generation failed: {e}"

        python_part = f"Python Analysis:\nSummary: columns={columns}, numeric_columns={numeric_cols}\n" \
                      + f"Analysis result sample: {str(list(analysis.keys())[:10])}\n"

        final_output = f"{python_part}\nAI Insight (with memory context):\n{ai_insight}\n\nRaw Analysis Output:\n{analysis}"
        return final_output

    except Exception as exc:
        logger.exception("Unexpected error in generate_insights")
        return f"Unexpected error in generate_insights: {exc}"
