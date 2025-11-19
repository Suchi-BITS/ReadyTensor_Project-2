
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ast
import builtins
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Optional ML imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
except Exception:
    IsolationForest = None
    LinearRegression = None

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


# Result model for compatibility with supervisor
class InsightResult(BaseModel):
    summary: str = Field(..., description="Summary of insights")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed analysis")
    dataframe_path: Optional[str] = Field(default=None, description="Path to saved dataframe")


# Safe analytics utilities
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
        return {"error": "IsolationForest not available"}
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
        return {"error": "LinearRegression not available"}
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
        return {"error": "insufficient numeric data"}
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
        return {"error": "not enough numeric columns"}
    corr = numeric.corr()
    return corr.to_dict()


# Local Python REPL
class LocalPythonREPL:
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

        self.globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "datetime": datetime,
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
            exec(code, self.globals, local_ns)

            if "result" not in local_ns:
                return "ERROR: No result variable produced"

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


@tool
def generate_insights(
    csv_path: str,
    user_query: Optional[str] = None,
    memory_context: Optional[str] = None,
    remembered_entities: Optional[Dict[str, Any]] = None,
    previous_insights: Optional[List[str]] = None,
    hybrid_mode: bool = True,
) -> InsightResult:
    """
    Analyze CSV data with dynamic insights generation
    
    Returns InsightResult object for compatibility with supervisor
    """

    try:
        csv_path = _validate_csv_path(csv_path)
    except Exception as e:
        logger.error("Invalid csv_path: %s", e)
        return InsightResult(
            summary=f"Invalid csv_path: {e}",
            details=None,
            dataframe_path=None
        )

    if not os.path.exists(csv_path):
        msg = f"CSV file not found: {csv_path}"
        logger.error(msg)
        return InsightResult(summary=msg, details=None, dataframe_path=None)

    try:
        df_preview = pd.read_csv(csv_path, nrows=5000)
    except Exception as e:
        logger.exception("Failed to read CSV")
        return InsightResult(
            summary=f"Failed to read CSV: {e}",
            details=None,
            dataframe_path=None
        )

    columns = df_preview.columns.tolist()
    dtypes = df_preview.dtypes.astype(str).to_dict()
    numeric_cols = df_preview.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in columns if "date" in c.lower()]

    try:
        prompt_text = load_prompt_from_hub("insight_agent")
    except Exception as e:
        logger.warning("Could not load prompt: %s", e)
        prompt_text = "You are a FinOps Insight Agent with memory."

    context_lines = [
        f"Columns: {columns}",
        f"Numeric columns: {numeric_cols}",
        f"Date columns: {date_cols}",
    ]

    if memory_context:
        context_lines.append(f"Memory: {memory_context[:1000]}")

    if user_query:
        context_lines.append(f"Query: {user_query}")

    llm_context = "\n".join(context_lines)

    if not os.getenv("GROQ_API_KEY") or not hybrid_mode:
        logger.info("Using safe default analysis")
        
        summary = {
            "rows": int(len(df_preview)),
            "columns": columns,
            "numeric_columns": numeric_cols,
        }

        # Save dataframe for downstream use
        output_path = os.path.join("results", "insight_data.csv")
        os.makedirs("results", exist_ok=True)
        df_preview.to_csv(output_path, index=False)

        summary_text = f"Dataset has {summary['rows']} rows and {len(columns)} columns"
        
        return InsightResult(
            summary=summary_text,
            details=summary,
            dataframe_path=output_path
        )

    # LLM-based dynamic analysis
    analysis_plan_prompt = f"""
Generate safe python code using only these functions:
- detect_anomalies_zscore(df, column)
- detect_anomalies_isolation(df, column)
- forecast_linear(df, date_col, value_col, periods)
- moving_average(df, column, window)
- correlation_matrix(df)

Columns: {columns}
Numeric: {numeric_cols}
Dates: {date_cols}
Query: {user_query or ''}

Return python that sets result variable. No imports allowed.
"""

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        messages = [
            SystemMessage(content=f"{prompt_text}\nGenerate safe python code only."),
            HumanMessage(content=analysis_plan_prompt + "\n" + llm_context),
        ]

        logger.info("Requesting analysis plan")
        plan_response = llm.invoke(messages)
        plan_code = getattr(plan_response, "content", str(plan_response)).strip()

    except Exception as e:
        logger.exception("LLM planning failed")
        return InsightResult(
            summary=f"LLM planning failed: {e}",
            details=None,
            dataframe_path=None
        )

    # Safety checks
    forbidden = ["import\n", "open(", "os.", "sys.", "subprocess", "eval(", "exec("]
    for f in forbidden:
        if f in plan_code.lower():
            logger.warning("Rejected unsafe plan")
            return InsightResult(
                summary=f"Rejected unsafe plan containing: {f}",
                details=None,
                dataframe_path=None
            )

    # Execute plan
    repl_code = f"""
import pandas as pd

try:
    df = pd.read_csv(r'''{csv_path}''')
except Exception as e:
    result = {{'error': f'Failed to load: {{e}}'}}

for c in {date_cols!r}:
    if c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except:
            pass

{plan_code}
"""

    logger.info("Executing plan")
    raw_output = python_repl.run(repl_code)

    try:
        analysis = _safe_parse_repl_output(raw_output)
    except Exception as e:
        logger.error("Parse failed: %s", e)
        return InsightResult(
            summary=f"Parse failed: {e}",
            details=None,
            dataframe_path=None
        )

    # Generate insight
    try:
        insight_prompt = f"Analysis: {analysis}\nContext: {llm_context}\nProvide actionable insights."
        
        messages2 = [
            SystemMessage(content=prompt_text),
            HumanMessage(content=insight_prompt),
        ]

        response2 = llm.invoke(messages2)
        ai_insight = getattr(response2, "content", str(response2)).strip()

    except Exception as e:
        logger.exception("Insight generation failed")
        ai_insight = f"Insight generation failed: {e}"

    # Save dataframe
    output_path = os.path.join("results", "insight_data.csv")
    os.makedirs("results", exist_ok=True)
    try:
        full_df = pd.read_csv(csv_path)
        full_df.to_csv(output_path, index=False)
    except:
        output_path = None

    final_summary = f"Analysis complete. {ai_insight}"
    
    return InsightResult(
        summary=final_summary,
        details=analysis,
        dataframe_path=output_path
    )
# Visualization result model
class VisualizationResult(BaseModel):
    chart_path: str = Field(..., description="Path to the generated chart image")


def visualize_data(dataframe_path: str) -> VisualizationResult:
    """
    Load CSV → auto-detect date/numeric columns → generate visualization → save image.
    Returns VisualizationResult(chart_path=...)
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(dataframe_path):
        raise FileNotFoundError(f"Dataframe not found: {dataframe_path}")

    df = pd.read_csv(dataframe_path)

    # Detect date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found for visualization")

    # Use highest-confidence column for Y-axis
    y_col = numeric_cols[0]

    # Prefer date column if available
    if date_cols:
        x_col = date_cols[0]
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col])

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=df[x_col], y=df[y_col])
        plt.title(f"Trend of {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    else:
        # Default: bar chart of first 20 rows
        x_col = df.columns[0]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=df[x_col].astype(str).head(20), y=df[y_col].head(20))
        plt.title(f"Bar Chart of {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)

    # Output path
    os.makedirs("results/visuals", exist_ok=True)
    chart_path = os.path.join("results", "visuals", f"chart_{datetime.now().timestamp()}.png")

    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return VisualizationResult(chart_path=chart_path)
