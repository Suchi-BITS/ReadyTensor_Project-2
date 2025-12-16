import os
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL

from utils.logger_setup import setup_execution_logger

load_dotenv()
logger = setup_execution_logger()

# -------------------------------------------------------------------
# OpenAI client (single, consistent style)
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------------------------------------------
# Python execution (STABLE wrapper)
# -------------------------------------------------------------------
_python_repl = PythonREPL()

def run_python(code: str) -> str:
    """
    Execute Python code via PythonREPL with a stable interface.
    This function is the ONLY place PythonREPL is touched.
    Tests mock THIS function.
    """
    try:
        # Newer versions
        return _python_repl.run_code(code)
    except AttributeError:
        # Older versions: callable
        return _python_repl(code)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
RESULTS_DIR = os.getenv("FINOPS_RESULTS_DIR", "results")
DEFAULT_COST_COL = "EffectiveCost"
FALLBACK_DATE_COLS = [
    "ChargePeriodStart",
    "BillingPeriodStart",
    "ChargePeriodEnd",
    "BillingPeriodEnd",
]

COMMON_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%Y-%m",
    "%b %Y",
    "%Y%m%d",
]

DETECTED_DATE_FORMATS: Dict[str, str] = {}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_dataframe(csv_path: Optional[str], df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV result not found: {csv_path}")
    return pd.read_csv(csv_path)


def _find_cost_column(df: pd.DataFrame, user_query: Optional[str]) -> str:
    q = (user_query or "").lower()

    if "billed" in q:
        for c in df.columns:
            if c.lower() == "billedcost":
                return c

    for c in df.columns:
        if c.lower() == DEFAULT_COST_COL.lower():
            return c

    candidates = [c for c in df.columns if "cost" in c.lower() or "amount" in c.lower()]
    return candidates[0] if candidates else df.columns[-1]


def _detect_date_format(series: pd.Series) -> Optional[str]:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return None

    for fmt in COMMON_DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            return fmt
    return None


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for fallback in FALLBACK_DATE_COLS:
        for col in df.columns:
            if col.lower() == fallback.lower():
                fmt = _detect_date_format(df[col])
                if fmt:
                    DETECTED_DATE_FORMATS[col] = fmt
                return col

    for col in df.columns:
        fmt = _detect_date_format(df[col])
        if fmt:
            DETECTED_DATE_FORMATS[col] = fmt
            return col

    return None


def _basic_python_analysis(
    df: pd.DataFrame,
    cost_col: str,
    date_col: Optional[str]
) -> Dict[str, Any]:

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "cost_column": cost_col,
        "total_cost": float(df[cost_col].sum()) if cost_col in df.columns else 0.0,
    }

    top_services = None
    if "ServiceName" in df.columns:
        try:
            top_services = (
                df.groupby("ServiceName")[cost_col]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
                .to_dict(orient="records")
            )
        except Exception:
            top_services = None

    time_series = None
    if date_col and date_col in DETECTED_DATE_FORMATS:
        try:
            fmt = DETECTED_DATE_FORMATS[date_col]
            tmp = df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], format=fmt, errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if not tmp.empty:
                monthly = (
                    tmp.groupby(tmp[date_col].dt.to_period("M"))[cost_col]
                    .sum()
                    .reset_index()
                )
                monthly[date_col] = monthly[date_col].astype(str)
                time_series = monthly.to_dict(orient="records")
        except Exception:
            time_series = None

    anomalies = []
    try:
        vals = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
        mean, std = vals.mean(), vals.std()
        if std > 0:
            z = (vals - mean) / std
            mask = z.abs() > 3
            if mask.any():
                cols = [c for c in ["ResourceId", "ServiceName", cost_col] if c in df.columns]
                anomalies = df.loc[mask, cols].head(10).to_dict(orient="records")
    except Exception:
        anomalies = []

    return {
        "summary": summary,
        "top_services": top_services,
        "time_series_sample": time_series,
        "anomalies": anomalies,
    }

# -------------------------------------------------------------------
# LLM Insight
# -------------------------------------------------------------------
def _ask_llm_for_insight(
    analysis: Dict[str, Any],
    user_query: str,
    model: str = "gpt-4o-mini",
) -> str:

    if openai_client is None:
        s = analysis.get("summary", {})
        return f"Python analysis: {s.get('rows', 0)} rows, total cost ≈ {s.get('total_cost', 0.0):.2f}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a FinOps assistant. Summarize the analysis in 3–6 sentences. "
                "Highlight cost drivers, trends, anomalies, and one next action."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"query": user_query, "analysis": analysis},
                default=str,
                indent=2,
            ),
        },
    ]

    for attempt in range(1, 3):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
                timeout=20,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("LLM attempt %s failed: %s", attempt, e)
            if attempt == 2:
                break
            time.sleep(2 ** (attempt - 1))

    s = analysis.get("summary", {})
    return f"Python analysis: {s.get('rows', 0)} rows, total cost ≈ {s.get('total_cost', 0.0):.2f} (LLM failed)"

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def generate_insights(
    user_query: str,
    csv_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    schema_context: Optional[Any] = None,
    save_dataframe: bool = True,
) -> Dict[str, Any]:

    try:
        df_local = _load_dataframe(csv_path, df)

        if df_local.empty:
            return {
                "summary": "No results to analyze.",
                "analysis": {},
                "dataframe_path": None,
                "error": False,
                "error_message": None,
            }

        cost_col = _find_cost_column(df_local, user_query)
        date_col = _find_date_column(df_local)

        df_local[cost_col] = pd.to_numeric(df_local[cost_col], errors="coerce").fillna(0.0)

        analysis = _basic_python_analysis(df_local, cost_col, date_col)

        dataframe_path = None
        if save_dataframe:
            ensure_results_dir()
            name = f"insight_df_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
            dataframe_path = os.path.join(RESULTS_DIR, name)
            try:
                df_local.to_csv(dataframe_path, index=False)
            except Exception:
                dataframe_path = None

        summary = _ask_llm_for_insight(analysis, user_query)

        return {
            "summary": summary,
            "analysis": analysis,
            "dataframe_path": dataframe_path,
            "error": False,
            "error_message": None,
        }

    except FileNotFoundError as e:
        return {
            "summary": str(e),
            "analysis": {},
            "dataframe_path": None,
            "error": True,
            "error_message": str(e),
        }

    except Exception as e:
        logger.exception("generate_insights failed")
        return {
            "summary": f"Insight generation failed: {e}",
            "analysis": {},
            "dataframe_path": None,
            "error": True,
            "error_message": str(e),
        }


if __name__ == "__main__":
    sample = os.getenv("FINOPS_SAMPLE_CSV")
    if sample and os.path.exists(sample):
        out = generate_insights("Give insights", csv_path=sample)
        print(out["summary"])
