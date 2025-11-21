"""
Insight Agent (patched): date parsing warnings fixed.

Changes made:
- _find_date_column now attempts to detect a strict datetime format by sampling column values
  using a list of common formats. If a format is detected with high confidence (>70%), it
  is recorded and used for strict parsing later.
- If no strict format is detected, we do NOT attempt a generic `pd.to_datetime` call that
  falls back to dateutil (which caused the warnings). Instead we treat `date_col` as None
  and skip time-series parsing.
- The time-series parsing in _basic_python_analysis uses the detected strict format (if any).
- Added small helper and a module-level cache DETECTED_DATE_FORMATS to hold detected formats.
- Kept behavior deterministic and avoided global warnings filtering.

This preserves insight generation while preventing the pandas "Could not infer format" warnings
and the deprecation warning for `infer_datetime_format`.
"""
import os
import re
import math
import tempfile
import json
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from utils.logger_setup import setup_execution_logger
from dotenv import load_dotenv
load_dotenv()
logger = setup_execution_logger()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

RESULTS_DIR = os.getenv("FINOPS_RESULTS_DIR", "results")
DEFAULT_COST_COL = "EffectiveCost"
FALLBACK_DATE_COLS = ["ChargePeriodStart", "BillingPeriodStart", "ChargePeriodEnd", "BillingPeriodEnd"]

# Cache for detected date formats: { column_name: format_string }
DETECTED_DATE_FORMATS: Dict[str, str] = {}

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
    "%d %b %Y %H:%M:%S",
]


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_dataframe(csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV result not found: {csv_path}")
    return pd.read_csv(csv_path)


def _find_cost_column(df: pd.DataFrame, user_query: Optional[str] = None) -> str:
    # explicit mention overrides
    q = (user_query or "").lower()
    if "billed" in q or "billedcost" in q:
        for c in df.columns:
            if c.lower() == "billedcost":
                return c
    # default mapping requested: EffectiveCost
    for c in df.columns:
        if c.lower() == "effectivecost":
            return c
    # fallback candidates
    candidates = [c for c in df.columns if "cost" in c.lower() or "amount" in c.lower()]
    return candidates[0] if candidates else df.columns[-1]


def _detect_date_format_for_column(series: pd.Series) -> Optional[str]:
    """
    Try a set of common formats on a sample of non-null values. Return the best format
    string if one yields > 70% parseable values, otherwise return None.
    """
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return None

    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            pct_valid = parsed.notna().mean()
            if pct_valid >= 0.7:
                return fmt
        except Exception:
            # if format raises unexpectedly, skip it
            continue
    return None


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    # First look for canonical date column names
    for c in FALLBACK_DATE_COLS:
        for col in df.columns:
            if col.lower() == c.lower():
                # try detect format for this column
                fmt = _detect_date_format_for_column(df[col])
                if fmt:
                    DETECTED_DATE_FORMATS[col] = fmt
                return col

    # Fallback: try to detect any column that looks like a date using common formats
    for col in df.columns:
        try:
            fmt = _detect_date_format_for_column(df[col])
            if fmt:
                DETECTED_DATE_FORMATS[col] = fmt
                return col
        except Exception:
            continue
    return None


def _basic_python_analysis(df: pd.DataFrame, cost_col: str, date_col: Optional[str]) -> Dict[str, Any]:
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "cost_column": cost_col,
        "total_cost": float(df[cost_col].astype(float).sum()) if cost_col in df.columns else 0.0
    }
    # top 5 services
    top_services = None
    if "ServiceName" in df.columns:
        try:
            top_services = df.groupby("ServiceName")[cost_col].sum().sort_values(ascending=False).head(5).reset_index().to_dict(orient="records")
        except Exception:
            top_services = None

    # time series summary if date_col exists AND we detected a strict format
    ts_summary = None
    if date_col and date_col in df.columns and date_col in DETECTED_DATE_FORMATS:
        try:
            fmt = DETECTED_DATE_FORMATS.get(date_col)
            # strict parse using detected format (avoids pandas format inference warnings)
            df_parsed = df.copy()
            df_parsed[date_col] = pd.to_datetime(df_parsed[date_col], format=fmt, errors="coerce")
            # drop rows that failed parsing
            df_parsed = df_parsed.dropna(subset=[date_col])
            if not df_parsed.empty:
                monthly = df_parsed.groupby(df_parsed[date_col].dt.to_period("M"))[cost_col].sum().reset_index()
                monthly[date_col] = monthly[date_col].astype(str)
                ts_summary = monthly.head(12).to_dict(orient="records")
        except Exception:
            ts_summary = None

    # anomaly detection (simple z-score)
    anomalies = []
    try:
        vals = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
        mean = vals.mean()
        std = vals.std() if vals.std() > 0 else 0.0
        if std > 0:
            z = (vals - mean) / std
            mask = z.abs() > 3.0
            if mask.any():
                # choose columns that commonly exist; tolerate missing columns gracefully
                subset_cols = [c for c in ["ResourceId", "ServiceName", cost_col] if c in df.columns]
                anomalies = df.loc[mask, subset_cols].head(10).to_dict(orient="records")
    except Exception:
        anomalies = []

    analysis = {
        "summary": summary,
        "top_services": top_services,
        "time_series_sample": ts_summary,
        "anomalies": anomalies
    }
    return analysis


def _ask_llm_for_insight(analysis: Dict[str, Any], user_query: str, schema_context: Optional[Any] = None, model: str = "gpt-4o-mini") -> str:
    """
    Call gpt-4o-mini to convert the python analysis dict into a concise, actionable insight summary.
    """
    if openai_client is None:
        # fallback: craft deterministic summary
        s = analysis.get("summary", {})
        total = s.get("total_cost", 0.0)
        rows = s.get("rows", 0)
        return f"Python analysis: {rows} rows, total cost ≈ {total:.2f}. (LLM not configured.)"

    system_prompt = (
        "You are a FinOps Insights assistant. Convert the analysis dictionary into a concise, "
        "actionable summary of 3-6 sentences. Highlight top cost drivers, recent trends, and any anomalies. "
        "If possible, suggest one short next action (e.g., investigate service X or check tags)."
    )

    # Keep the message compact but informative
    user_payload = {
        "user_query": user_query,
        "analysis": analysis
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Here is the analysis (JSON):\n\n" + json.dumps(user_payload, default=str, indent=2)}
    ]

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=400
        )
        # extract content robustly
        content = ""
        if hasattr(resp, "choices"):
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.content
            else:
                content = getattr(choice, "text", str(choice))
        elif isinstance(resp, dict):
            choice = resp["choices"][0]
            content = choice.get("message", {}).get("content") or choice.get("text", "")
        return content.strip() if content else "No insight generated by LLM."
    except Exception as e:
        logger.exception("LLM insight call failed")
        # fallback deterministic summary
        s = analysis.get("summary", {})
        total = s.get("total_cost", 0.0)
        rows = s.get("rows", 0)
        return f"Python analysis: {rows} rows, total cost ≈ {total:.2f}. (LLM call failed: {e})"


def generate_insights(
    user_query: str,
    csv_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    schema_context: Optional[Any] = None,
    save_dataframe: bool = True
) -> Dict[str, Any]:
    """
    Primary entrypoint.
    Returns dict with keys:
      - summary: textual insight
      - analysis: python analysis dict
      - dataframe_path: path to saved result csv (optional)
      - error: bool
      - error_message: optional
    """
    try:
        df_local = _load_dataframe(csv_path=csv_path, df=df)
        if df_local.empty:
            return {"summary": "No results to analyze.", "analysis": {}, "dataframe_path": None, "error": False}

        cost_col = _find_cost_column(df_local, user_query)
        date_col = _find_date_column(df_local)

        # Ensure numeric cost
        try:
            df_local[cost_col] = pd.to_numeric(df_local[cost_col], errors="coerce").fillna(0.0)
        except Exception:
            df_local[cost_col] = pd.to_numeric(df_local[cost_col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors="coerce").fillna(0.0)

        analysis = _basic_python_analysis(df_local, cost_col, date_col)

        # Optionally save dataframe snapshot
        dataframe_path = None
        if save_dataframe:
            ensure_results_dir()
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"insight_df_{ts}.csv"
            dataframe_path = os.path.join(RESULTS_DIR, filename)
            try:
                df_local.to_csv(dataframe_path, index=False)
            except Exception as e:
                logger.warning("Failed to save insight dataframe: %s", e)
                dataframe_path = None

        # Ask LLM to produce final concise insight
        llm_summary = _ask_llm_for_insight(analysis, user_query, schema_context=schema_context)

        return {
            "summary": llm_summary,
            "analysis": analysis,
            "dataframe_path": dataframe_path,
            "error": False,
            "error_message": None
        }

    except FileNotFoundError as fe:
        logger.error("generate_insights file error: %s", fe)
        return {"summary": f"CSV not found: {fe}", "analysis": {}, "dataframe_path": None, "error": True, "error_message": str(fe)}
    except Exception as e:
        logger.exception("generate_insights failed")
        return {"summary": f"Insight generation failed: {e}", "analysis": {}, "dataframe_path": None, "error": True, "error_message": str(e)}

if __name__ == "__main__":
    # quick test - requires a CSV in results/sql_result_sample.csv
    sample = os.getenv("FINOPS_SAMPLE_CSV", "results/sql_result_sample.csv")
    if os.path.exists(sample):
        out = generate_insights(user_query="Give insights about cost by ServiceName", csv_path=sample)
        print(out["summary"])
    else:
        print("Place sample result at results/sql_result_sample.csv to test.")
