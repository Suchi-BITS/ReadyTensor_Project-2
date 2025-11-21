# agents/visualizerAgent.py
"""
Visualizer agent:
- Accepts either a pandas.DataFrame or a csv_path (result from text2sql)
- Generates dynamic visualizations (bar, line, area, pie, top-n) based on user intent
- Saves plots to results/ and returns metadata

Assumptions:
- Results saved by text2sql are CSV files in results/
- All plots saved to results/ as PNG
- Uses gpt-4o-mini only to help determine plot type & axis hints when ambiguous
"""
import os
import io
import re
import math
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from utils.logger_setup import setup_execution_logger
from dotenv import load_dotenv
load_dotenv()
logger = setup_execution_logger()

RESULTS_DIR = os.getenv("FINOPS_RESULTS_DIR", "results")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

def _infer_plot_intent(user_query: str) -> Dict[str, Any]:
    """
    Simple heuristic to detect plotting intent and some hints.
    Returns dict: {"plot": bool, "kind": "bar/line/pie", "x_hint": ..., "y_hint": ...}
    """
    q = user_query.lower()
    hints = {"plot": False, "kind": None, "x_hint": None, "y_hint": None, "top_n": None}
    plot_words = ["plot", "chart", "graph", "visualize", "show", "draw"]
    if any(w in q for w in plot_words):
        hints["plot"] = True

    if "trend" in q or "over time" in q or "last" in q or "month" in q:
        hints["kind"] = "line"
    if "bar" in q or "by" in q or "group by" in q or "top" in q:
        hints["kind"] = "bar"
    if "pie" in q:
        hints["kind"] = "pie"

    # find top N
    m = re.search(r"top\s+(\d+)", q)
    if m:
        hints["top_n"] = int(m.group(1))

    # hints for x/y
    # common phrase: "by servicename", "cost by region"
    m_by = re.search(r"by\s+([a-z0-9_ ]+)", q)
    if m_by:
        hints["x_hint"] = m_by.group(1).strip()

    # y hint: cost/spend/billedcost/effectivecost
    if "billed" in q or "billedcost" in q:
        hints["y_hint"] = "BilledCost"
    elif "list cost" in q or "listcost" in q:
        hints["y_hint"] = "ListCost"
    elif "contracted" in q:
        hints["y_hint"] = "ContractedCost"
    else:
        # default mapping requested by you: EffectiveCost
        hints["y_hint"] = "EffectiveCost"

    return hints

def _clean_col(col: str) -> str:
    return col.strip()

def _choose_plot_kind(df: pd.DataFrame, intent: Dict[str, Any]) -> str:
    if intent.get("kind"):
        return intent["kind"]
    # default heuristics
    if df.shape[1] == 1:
        return "bar"
    # if time-like index or first column is date-like -> line
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col], errors="raise")
        return "line"
    except Exception:
        pass
    return "bar"

def _top_n_aggregate(df: pd.DataFrame, x_col: str, y_col: str, top_n: Optional[int]):
    grouped = df.groupby(x_col)[y_col].sum().reset_index()
    grouped = grouped.sort_values(by=y_col, ascending=False)
    if top_n:
        grouped = grouped.head(top_n)
    return grouped

def _save_plot(fig, prefix="plot"):
    ensure_results_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{prefix}_{ts}.png"
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def visualize_from_dataframe(
    df: pd.DataFrame,
    user_query: str,
    prefer_plot: Optional[bool] = True
) -> Dict[str, Any]:
    """
    Create a visualization from df according to user_query.
    Returns dict with keys: chart_path, caption, error, error_message
    """
    try:
        if df is None or df.empty:
            return {"chart_path": None, "caption": "No data to plot", "error": True, "error_message": "Empty dataframe"}

        # Basic cleaning
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        # Infer intent
        intent = _infer_plot_intent(user_query)
        logger.debug("Visualizer intent: %s", intent)

        # Auto-detect x and y
        y_col = intent.get("y_hint")
        # attempt to find exact column case-insensitively
        cols_lower = {c.lower(): c for c in df.columns}
        if y_col and y_col.lower() in cols_lower:
            y_col = cols_lower[y_col.lower()]
        else:
            # fallback: find any 'effective' or 'cost' substring
            cost_candidates = [c for c in df.columns if "effective" in c.lower() or "cost" in c.lower() or "amount" in c.lower()]
            y_col = cost_candidates[0] if cost_candidates else df.columns[-1]

        x_col = intent.get("x_hint")
        if x_col:
            # try to match column name
            x_col = x_col.strip().lower()
            if x_col in cols_lower:
                x_col = cols_lower[x_col]
            else:
                # try fuzzy: match words
                matched = None
                for c in df.columns:
                    if x_col in c.lower():
                        matched = c
                        break
                if matched:
                    x_col = matched
                else:
                    # fallback to first column
                    x_col = df.columns[0]
        else:
            # if time-like column exists, use it
            time_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "period", "time", "start", "end"))]
            x_col = time_cols[0] if time_cols else df.columns[0]

        # ensure y_col numeric
        try:
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
        except Exception:
            df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors="coerce").fillna(0.0)

        kind = _choose_plot_kind(df[[x_col, y_col]], intent)

        # If grouping by x_col makes sense (categorical), aggregate top N
        chart_path = None
        caption = ""
        if kind == "bar":
            top_n = intent.get("top_n") or 10
            grouped = _top_n_aggregate(df, x_col, y_col, top_n)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(grouped[x_col].astype(str), grouped[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col} (top {min(len(grouped), top_n)})")
            plt.xticks(rotation=45, ha="right")
            chart_path = _save_plot(fig, prefix="bar")
            caption = f"Bar chart: {y_col} by {x_col} (top {min(len(grouped), top_n)})"
        elif kind == "line":
            # try to parse x_col as datetime and aggregate monthly if requested
            try:
                df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
                # If frequency words present, choose monthly aggregation
                if "month" in user_query.lower() or "monthly" in user_query.lower():
                    df_grouped = df.groupby(df[x_col].dt.to_period("M"))[y_col].sum().reset_index()
                    df_grouped[x_col] = df_grouped[x_col].astype(str)
                else:
                    df_grouped = df.sort_values(by=x_col).groupby(x_col)[y_col].sum().reset_index()
            except Exception:
                df_grouped = df.groupby(x_col)[y_col].sum().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_grouped[x_col].astype(str), df_grouped[y_col], marker="o")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} trend over {x_col}")
            plt.xticks(rotation=45, ha="right")
            chart_path = _save_plot(fig, prefix="line")
            caption = f"Line chart: {y_col} trend over {x_col}"
        elif kind == "pie":
            grouped = _top_n_aggregate(df, x_col, y_col, 10)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(grouped[y_col], labels=grouped[x_col].astype(str), autopct="%1.1f%%", startangle=140)
            ax.set_title(f"Share of {y_col} by {x_col}")
            chart_path = _save_plot(fig, prefix="pie")
            caption = f"Pie chart: share of {y_col} by {x_col}"
        else:
            # fallback: bar
            grouped = _top_n_aggregate(df, x_col, y_col, 10)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(grouped[x_col].astype(str), grouped[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col}")
            plt.xticks(rotation=45, ha="right")
            chart_path = _save_plot(fig, prefix="bar_fallback")
            caption = f"Bar chart (fallback): {y_col} by {x_col}"

        return {"chart_path": chart_path, "caption": caption, "error": False, "error_message": None}

    except Exception as e:
        logger.exception("Visualizer failed")
        return {"chart_path": None, "caption": None, "error": True, "error_message": str(e)}

def visualize_from_csv_path(csv_path: str, user_query: str):
    """
    Helper: read csv_path into df and call visualize_from_dataframe
    """
    try:
        if not csv_path or not os.path.exists(csv_path):
            return {"chart_path": None, "caption": None, "error": True, "error_message": f"CSV not found: {csv_path}"}
        df = pd.read_csv(csv_path)
        return visualize_from_dataframe(df, user_query)
    except Exception as e:
        logger.exception("Failed to visualize from csv_path")
        return {"chart_path": None, "caption": None, "error": True, "error_message": str(e)}

if __name__ == "__main__":
    # basic smoke test (developer can run locally)
    sample_csv = os.getenv("FINOPS_SAMPLE_CSV", "results/sql_result_sample.csv")
    if os.path.exists(sample_csv):
        out = visualize_from_csv_path(sample_csv, "plot cost by ServiceName top 5")
        print(out)
    else:
        print("No sample csv found; place one at results/sql_result_sample.csv to test.")
