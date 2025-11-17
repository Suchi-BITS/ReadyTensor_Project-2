import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from pydantic import BaseModel, Field
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()


class VisualizerResult(BaseModel):
    summary: str = Field(..., description="Short description of what the visualization represents.")
    chart_path: str = Field(..., description="Path to the generated chart image.")
    execution_status: str = Field(..., description="Status of visualization execution ('success' or 'error').")


def visualize_data(csv_path: str, query: str = "") -> VisualizerResult:
    """
    Generates either a bar chart (top N services/products) or trend line (by date).
    Adds full debug logging and always creates a plot if data is valid.
    """
    if not os.path.exists(csv_path):
        return VisualizerResult(summary="❌ CSV file not found.", chart_path="", execution_status="error")

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"[Visualizer] Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

        os.makedirs("results", exist_ok=True)
        chart_path = os.path.abspath(os.path.join("results", "auto_visualization.png"))

        # --- Detect columns ---
        date_cols = [c for c in df.columns if "date" in c.lower()]
        cost_cols = [c for c in df.columns if "cost" in c.lower() or "amount" in c.lower()]
        service_cols = [c for c in df.columns if any(k in c.lower() for k in ["service", "product", "category"])]
        if not service_cols:
            service_cols = [c for c in df.columns if "resource" in c.lower()]

        logger.info(f"[Visualizer] Detected columns → date={date_cols}, cost={cost_cols}, service={service_cols}")

        if not cost_cols:
            return VisualizerResult(
                summary="⚠️ No cost-related column found in CSV.",
                chart_path="",
                execution_status="error",
            )

        cost_col = cost_cols[0]
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)

        # --- Choose chart type ---
        is_bar_query = any(w in query.lower() for w in ["bar", "top", "service", "product", "category"])
        is_trend_query = any(w in query.lower() for w in ["trend", "month", "daily", "weekly", "time"])

        # === BAR CHART ===
        if is_bar_query and service_cols:
            service_col = service_cols[0]
            logger.info(f"[Visualizer] Creating bar chart using: {service_col} vs {cost_col}")

            df_top = (
                df.groupby(service_col, dropna=True)[cost_col]
                .sum()
                .nlargest(3)
                .reset_index()
            )

            if df_top.empty:
                return VisualizerResult(summary="⚠️ No data to plot after grouping.", chart_path="", execution_status="error")

            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(9, 6))

            bars = ax.barh(df_top[service_col], df_top[cost_col],
                           color=cm.Blues_r(range(len(df_top))))

            for i, bar in enumerate(bars):
                val = df_top[cost_col].iloc[i]
                ax.text(val + (0.01 * df_top[cost_col].max()),
                        bar.get_y() + bar.get_height()/2,
                        f"${val:,.0f}", va='center', ha='left', fontsize=10, fontweight='bold')

            ax.set_xlabel("Total Cost (USD)")
            ax.set_ylabel(service_col.replace("_", " ").title())
            ax.set_title("Top 3 Services by Spend", fontsize=15, fontweight='bold')
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
            fig.subplots_adjust(left=0.3, bottom=0.15)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"[Visualizer] Saved bar chart → {chart_path}")
            summary = f"Bar chart showing top 3 {service_col} by spend saved at {chart_path}"
            return VisualizerResult(summary=summary, chart_path=chart_path, execution_status="success")

        # === TREND CHART ===
        elif is_trend_query and date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            trend_df = df.groupby(df[date_col].dt.to_period("M"))[cost_col].sum().reset_index()
            trend_df[date_col] = trend_df[date_col].astype(str)

            if trend_df.empty:
                return VisualizerResult(summary="⚠️ No data to plot trend.", chart_path="", execution_status="error")

            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(trend_df[date_col], trend_df[cost_col], marker="o", linewidth=2, color="steelblue")
            ax.set_title("Monthly Cloud Cost Trend", fontsize=15, fontweight='bold')
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Cost (USD)")
            plt.xticks(rotation=45, ha='right')
            fig.subplots_adjust(bottom=0.25)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"[Visualizer] Saved trend chart → {chart_path}")
            summary = f"Trend chart saved at {chart_path}"
            return VisualizerResult(summary=summary, chart_path=chart_path, execution_status="success")

        else:
            logger.warning("[Visualizer] No matching visualization type detected.")
            return VisualizerResult(
                summary="⚠️ Could not detect suitable visualization columns.",
                chart_path="",
                execution_status="error"
            )

    except Exception as e:
        logger.error(f"[Visualizer] Visualization failed: {e}", exc_info=True)
        return VisualizerResult(summary=f"⚠️ Visualization failed: {e}", chart_path="", execution_status="error")


# --- backward compatibility ---
def visualize_trend(csv_path: str, query: str = ""):
    return visualize_data(csv_path, query)
