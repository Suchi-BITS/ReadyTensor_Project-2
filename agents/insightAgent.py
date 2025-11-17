# agents/insight_agent.py

import os
import ast
from typing import Any, Dict
from langchain_core.tools import tool
#from langchain.tools import PythonREPLTool
#from langchain_community.tools.python.tool import PythonREPLTool
from langchain_experimental.tools import PythonREPLTool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger

logger = setup_execution_logger()
python_repl = PythonREPLTool()


@tool
def generate_insights(csv_path: str) -> str:
    """
    Insight agent that:
      - executes data analysis in a Python REPL (pandas)
      - sends the computed summary/trend to Groq (prompt taken from prompts library)
      - returns combined output (python analysis + LLM insight)

    Uses:
      - utils.prompt_loader.load_prompt_from_hub("insight_agent") to load the LLM prompt
      - PythonREPLTool to run pandas code and produce a result dict
    """
    try:
        if not os.path.exists(csv_path):
            msg = f"CSV file not found at: {csv_path}"
            logger.error(msg)
            return msg

        # Build the python code that will run inside the REPL.
        # The code returns a Python dict named `result`.
        repl_code = f"""
import pandas as pd
from datetime import datetime

df = pd.read_csv(r'''{csv_path}''')

# auto-detect date and cost columns
date_cols = [c for c in df.columns if 'date' in c.lower()]
cost_cols = [c for c in df.columns if 'cost' in c.lower() or 'amount' in c.lower()]

summary = {{
    "rows": len(df),
    "columns": list(df.columns)
}}

trend = None
if date_cols and cost_cols:
    try:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        # group by month period and sum cost
        trend_df = df.groupby(df[date_cols[0]].dt.to_period('M'))[cost_cols[0]].sum().reset_index()
        trend_df[date_cols[0]] = trend_df[date_cols[0]].astype(str)
        trend = trend_df
    except Exception as e:
        trend = None

result = {{
    "summary": summary,
    "trend": trend.to_dict(orient="records") if trend is not None else None
}}

result
"""

        logger.info("Running data analysis in Python REPL...")
        repl_output = python_repl.run(repl_code)
        logger.debug(f"Raw REPL output: {repl_output!r}")

        # Safe parse of the REPL string to Python object
        try:
            analysis = ast.literal_eval(repl_output)
        except Exception as e:
            logger.error(f"Failed to parse REPL output: {e}")
            return f"Failed to parse python REPL output: {e}\nRaw output:\n{repl_output}"

        # Load prompt from your prompts library (fallback to a simple built-in prompt)
        try:
            prompt_text = load_prompt_from_hub("insight_agent")
            logger.info("Loaded prompt from prompt hub: insight_agent")
        except Exception as e:
            logger.warning(f"Could not load prompt from hub: {e}. Using default prompt.")
            prompt_text = (
                "You are a FinOps Insight Agent. Given the dataset summary and monthly trend, "
                "produce a concise, business-focused response with: "
                "- one-line overview, "
                "- two key findings, "
                "- one actionable recommendation. "
                "Answer in plain text (no JSON)."
            )

        # If GROQ API key not present, return python analysis only
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not found. Returning Python analysis only.")
            python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\nTrend: {analysis.get('trend')}"
            return python_part

        # Prepare context for the LLM
        context_lines = [
            f"Rows: {analysis['summary'].get('rows')}",
            f"Columns: {', '.join(analysis['summary'].get('columns', [])[:20])}",
        ]
        if analysis.get("trend"):
            context_lines.append("Monthly trend (first 10 rows):")
            # include up to 10 rows of trend for compact context
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
            logger.info("Invoking Groq LLM for insights...")
            response = llm.invoke(messages)
            ai_insight = response.content.strip()
            logger.info("Groq LLM returned insights.")
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            ai_insight = f"LLM insight generation failed: {e}"

        # Combine analysis + AI insight
        python_part = f"Python Analysis:\nSummary: {analysis.get('summary')}\n"
        python_part += f"Trend (sample): {analysis.get('trend')[:5] if analysis.get('trend') else 'None'}\n"

        final_output = (
            f"{python_part}\n"
            f"AI Insight:\n{ai_insight}"
        )

        return final_output

    except Exception as exc:
        logger.exception("Unexpected error in generate_finops_insights")
        return f"Unexpected error in generate_finops_insights: {exc}"
if __name__ == "__main__":
    print("Insight Agent loaded successfully")
