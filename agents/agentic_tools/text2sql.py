# agents/agentic_tools/text2sql.py
"""
Text2SQL module for FinOps agent.

Responsibilities:
- Load schema_context (from prompt hub) or fallback
- Ensure SQLite DB (finops.db) has table `finops_data` loaded from data/data.csv
- Generate SQLite-compatible SQL using gpt-4o-mini
- Validate SQL for security (allow only single SELECT)
- Execute SQL and return pandas DataFrame and path to saved CSV results

Usage:
from agents.agentic_tools.text2sql import generate_sql_and_execute
sql, df, csv_path = generate_sql_and_execute(user_query)
"""
import os
import json
import re
import sqlite3
import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from openai import OpenAI
from utils.logger_setup import setup_execution_logger
from dotenv import load_dotenv
load_dotenv()
logger = setup_execution_logger()

# Try to load prompt loader for schema_context
try:
    from utils.prompt_loader import load_prompt_from_hub
except Exception:
    load_prompt_from_hub = None

# Config
DB_PATH = os.getenv("FINOPS_SQLITE_DB", "finops.db")
CSV_PATH = os.getenv("FINOPS_CSV_PATH", "data/data.csv")  # confirmed by user: data/data.csv
TABLE_NAME = os.getenv("FINOPS_TABLE_NAME", "finops_data")
RESULTS_DIR = os.getenv("FINOPS_RESULTS_DIR", "results")
MAX_SQL_LENGTH = 2000

# Security patterns - if any match, SQL will be rejected
BLOCKED_SQL_PATTERNS = [
    r'(?i)\bDROP\s+TABLE\b',
    r'(?i)\bDROP\s+DATABASE\b',
    r'(?i)\bDELETE\s+FROM\b',
    r'(?i)\bTRUNCATE\b',
    r'(?i)\bALTER\s+TABLE\b',
    r'(?i)\bCREATE\s+TABLE\b',
    r'(?i)\bINSERT\s+INTO\b',
    r'(?i)\bUPDATE\s+SET\b',
    r'(?i)\bEXEC\b',
    r'(?i)\bEXECUTE\b',
    r'(?i);\s*DROP\b',
    r'(?i);\s*DELETE\b',
    r'--',  # SQL comments might be allowed for read-only but we disallow to simplify
    r'/\*',  # Block comments
]

# Prepare OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. LLM calls will fail if attempted.")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create results directory {RESULTS_DIR}: {e}")

def load_schema_context() -> Dict[str, Any]:
    """
    Load schema_context from prompt hub using utils.prompt_loader if available.
    The prompt hub is expected to return JSON (string or already parsed).
    Fallback to minimal inferred schema if not available.
    """
    if load_prompt_from_hub:
        try:
            raw = load_prompt_from_hub("schema_context")
            # If raw is JSON string, parse. If it's a dict/list already, return as-is.
            if isinstance(raw, (dict, list)):
                return raw
            raw = raw.strip()
            try:
                parsed = json.loads(raw)
                logger.info("Loaded schema_context from prompt hub (JSON parsed).")
                return parsed
            except Exception:
                # If prompt hub returns textual schema, try to extract JSON substring
                m = re.search(r'(\[.*\])', raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        logger.info("Loaded schema_context (extracted JSON).")
                        return parsed
                    except Exception:
                        pass
                # give up and return minimal fallback
                logger.warning("schema_context from prompt hub couldn't be parsed as JSON; falling back.")
        except Exception as e:
            logger.warning(f"load_prompt_from_hub('schema_context') failed: {e}")

    # Fallback minimal schema (empty) — downstream code will still function using CSV introspection
    logger.info("Using fallback empty schema_context.")
    return []

def infer_schema_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    Read the CSV and return a schema-like list (same structure expected in schema_context)
    """
    try:
        df = pd.read_csv(csv_path, nrows=1000)
    except Exception as e:
        logger.error(f"Failed to read CSV for schema inference: {e}")
        return []

    schema = []
    for col in df.columns:
        col_series = df[col]
        dtype = "NUMERIC" if pd.api.types.is_numeric_dtype(col_series) else "STRING"
        example_values = []
        try:
            example_values = [str(x) for x in list(col_series.dropna().astype(str).unique()[:3])]
        except Exception:
            example_values = []
        schema.append({
            "name": col,
            "source_name": col,
            "type": dtype,
            "nullable": bool(col_series.isnull().any()),
            "null_pct": float(col_series.isnull().mean()) if len(col_series) > 0 else 1.0,
            "distinct": int(col_series.nunique(dropna=True)),
            "example_values": example_values,
            "description": ""
        })
    return schema

def ensure_sqlite_table(csv_path: str = CSV_PATH, db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> None:
    """
    Ensure the SQLite DB exists and has table `table_name` loaded from csv_path.
    If table exists, we will replace it to ensure fresh data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")

    # Connect and write to sqlite
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"Loaded CSV into SQLite table {table_name} ({len(df)} rows) at {db_path}")
    except Exception as e:
        logger.error(f"Failed to write CSV to SQLite: {e}")
        raise
    finally:
        conn.close()

def _sanitize_sql(sql: str) -> str:
    """
    Minor cleanup of sql string.
    """
    return sql.strip().rstrip(';')

def validate_sql_query(sql_query: str) -> str:
    """
    Validate SQL query to ensure it's a safe single SELECT statement suitable for SQLite.
    Raises ValueError on failure.
    """
    if not sql_query or not isinstance(sql_query, str):
        raise ValueError("SQL query must be a non-empty string")
    sql_query = _sanitize_sql(sql_query)

    # length
    if len(sql_query) > MAX_SQL_LENGTH:
        raise ValueError("SQL query exceeds maximum allowed length")

    # blocked patterns
    for patt in BLOCKED_SQL_PATTERNS:
        if re.search(patt, sql_query):
            raise ValueError("SQL query contains blocked operation: " + patt)

    # must start with SELECT
    if not re.match(r'(?is)^\s*SELECT\b', sql_query):
        raise ValueError("Only SELECT queries are allowed")

    # disallow multiple statements
    if ';' in sql_query:
        # allow if semicolon was trimmed; but if any remain, block
        raise ValueError("Multiple SQL statements are not allowed")

    # final safety
    return sql_query

def call_gpt_generate_sql(user_query: str, schema: Any, table_name: str = TABLE_NAME, model: str = "gpt-4o-mini") -> str:
    """
    Build prompt with schema and call OpenAI gpt-4o-mini to generate SQL.
    The model is instructed to return ONLY the SQL query, nothing else.
    """
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY in environment.")

    # Prepare schema snippet (stringified JSON but truncated if huge)
    try:
        schema_str = json.dumps(schema, indent=2) if schema else "[]"
    except Exception:
        schema_str = str(schema)

    # Safety: limit schema length in prompt (models don't need entire huge stats)
    if len(schema_str) > 10000:
        schema_str = schema_str[:10000] + "...(truncated)"

    system_instructions = f"""
You are a SQL generation assistant that MUST output a single valid SQLite SELECT statement only (no explanation, no markdown, no code fences).
The database table is named `{table_name}`.
You have this table schema (JSON list of column objects with 'name' and 'type'):
{schema_str}

Mapping rules (apply these priorities):
- If the user asks for "cost" and doesn't specify which cost field, use "EffectiveCost".
- If the user explicitly asks for billed cost (keywords: 'billedcost', 'billed cost', 'invoice cost'), use 'BilledCost'.
- If the user mentions 'list cost', use 'ListCost'; 'contracted cost' -> 'ContractedCost'.
- Default date column for time-series/trend analysis = 'ChargePeriodStart' unless user says 'billing period' or 'service period'.
- Table name is `{table_name}`. Use SQLite-compatible SQL (no LIMIT/OFFSET syntax differences).
- Use SUM() for cost aggregations.
- Use GROUP BY / ORDER BY as needed to answer the user's question.
- Do NOT use any DDL or DML statements (no CREATE/INSERT/UPDATE/DELETE/ALTER/DROP).
- Avoid window functions if the user didn't ask for them.
- Prefer explicit column names from schema. If user asks for 'service', map to 'ServiceName'.
- If user asks for 'top N', include 'LIMIT N'.
- If user asks for an aggregation over time, group by strftime('%Y-%m', <date_col>) for monthly aggregation or strftime with %Y-%m-%d for daily.
- Respond ONLY with the SQL query.
"""

    user_content = f"User request: {user_query}\n\nProduce only SQL."

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content}
    ]

    logger.debug("Calling OpenAI gpt-4o-mini to generate SQL.")
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        raise

    # Extract textual response - attempt to handle different client return types
    try:
        content = ""
        if hasattr(resp, "choices"):
            # modern response
            content = resp.choices[0].message.content if hasattr(resp.choices[0], "message") else resp.choices[0].text
        elif isinstance(resp, dict) and "choices" in resp:
            choice = resp["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            else:
                content = choice.get("text", "")
        else:
            content = str(resp)
    except Exception as e:
        logger.error(f"Failed to parse OpenAI response: {e}")
        raise

    content = content.strip()
    logger.debug("Raw SQL from model: %s", content[:1000])
    return content

def extract_sql_from_response(raw_sql: str) -> str:
    """
    Remove any markdown fences or surrounding text - naive approach.
    """
    sql = raw_sql
    # remove triple backtick blocks
    if "```" in sql:
        parts = sql.split("```")
        # try to find a block that contains 'select'
        for p in parts:
            if re.search(r'(?is)\bselect\b', p):
                sql = p
                break
        else:
            sql = parts[1] if len(parts) > 1 else parts[0]

    # remove any leading explanation lines
    # find first SELECT
    m = re.search(r'(?is)\bselect\b', sql)
    if m:
        sql = sql[m.start():]
    return sql.strip()

def execute_sql_and_save(sql: str, db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> Tuple[pd.DataFrame, str]:
    """
    Execute validated SQL in sqlite DB and save results to results/ with timestamped CSV.
    Returns (DataFrame, csv_path)
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise RuntimeError(f"SQL execution failed: {e}")
    finally:
        conn.close()

    ensure_results_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"sql_result_{ts}.csv"
    csv_path = os.path.join(RESULTS_DIR, filename)
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        logger.warning(f"Failed to save SQL result to CSV: {e}")
    logger.info(f"SQL executed — results saved to {csv_path} ({len(df)} rows)")
    return df, csv_path

def generate_sql_and_execute(
    user_query: str,
    csv_path: str = CSV_PATH,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    schema_context_override: Optional[Any] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    End-to-end: ensure DB loaded, infer or load schema, call LLM to generate SQL,
    validate SQL, execute it, and return structured result.

    Returns dict:
    {
      "sql": "<validated_sql>",
      "dataframe": <pd.DataFrame>,
      "csv_path": "<path to saved results csv>",
      "error": False or True,
      "error_message": str or None
    }
    """
    try:
        # Ensure CSV -> SQLite
        ensure_sqlite_table(csv_path=csv_path, db_path=db_path, table_name=table_name)

        # Load schema (preference: provided override > prompt hub > infer from CSV)
        if schema_context_override:
            schema = schema_context_override
        else:
            schema = load_schema_context()
            if not schema:
                schema = infer_schema_from_csv(csv_path)

        # Call LLM to generate SQL
        raw_sql = call_gpt_generate_sql(user_query=user_query, schema=schema, table_name=table_name, model=model)
        if not raw_sql or len(raw_sql.strip()) == 0:
            raise RuntimeError("LLM returned empty SQL")

        sql_candidate = extract_sql_from_response(raw_sql)

        # Validate
        validated_sql = validate_sql_query(sql_candidate)

        # Execute
        df, result_csv = execute_sql_and_save(validated_sql, db_path=db_path, table_name=table_name)

        return {
            "sql": validated_sql,
            "dataframe": df,
            "csv_path": result_csv,
            "error": False,
            "error_message": None
        }

    except Exception as exc:
        logger.exception("generate_sql_and_execute failed")
        return {
            "sql": None,
            "dataframe": pd.DataFrame(),
            "csv_path": None,
            "error": True,
            "error_message": str(exc)
        }

# Quick test function (safe to call in dev)
def _self_test():
    """
    Basic sanity test - generates SQL for a simple prompt (requires OPENAI_API_KEY set).
    """
    q = "Show total cost by ServiceName for the last 3 months, ordered by cost desc, top 10."
    res = generate_sql_and_execute(q)
    print("SQL:", res.get("sql"))
    print("Rows:", len(res.get("dataframe", [])))
    print("CSV:", res.get("csv_path"))
    if res.get("error"):
        print("Error:", res.get("error_message"))

if __name__ == "__main__":
    # run a quick smoke test
    try:
        _self_test()
    except Exception as e:
        logger.error(f"Self test failed: {e}")
