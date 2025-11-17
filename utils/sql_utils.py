import re
import pandas as pd
from decimal import Decimal
import sqlglot
import snowflake.connector
import time
import os
import uuid
import sys
import csv
import gzip
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config.snowflake_config import get_snowflake_config
db_config = get_snowflake_config()
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from utils.logger_setup import setup_execution_logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from snowflake.sqlalchemy import URL
logger = setup_execution_logger()

class ExecutionResult(BaseModel):
    status: str = Field(..., description="Execution status, e.g., 'success' or 'error'")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Time taken to execute the query in seconds")
    row_count: int = Field(..., description="Number of rows returned by the query")
    execution_result: List[dict] = Field(..., description="List of rows returned by the query")

def clean_sql_response(llm_response_text):
    """Cleans the raw LLM SQL response and extracts SQL and columns, removing backslashes."""
    try:
        sql_match = re.search(r"```sql(.*?)```", llm_response_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql_text = sql_match.group(1).strip()
        else:
            sql_match = re.search(r"(SELECT.*?;)", llm_response_text, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_text = sql_match.group(1).strip()
            else:
                sql_text = llm_response_text.strip()
        sql_text = sql_text.replace("\\", "")
        print(f'SQL Code after cleaned : {sql_text}')
        return sql_text
    except Exception as e:
        return llm_response_text


def validate_and_transpile_sql(sql_query: str, target_dialect: str = "snowflake") -> dict:
    """
    Validates SQL query. For Snowflake with JSON syntax, skips transpilation.
    """
    try:
        # For Snowflake with JSON bracket notation, just validate and return original
        if target_dialect.lower() == "snowflake" and '":"' in sql_query:
            try:
                # Validate syntax by parsing
                parsed = sqlglot.parse_one(sql_query, dialect="snowflake")
                if parsed:
                    
                    return {
                        "transpiled_sql": sql_query.strip(),  # Return original SQL
                        "sql_valid": True,
                        "sql_validation_error": None
                    }
            except Exception as e:
                return {
                    "transpiled_sql": None,
                    "sql_valid": False,
                    "sql_validation_error": f"SQL validation failed: {str(e)}"
                }
        
        # Normal transpilation for other cases
        parsed = sqlglot.parse_one(sql_query, dialect=target_dialect)
        if parsed is None:
            return {
                "transpiled_sql": None,
                "sql_valid": False,
                "sql_validation_error": "Failed to parse SQL query"
            }
        
        transpiled_sql = parsed.sql(dialect=target_dialect, pretty=True)
        
        return {
            "transpiled_sql": transpiled_sql,
            "sql_valid": True,
            "sql_validation_error": None
        }
        
    except Exception as e:
        return {
            "transpiled_sql": None,
            "sql_valid": False,
            "sql_validation_error": f"SQL validation failed: {str(e)}"
        }

def stream_snowflake_query_to_csv(
    sql_query: str,
    output_path: str,
    chunk_size: int = 100000,
    overwrite: bool = True,
    include_header: bool = True,
    gzip_compress: bool = False,
    max_rows: Optional[int] = None,
    network_timeout: int = 200,sample_size = 5
) -> dict:
    """
    Incrementally stream a large SELECT result to a local CSV (optionally .gz) without loading all rows into memory.
    
    Args:
        sql_query: Snowflake SELECT statement.
        output_path: Destination file path (.csv or .csv.gz). If endswith .gz, gzip_compress auto-enabled.
        chunk_size: Rows fetched per round trip (tune 20k–100k).
        overwrite: If True, remove existing file first.
        include_header: Write header row once.
        gzip_compress: Force gzip (ignored if output_path endswith .gz which auto-enables it).
        max_rows: Optional cap (stop after writing this many rows).
        network_timeout: Snowflake network timeout seconds (raise for big exports).
    Returns:
        Dict with status, rows written, seconds elapsed, and file info.
    """
    if output_path.endswith(".gz"):
        gzip_compress = True

    t0 = time.time()
    written = 0
    engine = None
    sample_rows = []
    try:
        # Prep filesystem
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if overwrite and os.path.exists(output_path):
            os.remove(output_path)

        connection_string = URL(
            user=db_config["user"],
            private_key=db_config["private_key"],
            account=db_config["account"],
            warehouse=db_config["warehouse"],
            database=db_config["database"],
            schema=db_config["schema"]
        )
        engine = create_engine(
            connection_string,
            poolclass=NullPool,
            connect_args={
                "client_session_keep_alive": True,
                "network_timeout": network_timeout,
                "login_timeout": 120,
                "autocommit": False,
                "numpy": True
            }
        )
        logger.info(f"[stream] Start query export → {output_path}")
        raw_conn = engine.raw_connection()
        try:
            sf_conn = raw_conn.connection
            cur = sf_conn.cursor()
            try:
                cur.execute(f'USE WAREHOUSE {db_config["warehouse"]}')
                cur.execute(sql_query)
                col_names = [c[0] for c in cur.description]

                open_fn = gzip.open if gzip_compress else open
                mode = "wt"  # always text write (overwrite already handled)
                encoding = None if gzip_compress else "utf-8"

                with open_fn(output_path, mode=mode, newline="", encoding=encoding) as fh:
                    writer = csv.writer(fh)
                    if include_header:
                        writer.writerow(col_names)

                    cur.arraysize = chunk_size
                    while True:
                        batch = cur.fetchmany(chunk_size)
                        if not batch:
                            break
                        writer.writerows(batch)
                        if len(sample_rows) < sample_size:
                            need = sample_size - len(sample_rows)
                            for row in batch[:need]:
                                sample_rows.append({col: val for col, val in zip(col_names, row)})
                        
                        written += len(batch)

                        if written and written % (chunk_size * 2) == 0:
                            logger.info(f"[stream] Progress: {written} rows")

                        if max_rows and written >= max_rows:
                            break
            finally:
                cur.close()

            elapsed = time.time() - t0
            logger.info(f"[stream] Completed {written} rows in {elapsed:.2f}s → {output_path}")
            return {
                "status": "success",
                "rows": written,
                "seconds": elapsed,
                "output_path": output_path,
                "sample_rows" : sample_rows,
                "error" : None
            }
        except Exception as e:
            logger.error(f"[stream] Failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "rows": written,
                "output_path": output_path
            }
    finally:
        if engine:
            engine.dispose()
if __name__ == "__main__":
    # sql = """SELECT "CHARGEPERIODSTART" AS "USAGESTARTDATE", "CHARGEPERIODEND" AS "USAGEENDDATE", CAST("CHARGEPERIODSTART" AS DATE) AS "USAGEDATE", "BILLINGACCOUNTID" AS "INVOICEPAYERACCOUNTID", "BILLINGACCOUNTNAME" AS "INVOICEPAYERACCOUNTALIAS", "RESOURCEID" AS "LINEITEMRESOURCEID", "SERVICENAME" AS "PRODUCTSERVICE", "CHARGEDESCRIPTION" AS "USAGETYPE", "EFFECTIVECOST" AS "COST", "BILLINGCURRENCY" AS "CURRENCY", "CHARGECATEGORY" AS "CHARGETYPE", "SUBACCOUNTID" AS "ACCOUNTID", "INVOICEISSUERNAME" AS "BILLINGENTITY", "REGIONID" AS "REGION" FROM azure_fulldata WHERE "CHARGEPERIODSTART" >= DATE_TRUNC('month', '2025-04-01'::DATE) AND "CHARGEPERIODSTART" < DATE_TRUNC('month', '2025-06-01'::DATE) ORDER BY "USAGEDATE" ASC;"""
    sql = """SELECT "BILLINGANTID", "BILLINGCURRENCY", "COMMITMENTDISCOUNTID", "COMMITMENTDISCOUNTTYPE", "BILLEDCOST", "BILLINGPERIODSTART", "CHARGEPERIODSTART", "CHARGECATEGORY", "CHARGEDESCRIPTION" FROM azure_fulldata WHERE "BILLINGPERIODSTART" >= DATE_TRUNC('month', '2025-06-01'::DATE) AND "BILLINGPERIODSTART" < DATE_TRUNC('month', '2025-06-01'::DATE + INTERVAL '1 month') AND LOWER("COMMITMENTDISCOUNTTYPE") IN ('savings plan','reserved instance') AND "COMMITMENTDISCOUNTID" IS NOT NULL AND LOWER("CHARGECATEGORY") LIKE '%purchase%' ORDER BY "COMMITMENTDISCOUNTTYPE", "BILLINGPERIODSTART";"""
    # sql = """SELECT TO_CHAR(DATE_TRUNC('MONTH', CAST("CHARGEPERIODSTART" AS DATE)), 'YYYY-MM') AS "INVOICEMONTH", CAST("CHARGEPERIODSTART" AS DATE) AS "USAGEDATE", "BILLINGACCOUNTID", "BILLINGCURRENCY", "SERVICENAME" AS "PRODUCTSERVICE", "RESOURCEID", "TAGS", SUM("EFFECTIVECOST") AS "DAILYCOST" FROM azure_fulldata WHERE (("CHARGEPERIODSTART" >= '2025-03-01'::DATE AND "CHARGEPERIODSTART" < '2025-05-01'::DATE) OR ("CHARGEPERIODSTART" >= '2025-05-01'::DATE AND "CHARGEPERIODSTART" < '2025-05-16'::DATE) OR ("CHARGEPERIODSTART" >= '2025-06-01'::DATE AND "CHARGEPERIODSTART" < '2025-06-16'::DATE)) GROUP BY TO_CHAR(DATE_TRUNC('MONTH', CAST("CHARGEPERIODSTART" AS DATE)), 'YYYY-MM'), CAST("CHARGEPERIODSTART" AS DATE), "BILLINGACCOUNTID", "BILLINGCURRENCY", "SERVICENAME", "RESOURCEID", "TAGS" ORDER BY "USAGEDATE", "BILLINGACCOUNTID", "PRODUCTSERVICE", "RESOURCEID";"""
    # timed = time.time()
    # response,df = execute_sql_and_format_output_snowflake(sql)
    # end_time = time.time()
    response = stream_snowflake_query_to_csv(sql,"test_stream_second.csv")
    print(response)
    # print(f'Total time : {end_time - timed} seconds')
    # print(df.head())
    print("======")
    # print(df.tail())