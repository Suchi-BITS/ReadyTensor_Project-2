import pandas as pd
import snowflake.connector

def run_snowflake_query(query, conn_params):
    try:
        conn = snowflake.connector.connect(**conn_params)
        cur = conn.cursor()
        cur.execute(query)
        if cur.description:  # SELECT query
            result = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pd.DataFrame(result, columns=columns)
        cur.close()
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

