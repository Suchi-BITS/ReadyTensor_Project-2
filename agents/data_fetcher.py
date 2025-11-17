# agents/data_fetcher.py
import os
import pandas as pd

# -------------------------------------------------------------------
# Local CSV Data Fetcher
# -------------------------------------------------------------------

class DataFetchResult:
    def __init__(self, dataframe_path=None, message=None, rows=0, columns=None):
        self.dataframe_path = dataframe_path
        self.message = message
        self.rows = rows
        self.columns = columns or []


def fetch_data(csv_path: str) -> DataFetchResult:
    """
    Reads the uploaded CSV file, validates it, and returns metadata.
    """
    if not os.path.exists(csv_path):
        return DataFetchResult(message=f" File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        rows, cols = df.shape
        columns = list(df.columns)

        # Save a cleaned copy for downstream use
        output_path = os.path.join("results", "data_cleaned.csv")
        os.makedirs("results", exist_ok=True)
        df.to_csv(output_path, index=False)

        message = f"Loaded {rows} rows and {len(columns)} columns from {os.path.basename(csv_path)}"
        print(message)

        return DataFetchResult(
            dataframe_path=output_path,
            message=message,
            rows=rows,
            columns=columns,
        )

    except Exception as e:
        return DataFetchResult(message=f" Failed to load CSV: {e}")


# -------------------------------------------------------------------
# Test script
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_csv = os.path.join("data", "sample_data.csv")
    result = fetch_data(sample_csv)
    print(f"\nMessage: {result.message}")
    print(f"Rows: {result.rows}")
    print(f"Columns: {result.columns}")
    print(f"Data saved at: {result.dataframe_path}")
