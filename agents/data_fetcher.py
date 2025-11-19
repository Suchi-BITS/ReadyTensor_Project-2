import os
import pandas as pd
from typing import Dict, Any, List, Optional

class DataFetchResult:
    def __init__(self, dataframe_path=None, message=None, rows=0, columns=None):
        self.dataframe_path = dataframe_path
        self.message = message
        self.rows = rows
        self.columns = columns or []


def fetch_data(
    csv_path: str,
    memory_context: Optional[str] = None,
    remembered_entities: Optional[Dict[str, Any]] = None
) -> DataFetchResult:
    """
    Reads the uploaded CSV file with memory awareness
    
    Args:
        csv_path: Path to CSV file
        memory_context: Previous conversation context
        remembered_entities: Previously mentioned entities (filters, columns, etc.)
    
    Returns:
        DataFetchResult with metadata
    """
    if not os.path.exists(csv_path):
        return DataFetchResult(message=f"File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        rows, cols = df.shape
        columns = list(df.columns)
        
        # Apply remembered filters if available
        if remembered_entities and remembered_entities.get("last_filters"):
            filters = remembered_entities["last_filters"]
            message_parts = [f"Loaded {rows} rows and {len(columns)} columns"]
            
            # Check if previously mentioned columns exist
            if filters.get("columns"):
                available_cols = [c for c in filters["columns"] if c in columns]
                if available_cols:
                    message_parts.append(f"Found previously mentioned columns: {', '.join(available_cols)}")
            
            message = " | ".join(message_parts)
        else:
            message = f"Loaded {rows} rows and {len(columns)} columns from {os.path.basename(csv_path)}"

        # Save cleaned copy
        output_path = os.path.join("results", "data_cleaned.csv")
        os.makedirs("results", exist_ok=True)
        df.to_csv(output_path, index=False)

        print(message)

        return DataFetchResult(
            dataframe_path=output_path,
            message=message,
            rows=rows,
            columns=columns,
        )

    except Exception as e:
        return DataFetchResult(message=f"Failed to load CSV: {e}")
