# ============================================================
# FILE 1: integrations/main.py
# ============================================================
import os, sys
import toml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

from agents.supervisor import run_supervisor
from schema.state import init_state
from utils.logger_setup import setup_execution_logger

# Load secrets from .streamlit/secrets.toml
secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, 'r') as f:
        secrets = toml.load(f)
        for key, value in secrets.items():
            os.environ[key] = str(value)

logger = setup_execution_logger()

def process_query(user_query: str, csv_path: str):
    if not os.path.exists(csv_path):
        return {"response": "❌ CSV file not found", "chart_path": None}

    try:
        logger.info(f"[Supervisor] Running workflow for query: {user_query}")
        print(f"[DEBUG] process_query received user_query = {user_query}")
        state = init_state(user_query, session_id="streamlit-session")
        print(f"[DEBUG] State after init_state: {state.keys()}")
        print(f"[DEBUG] State original_query: {state.get('original_query')}")
        print(f"[MAIN DEBUG] Calling run_supervisor with query: {user_query}")
        result = run_supervisor(state, csv_path)
        
        print(f"[MAIN DEBUG] Received result type: {type(result)}")
        print(f"[MAIN DEBUG] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        print(f"[MAIN DEBUG] Response value: {result.get('response') if isinstance(result, dict) else 'N/A'}")
        print(f"[MAIN DEBUG] Response type: {type(result.get('response')) if isinstance(result, dict) else 'N/A'}")
        
        # Ensure result is a proper dict with both keys
        if not isinstance(result, dict):
            print(f"[MAIN ERROR] Result is not a dict: {result}")
            return {"response": f"⚠️ Invalid result type: {type(result)}", "chart_path": None}
        
        # Extract response and chart_path with fallbacks
        response = result.get("response")
        chart_path = result.get("chart_path")
        
        # Handle None or empty response
        if not response or response == "None":
            response = "⚠️ Pipeline completed but no response was generated."
            print(f"[MAIN WARNING] Response was None or empty, using fallback")
        
        final_result = {
            "response": str(response),  # Force to string
            "chart_path": chart_path
        }
        
        print(f"[MAIN DEBUG] Returning final result: response length={len(final_result['response'])}, chart={final_result['chart_path']}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        import traceback
        traceback.print_exc()
        return {"response": f"⚠️ Error running supervisor: {e}", "chart_path": None}


'''if __name__ == "__main__":
    query = "What is the total cost?"
    csv_path = os.path.join("data", "sample_data.csv")
    result = process_query(query, csv_path)
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print(f"Response: {result['response']}")
    print(f"Chart: {result['chart_path']}")
    print("="*60)'''