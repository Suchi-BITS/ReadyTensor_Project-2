import os
import sys
import toml
from typing import List, Dict, Any, Optional
import traceback


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

# Import from other modules (NOT from main)
from agents.supervisor import run_supervisor
from schema.state import init_state
from utils.logger_setup import setup_execution_logger
from utils.validators import (
    validate_query,
    validate_csv_path,
    validate_environment,
    sanitize_result,
    ValidationError,
    SecurityError
)
from utils.tracing import setup_langsmith_tracing
import concurrent.futures
from utils.resilience import retry_with_backoff

#from langchain.callbacks.tracers import LangChainTracer

# Initialize tracing when module loads
#setup_langsmith_tracing()
# Load secrets from .streamlit/secrets.toml
secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    try:
        with open(secrets_path, 'r') as f:
            secrets = toml.load(f)
            for key, value in secrets.items():
                os.environ[key] = str(value)
    except Exception as e:
        print(f"Warning: Could not load secrets file: {e}")

logger = setup_execution_logger()
MAX_TURNS = int(os.getenv("FINOPS_MAX_TURNS", 20))
SUPERVISOR_TIMEOUT = int(os.getenv("FINOPS_SUPERVISOR_TIMEOUT", 60))

def process_query(
    user_query: str,
    csv_path: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "default-session"
):
    """
    Main entry point with memory support and comprehensive validation
    
    Args:
        user_query: User's question or command
        csv_path: Path to the CSV data file
        conversation_history: Previous conversation turns for context
        session_id: Unique session identifier
        
    Returns:
        Dictionary with response and optional chart_path
    """
    try:
        # Input validation
        logger.info("Starting query processing with validation and memory")
        
        # Validate environment
        validate_environment()
        
        # Validate and sanitize inputs
        user_query = validate_query(user_query)
        csv_path = validate_csv_path(csv_path)
        
        # Validate conversation history
        if conversation_history is None:
            conversation_history = []
        
        if not isinstance(conversation_history, list):
            logger.warning("Invalid conversation_history type, resetting to empty list")
            conversation_history = []
        
        logger.info(f"Validated query: {user_query[:50]}...")
        logger.info(f"Validated CSV path: {csv_path}")
        logger.info(f"Conversation history: {len(conversation_history)} entries")
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "response": f"Validation Error: {str(e)}",
            "chart_path": None,
            "error": True
        }
    
    except SecurityError as e:
        logger.error(f"Security error: {e}")
        return {
            "response": f"Security Error: {str(e)}",
            "chart_path": None,
            "error": True
        }
    
    # Process the query with memory
    try:
        logger.info(f"Running workflow for query: {user_query}")
        
        # Initialize state with conversation history
        state = init_state(
            original_query=user_query,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        if not state or not isinstance(state, dict):
            raise ValueError("Failed to initialize state")
        
        logger.info(f"State initialized with keys: {list(state.keys())}")
        logger.info(f"Memory context length: {len(state.get('memory_context', ''))}")
        logger.info(f"Turn number: {state.get('turn_number', 0)}")
        
        # Turn limit guard
        if state.get("turn_number", 0) > MAX_TURNS:
            return {
                "response": "Conversation limit reached. Please start a new session.",
                "error": True
            }

        def _run_supervisor():
            return run_supervisor(state, csv_path)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: retry_with_backoff(
                        fn=_run_supervisor,
                        exceptions=(RuntimeError, TimeoutError),
                        max_retries=2,
                        base_delay=1.0
                    )
                )
                result = future.result(timeout=SUPERVISOR_TIMEOUT)

        except concurrent.futures.TimeoutError:
            logger.error("Supervisor execution timed out")
            return {
                "response": "Request timed out. Please try again.",
                "error": True
            }
        except Exception as e:
            logger.error(f"Supervisor execution error: {str(e)}", exc_info=True)
            error_trace = traceback.format_exc()
            logger.error(f"Full error traceback: {error_trace}")
            return {
                "response": f"System error: {str(e)}\n\nDetails: Check logs for full traceback.",
                "error": True,
                "error_details": str(e),
                "error_traceback": error_trace
            }
        
        # Validate and sanitize result
        final_result = sanitize_result(result)
        
        # Add memory metadata to result
        final_result["turn_number"] = state.get("turn_number", 0)
        final_result["intent"] = state.get("intent")
        final_result["subagent"] = state.get("subagent")
        
        logger.info("Query processing completed successfully with memory")
        return final_result
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return {
            "response": f"File Error: {str(e)}",
            "chart_path": None,
            "error": True
        }
    
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return {
            "response": f"Permission Error: Cannot access required files",
            "chart_path": None,
            "error": True
        }
    
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return {
            "response": "Memory Error: File too large or system resources exhausted",
            "chart_path": None,
            "error": True
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in process_query: {e}", exc_info=True)
        error_trace = traceback.format_exc()
        logger.error(f"Full traceback: {error_trace}")
        
        return {
            "response": f"System Error: {str(e)}. Please check logs for details.",
            "chart_path": None,
            "error": True,
            "error_details": str(e),
            "error_traceback": error_trace
        }


def health_check():
    """
    Verify system is properly configured
    
    Returns:
        Dictionary with health status
    """
    health_status = {
        "status": "healthy",
        "issues": []
    }
    
    try:
        validate_environment()
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["issues"].append(f"Environment: {str(e)}")
    
    return health_status
