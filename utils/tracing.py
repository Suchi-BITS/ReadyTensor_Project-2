import os
from typing import Optional, Dict, Any
from functools import wraps
import time
from datetime import datetime

# LangSmith tracing setup
def setup_langsmith_tracing():
    """
    Configure LangSmith tracing for the entire application
    """
    # Set environment variables for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # These should be set in .env or environment
    # os.environ["LANGCHAIN_API_KEY"] = "your_api_key_here"
    # os.environ["LANGCHAIN_PROJECT"] = "finops-agent-module3"
    
    # Verify configuration
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("WARNING: LANGCHAIN_API_KEY not set. Tracing will not work.")
        return False
    
    project_name = os.getenv("LANGCHAIN_PROJECT", "finops-agent-module3")
    print(f"LangSmith tracing enabled. Project: {project_name}")
    return True


def trace_agent(agent_name: str):
    """
    Decorator to trace individual agent executions
    
    Usage:
        @trace_agent("data_fetcher")
        def data_fetcher_node(state):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from langsmith import traceable
            
            @traceable(
                name=agent_name,
                run_type="chain",
                metadata={
                    "agent": agent_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            def traced_execution(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Add execution metadata to result if possible
                if isinstance(result, dict):
                    result["_trace_metadata"] = {
                        "agent": agent_name,
                        "execution_time": execution_time
                    }
                
                return result
            
            return traced_execution(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_llm_call(call_name: str):
    """
    Decorator to trace LLM API calls
    
    Usage:
        @trace_llm_call("groq_sql_generation")
        def generate_sql(prompt):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from langsmith import traceable
            
            @traceable(
                name=call_name,
                run_type="llm",
                metadata={
                    "call_type": "llm",
                    "timestamp": datetime.now().isoformat()
                }
            )
            def traced_llm_call(*args, **kwargs):
                return func(*args, **kwargs)
            
            return traced_llm_call(*args, **kwargs)
        
        return wrapper
    return decorator