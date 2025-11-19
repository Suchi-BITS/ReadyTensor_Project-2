# integrations/main.py
import os, sys
import toml
import re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

from agents.supervisor import run_supervisor
from schema.state import init_state
from utils.logger_setup import setup_execution_logger

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

# Security and Validation Constants
MAX_QUERY_LENGTH = 1000
MAX_FILE_SIZE_MB = 100
ALLOWED_CSV_EXTENSIONS = ['.csv', '.CSV']
BLOCKED_PATTERNS = [
    r'(?i)(drop|delete|truncate|alter)\s+(table|database)',
    r'(?i)(exec|execute|eval|system|import\s+os)',
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'onerror\s*=',
    r'\.\./|\.\.',  # Path traversal
    r'file://',
    r'[;\|&`$]'  # Command injection characters
]

class ValidationError(Exception):
    pass

class SecurityError(Exception):
    pass

def validate_query(user_query: str) -> str:
    """
    Validate and sanitize user query input
    
    Args:
        user_query: Raw user input query
        
    Returns:
        Sanitized query string
        
    Raises:
        ValidationError: If query fails validation
        SecurityError: If query contains malicious patterns
    """
    # Check if query exists
    if not user_query:
        raise ValidationError("Query cannot be empty")
    
    # Check type
    if not isinstance(user_query, str):
        raise ValidationError(f"Query must be a string, got {type(user_query)}")
    
    # Strip whitespace
    user_query = user_query.strip()
    
    # Check length
    if len(user_query) < 3:
        raise ValidationError("Query is too short. Please provide at least 3 characters")
    
    if len(user_query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")
    
    # Security checks for malicious patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, user_query):
            logger.warning(f"Blocked malicious query pattern: {pattern}")
            raise SecurityError("Query contains potentially harmful content and has been blocked")
    
    # Remove control characters except newline and tab
    user_query = ''.join(char for char in user_query if ord(char) >= 32 or char in ['\n', '\t'])
    
    logger.info(f"Query validation passed: {user_query[:50]}...")
    return user_query

def validate_csv_path(csv_path: str) -> str:
    """
    Validate CSV file path for security and accessibility
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Validated absolute path
        
    Raises:
        ValidationError: If path fails validation
        SecurityError: If path contains security risks
    """
    # Check if path exists
    if not csv_path:
        raise ValidationError("CSV path cannot be empty")
    
    if not isinstance(csv_path, str):
        raise ValidationError(f"CSV path must be a string, got {type(csv_path)}")
    
    # Convert to Path object for safer handling
    try:
        file_path = Path(csv_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path format: {e}")
    
    # Check for path traversal attempts
    if '..' in csv_path:
        raise SecurityError("Path traversal detected and blocked")
    
    # Verify file exists
    if not file_path.exists():
        raise ValidationError(f"CSV file not found: {csv_path}")
    
    # Verify it is a file not a directory
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {csv_path}")
    
    # Check file extension
    if file_path.suffix not in ALLOWED_CSV_EXTENSIONS:
        raise ValidationError(f"Invalid file type. Only CSV files are allowed, got: {file_path.suffix}")
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB")
    
    # Check read permissions
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"Cannot read file: {csv_path}. Check permissions")
    
    logger.info(f"CSV validation passed: {file_path}")
    return str(file_path)

def validate_environment():
    """
    Validate required environment variables and configurations
    
    Raises:
        ValidationError: If required environment is not set up
    """
    required_env_vars = ['GROQ_API_KEY']  # Add other required vars
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValidationError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment validation passed")

def sanitize_result(result: dict) -> dict:
    """
    Sanitize and validate the result dictionary
    
    Args:
        result: Raw result from supervisor
        
    Returns:
        Sanitized result dictionary
    """
    if not isinstance(result, dict):
        logger.error(f"Result is not a dict: {type(result)}")
        return {
            "response": "System error: Invalid result format",
            "chart_path": None,
            "error": True
        }
    
    # Extract and sanitize response
    response = result.get("response")
    if not response or response == "None" or str(response).strip() == "":
        response = "No response generated. Please try rephrasing your question"
        logger.warning("Empty response detected, using fallback message")
    
    # Validate chart path if provided
    chart_path = result.get("chart_path")
    if chart_path:
        try:
            chart_path = str(chart_path)
            if not os.path.exists(chart_path):
                logger.warning(f"Chart path provided but file not found: {chart_path}")
                chart_path = None
        except Exception as e:
            logger.error(f"Error validating chart path: {e}")
            chart_path = None
    
    sanitized_result = {
        "response": str(response),
        "chart_path": chart_path,
        "error": False
    }
    
    return sanitized_result

def process_query(user_query: str, csv_path: str):
    """
    Main entry point with comprehensive validation and error handling
    
    Args:
        user_query: User's question or command
        csv_path: Path to the CSV data file
        
    Returns:
        Dictionary with response and optional chart_path
    """
    try:
        # Input validation
        logger.info("Starting query processing with validation")
        
        # Validate environment first
        validate_environment()
        
        # Validate and sanitize inputs
        user_query = validate_query(user_query)
        csv_path = validate_csv_path(csv_path)
        
        logger.info(f"Validated query: {user_query[:50]}...")
        logger.info(f"Validated CSV path: {csv_path}")
        
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
    
    # Process the query
    try:
        logger.info(f"Running workflow for query: {user_query}")
        
        # Initialize state
        state = init_state(user_query, session_id="streamlit-session")
        
        if not state or not isinstance(state, dict):
            raise ValueError("Failed to initialize state")
        
        logger.info(f"State initialized with keys: {list(state.keys())}")
        
        # Run supervisor
        result = run_supervisor(state, csv_path)
        
        # Validate and sanitize result
        final_result = sanitize_result(result)
        
        logger.info("Query processing completed successfully")
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
        logger.error(f"Unexpected error in process_query: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Full traceback: {error_trace}")
        
        return {
            "response": f"System Error: An unexpected error occurred. Please try again or contact support",
            "chart_path": None,
            "error": True
        }

# Health check function
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