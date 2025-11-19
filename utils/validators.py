import re
import os
from pathlib import Path

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
    r'\.\./|\.\.',
    r'file://',
    r'[;\|&`$]'
]

class ValidationError(Exception):
    pass

class SecurityError(Exception):
    pass

def validate_query(user_query: str) -> str:
    if not user_query:
        raise ValidationError("Query cannot be empty")
    
    if not isinstance(user_query, str):
        raise ValidationError(f"Query must be a string, got {type(user_query)}")
    
    user_query = user_query.strip()
    
    if len(user_query) < 3:
        raise ValidationError("Query is too short. Please provide at least 3 characters")
    
    if len(user_query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")
    
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, user_query):
            raise SecurityError("Query contains potentially harmful content and has been blocked")
    
    user_query = ''.join(char for char in user_query if ord(char) >= 32 or char in ['\n', '\t'])
    
    return user_query

def validate_csv_path(csv_path: str) -> str:
    if not csv_path:
        raise ValidationError("CSV path cannot be empty")
    
    if not isinstance(csv_path, str):
        raise ValidationError(f"CSV path must be a string, got {type(csv_path)}")
    
    try:
        file_path = Path(csv_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path format: {e}")
    
    if '..' in csv_path:
        raise SecurityError("Path traversal detected and blocked")
    
    if not file_path.exists():
        raise ValidationError(f"CSV file not found: {csv_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {csv_path}")
    
    if file_path.suffix not in ALLOWED_CSV_EXTENSIONS:
        raise ValidationError(f"Invalid file type. Only CSV files are allowed, got: {file_path.suffix}")
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB")
    
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"Cannot read file: {csv_path}. Check permissions")
    
    return str(file_path)

def validate_environment():
    required_env_vars = ['GROQ_API_KEY']
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValidationError(f"Missing required environment variables: {', '.join(missing_vars)}")

def sanitize_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return {
            "response": "System error: Invalid result format",
            "chart_path": None,
            "error": True
        }
    
    response = result.get("response")
    if not response or response == "None" or str(response).strip() == "":
        response = "No response generated. Please try rephrasing your question"
    
    chart_path = result.get("chart_path")
    if chart_path:
        try:
            chart_path = str(chart_path)
            if not os.path.exists(chart_path):
                chart_path = None
        except Exception:
            chart_path = None
    
    sanitized_result = {
        "response": str(response),
        "chart_path": chart_path,
        "error": False
    }
    
    return sanitized_result