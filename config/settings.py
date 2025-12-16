import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Environment
    ENV = os.getenv("ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # OpenAI / LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # LangSmith / LangChain
    LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING", "false").lower() == "true"
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

    # Storage & paths
    DATABASE_PATH = os.getenv("DATABASE_PATH", "finops_memory.db")
    FINOPS_SQLITE_DB = os.getenv("FINOPS_SQLITE_DB", "finops.db")
    FINOPS_CSV_PATH = os.getenv("FINOPS_CSV_PATH", "data/data.csv")
    FINOPS_TABLE_NAME = os.getenv("FINOPS_TABLE_NAME", "finops_data")
    FINOPS_RESULTS_DIR = os.getenv("FINOPS_RESULTS_DIR", "results")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

    # Execution limits (resilience)
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
    SUPERVISOR_TIMEOUT = int(os.getenv("SUPERVISOR_TIMEOUT", 60))
    MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", 20))
    MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 10))
