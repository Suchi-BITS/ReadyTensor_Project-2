import logging

def setup_execution_logger(log_path="execution_trace.log"):
    logger = logging.getLogger("finops_trace")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(message)s")  # <--- CHANGE HERE
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger