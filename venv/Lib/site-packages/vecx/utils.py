import re

def is_valid_index_name(index_name):
    # Regular expression for alphanumeric characters and underscores
    pattern = re.compile(r'^[a-zA-Z0-9_]+$')
    if pattern.match(index_name) and len(index_name) <= 48:
        return True
    else:
        return False