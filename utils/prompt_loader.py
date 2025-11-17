# utils/prompt_loader.py
import os

# Base directory for project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")

def load_prompt_from_hub(prompt_key: str, **interpolation_vars) -> str:
    """
    Loads a prompt text from the local 'prompts/' folder.

    Example:
        load_prompt_from_hub("supervisor")
        -> loads prompts/supervisor.txt

    You can also pass template variables like:
        load_prompt_from_hub("insight_agent", metric="cost", timeframe="Q1")
        if your prompt file contains placeholders like:
        "Analyze {metric} for {timeframe}"
    """
    # Support multiple possible extensions
    possible_files = [
        os.path.join(PROMPTS_DIR, f"{prompt_key}.txt"),
        os.path.join(PROMPTS_DIR, f"{prompt_key}.prompt"),
        os.path.join(PROMPTS_DIR, f"{prompt_key}.md"),
    ]

    # Locate file
    prompt_file = next((p for p in possible_files if os.path.exists(p)), None)
    if not prompt_file:
        raise FileNotFoundError(f"❌ Prompt file for '{prompt_key}' not found in {PROMPTS_DIR}")

    # Read file
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Optional variable substitution
    if interpolation_vars:
        try:
            prompt_text = prompt_text.format(**interpolation_vars)
        except KeyError as e:
            raise KeyError(f"Missing interpolation variable in prompt '{prompt_key}': {e}")

    return prompt_text


if __name__ == "__main__":
    # Test standalone usage
    key = "supervisor"
    try:
        prompt = load_prompt_from_hub(key)
        print(f"\n✅ Loaded prompt '{key}':\n{'-'*60}\n{prompt}")
    except Exception as e:
        print(f"❌ Error loading prompt: {e}")
