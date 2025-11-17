import os
from deepeval.prompt import Prompt
os.environ['DEEPEVAL_AI_API_KEY'] = "confident_us_SNhVIptQyNLjsfW4KqEWmxMMaXiLMaQOJgBTGN2O2d8="

prompt = Prompt(alias = "data_fetcher_v1")
prompt.pull(version="00.00.01")
print(prompt)
print(prompt._messages_template)