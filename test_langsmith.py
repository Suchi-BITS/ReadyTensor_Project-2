import requests
import os

resp = requests.get(
    "https://api.smith.langchain.com/info",
    headers={"x-api-key": os.getenv("LANGCHAIN_API_KEY")},
)

print(resp.status_code, resp.text)

from langsmith import Client
c = Client()
print("Your API Key Org:", c.api_url)
