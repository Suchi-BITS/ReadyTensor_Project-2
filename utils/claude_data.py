import anthropic

client = anthropic.Anthropic(api_key="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")

response = client.messages.create(
    model="claude-Sonnet-3.7",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1000
)
print(response.content)