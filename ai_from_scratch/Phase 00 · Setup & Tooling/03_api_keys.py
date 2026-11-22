import os

import anthropic

client = anthropic.Anthropic()
Model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-3.5")

response = client.messages.create(
    model=Model,
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.content[0])
