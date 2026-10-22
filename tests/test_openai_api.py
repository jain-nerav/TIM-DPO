import os
from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), (
    "OPENAI_API_KEY is not set."
)

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Reply with exactly: API test successful.",
)

print(response.output_text)