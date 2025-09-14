"""
Simple test script to verify LM Studio local API connection.
Loads environment variables, sends a short test prompt to the model
and prints the response. 
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct-v0.3")

client = OpenAI(base_url=BASE, api_key=KEY)

# Minimal test prompt
messages = [
    {"role": "user", "content": "Say hi in one short sentence."}
]

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=50,
    )
    print("RESPONSE:\n", resp.choices[0].message.content.strip())
except Exception as e:
    print("ERROR:", repr(e))
