import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct-v0.3")

client = OpenAI(base_url=BASE, api_key=KEY)

messages = [
    {"role": "user", "content": "You are a helpful assistant. Be concise. Say hi in one short sentence."}
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
