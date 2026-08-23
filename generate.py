import json
import urllib.request
import urllib.error
from config import OPENAI_API_KEY,OPENAI_API_BASE,MODEL_NAME


def generate(query,context):
    messages=[
        {
            "role":"system",
            "content":"You are a cybersecurity and AI security expert. Answer questions using ONLY the provided context. If the context doesn't contain enough information, say so. Always cite your sources."
        },
        {
            "role":"user",
            "content":f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    payload=json.dumps({
        "model":MODEL_NAME,
        "messages":messages,
        "temperature":0.3,
        "max_tokens":1024
    }).encode()

    req=urllib.request.Request(
        f"{OPENAI_API_BASE}/chat/completions",
        data=payload,
        headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {OPENAI_API_KEY}"
        }
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data=json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"LLM error: {e}"


if __name__=="__main__":
    answer=generate("What is prompt injection?","Prompt injection is an attack where malicious input tricks an LLM into ignoring its instructions.")
    print(answer)
