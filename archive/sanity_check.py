import os, requests

r = requests.post(
    "https://router.huggingface.co/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json"
    },
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Dis bonjour en français."}],
        "max_tokens": 30
    }
)

print(r.status_code)
print(r.text)