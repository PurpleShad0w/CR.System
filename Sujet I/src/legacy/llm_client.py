# llm_client.py (FINAL – LLaMA approved)

import os
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]

class HuggingFaceChatClient:
    def __init__(self):
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("HF_TOKEN not set")

        self.model = os.environ.get(
            "HF_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct"
        )

        self.url = "https://router.huggingface.co/v1/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }

        r = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=180
        )

        if r.status_code >= 400:
            raise RuntimeError(
                f"HF error {r.status_code}\n"
                f"Model={self.model}\n"
                f"Response={r.text[:2000]}"
            )

        raw = r.json()
        text = raw["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=raw)

def make_client():
    return HuggingFaceChatClient()