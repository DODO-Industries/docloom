import requests

from module_AI.config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL, LM_STUDIO_TIMEOUT


class LMStudioClient:
    """Thin client for a local LM Studio server's OpenAI-compatible chat API."""

    def __init__(self, base_url: str = None, model: str = None, timeout: float = None):
        self.base_url = (base_url or LM_STUDIO_BASE_URL).rstrip("/")
        self.model = model or LM_STUDIO_MODEL
        self.timeout = timeout or LM_STUDIO_TIMEOUT

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def is_reachable(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False
