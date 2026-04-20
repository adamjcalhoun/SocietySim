import aiohttp
import json
from .client import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        session = self._get_session()
        async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["message"]["content"]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
