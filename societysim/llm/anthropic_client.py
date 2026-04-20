import anthropic
from .client import LLMClient

# Pin to specific version for reproducibility (per planning doc)
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 64):
        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> str:
        msg = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    async def close(self):
        await self._client.close()
