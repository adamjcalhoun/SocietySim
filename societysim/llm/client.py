from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Return the model's text response."""
        ...

    async def close(self):
        pass
