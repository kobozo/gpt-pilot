import datetime
from typing import Optional

from ollama import Client
from core.config import LLMProvider
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)


class OllamaClient(BaseLLMClient):
    provider = LLMProvider.OLLAMA

    def _init_client(self):
        self.client = Client(
            host=self.config.base_url,
            timeout=self.config.connect_timeout
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        completion_kwargs = {
            "model": self.config.model,
            "messages": convo.messages,
            "stream": True,
            "format": "json",
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
                "num_ctx": 131072
            }
        }
        
        stream = self.client.chat(**completion_kwargs)
        response = []
        prompt_tokens = 0  # Ollama doesn't expose token count directly yet.
        completion_tokens = 0  # Token counts might be calculated separately.

        for chunk in stream:
            message = chunk.get("message", {}).get("content", "")
            if message:
                response.append(message)
                if self.stream_handler:
                    await self.stream_handler(message)

        
        response_str = "".join(response)
        
        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        return response_str, prompt_tokens, completion_tokens

__all__ = ["OllamaClient"]