"""
OpenAI provider implementation.
"""

import os
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from .base import BaseLLMProvider, LLMMessage, LLMResponse, retry_provider_request

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider for GPT models.
    
    Supports: GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            max_retries: Number of retry attempts
            retry_delay: Initial delay between retries
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)
        client_options = {key: kwargs[key] for key in ("base_url", "timeout") if kwargs.get(key) is not None}
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"), max_retries=0, **client_options
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @retry_provider_request
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        kwargs = {**self.config, **kwargs}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        
        # Track usage
        if hasattr(response, 'usage') and response.usage:
            self.update_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        
        return response.choices[0].message.content
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a JSON object, raising if all attempts fail."""
        kwargs = {**self.config, **kwargs}

        async def request():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1000),
                response_format={"type": "json_object"},
            )
            if response.usage:
                self.update_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        return await self._generate_json_with_retry(request)

    @retry_provider_request
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        kwargs = {**self.config, **kwargs}
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        
        usage_dict = None
        cost = None
        
        if hasattr(response, 'usage') and response.usage:
            usage_dict = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            self.update_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage=usage_dict,
            cost=cost
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (if available).
        Falls back to character-based estimation.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimate
            return super().count_tokens(text)
