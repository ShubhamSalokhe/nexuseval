"""
Anthropic provider implementation.
"""

import os
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse, retry_provider_request

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider for Claude models.
    
    Supports: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
    
    Requires: pip install anthropic
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Claude model name
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            max_retries: Number of retry attempts
            retry_delay: Initial delay between retries
            **kwargs: Additional configuration
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        
        super().__init__(model, **kwargs)
        client_options = {key: kwargs[key] for key in ("base_url", "timeout") if kwargs.get(key) is not None}
        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), max_retries=0, **client_options
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @retry_provider_request
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        kwargs = {**self.config, **kwargs}
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            self.update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
        
        return response.content[0].text
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a JSON object, raising if all attempts fail."""
        kwargs = {**self.config, **kwargs}

        async def request():
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                system="Respond with valid JSON only, without explanatory text or markdown.",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
            )
            if response.usage:
                self.update_usage(response.usage.input_tokens, response.usage.output_tokens)
            return response.content[0].text

        return await self._generate_json_with_retry(request)

    @retry_provider_request
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        kwargs = {**self.config, **kwargs}
        # Claude requires alternating user/assistant messages
        formatted_messages = []
        system_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        system_options = {"system": "\n".join(system_messages)} if system_messages else {}
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            **system_options,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", 0.0),
        )
        
        usage_dict = None
        cost = None
        
        if hasattr(response, 'usage'):
            usage_dict = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            cost = self.calculate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            self.update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage=usage_dict,
            cost=cost
        )
