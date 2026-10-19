"""
Google Gemini provider implementation.
"""

import os
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse, retry_provider_request

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

class GoogleProvider(BaseLLMProvider):
    """
    Google provider for Gemini models.
    
    Supports: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro
    
    Requires: pip install google-generativeai
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
        "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Google provider.
        
        Args:
            model: Gemini model name
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            max_retries: Number of retry attempts
            retry_delay: Initial delay between retries
            **kwargs: Additional configuration
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google GenerativeAI package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        super().__init__(model, **kwargs)
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @retry_provider_request
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        kwargs = {**self.config, **kwargs}
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        
        response = await self.client.generate_content_async(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": kwargs.get("timeout", 30)},
        )
        
        # Track usage (approximate)
        if response.text:
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response.text)
            self.update_usage(input_tokens, output_tokens)
        
        return response.text
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a JSON object, raising if all attempts fail."""
        kwargs = {**self.config, **kwargs}

        async def request():
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.0),
                max_output_tokens=kwargs.get("max_tokens", 1024),
                response_mime_type="application/json",
            )
            response = await self.client.generate_content_async(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": kwargs.get("timeout", 30)},
            )
            content = response.text
            self.update_usage(self.count_tokens(prompt), self.count_tokens(content))
            return content

        return await self._generate_json_with_retry(request)

    @retry_provider_request
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        kwargs = {**self.config, **kwargs}
        if not messages or messages[-1].role != "user":
            raise ValueError("Chat requires a final user message")
        system_text = "\n".join(msg.content for msg in messages if msg.role == "system")
        history = [
            {"role": "model" if msg.role == "assistant" else "user", "parts": [msg.content]}
            for msg in messages if msg.role != "system"
        ]
        if system_text:
            first_user = next((item for item in history if item["role"] == "user"), None)
            if first_user is not None:
                first_user["parts"][0] = system_text + "\n\n" + first_user["parts"][0]
        chat = self.client.start_chat(history=history[:-1])
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        response = await chat.send_message_async(
            history[-1]["parts"][0],
            generation_config=generation_config,
            request_options={"timeout": kwargs.get("timeout", 30)},
        )

        # Track usage (approximate)
        input_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        output_tokens = self.count_tokens(response.text)
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.update_usage(input_tokens, output_tokens)
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            cost=cost
        )
