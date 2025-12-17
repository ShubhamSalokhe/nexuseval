"""
Groq provider implementation.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class GroqProvider(BaseLLMProvider):
    """
    Groq provider for ultra-fast inference.
    
    Supports: Llama 3.3, Llama 3.2, Mixtral, Gemma
    
    Requires: pip install groq
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
        "llama-3.2-11b-vision-preview": {"input": 0.18, "output": 0.18},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    }
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Groq provider.
        
        Args:
            model: Groq model name
            api_key: Groq API key (uses GROQ_API_KEY env var if not provided)
            max_retries: Number of retry attempts
            retry_delay: Initial delay between retries
            **kwargs: Additional configuration
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq package not installed. "
                "Install with: pip install groq"
            )
        
        super().__init__(model, **kwargs)
        self.client = AsyncGroq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        
        # Track usage
        if hasattr(response, 'usage') and response.usage:
            self.update_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        
        return response.choices[0].message.content
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output."""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Track usage
                if hasattr(response, 'usage') and response.usage:
                    self.update_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                
                return result
                
            except json.JSONDecodeError as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Invalid JSON response: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Groq Error: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {"score": 0.0, "reason": "Max retries exceeded"}
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
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
