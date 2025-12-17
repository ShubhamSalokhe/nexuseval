"""
Anthropic provider implementation.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse

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
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
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
        """
        Generate structured JSON output.
        
        Note: Claude doesn't have native JSON mode, so we use prompt engineering.
        """
        system_prompt = (
            "You must respond with valid JSON only. "
            "Do not include any explanatory text before or after the JSON. "
            "Your entire response must be parseable as JSON."
        )
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get("max_tokens", 1024),
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                )
                
                content = response.content[0].text
                
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                # Track usage
                if hasattr(response, 'usage'):
                    self.update_usage(
                        response.usage.input_tokens,
                        response.usage.output_tokens
                    )
                
                return result
                
            except json.JSONDecodeError as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Invalid JSON response: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Anthropic Error: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {"score": 0.0, "reason": "Max retries exceeded"}
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
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
        
        system_prompt = " ".join(system_messages) if system_messages else None
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            system=system_prompt,
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
