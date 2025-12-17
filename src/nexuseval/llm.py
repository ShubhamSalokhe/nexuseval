import os
import json
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any, Optional
import time

# Import cache if available
try:
    from .cache import CacheManager, InMemoryCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

class LLMClient:
    """
    Enhanced LLM client with caching, cost tracking, and retry logic.
    """
    
    # Pricing per 1K tokens (approximate, update as needed)
    PRICING = {
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        cache_manager: Optional['CacheManager'] = None,
        enable_cache: bool = True,
        enable_cost_tracking: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            model: OpenAI model name
            cache_manager: Optional cache manager (creates default if None and caching enabled)
            enable_cache: Whether to use caching
            enable_cost_tracking: Whether to track API costs
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Caching
        self.enable_cache = enable_cache and CACHE_AVAILABLE
        if self.enable_cache:
            if cache_manager is None:
                # Create default in-memory cache
                from .cache import CacheManager, InMemoryCache
                self.cache_manager = CacheManager(InMemoryCache())
            else:
                self.cache_manager = cache_manager
        else:
            self.cache_manager = None

    async def get_score(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and enforces a JSON response.
        Includes retry logic, caching, and cost tracking.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Dict containing the LLM response (parsed JSON)
        """
        # Generate cache key if caching enabled
        if self.enable_cache and self.cache_manager:
            cache_key = self.cache_manager.generate_key(
                prompt=prompt,
                model=self.model,
                **kwargs
            )
            
            # Try cache first
            cached_result = await self.cache_manager.backend.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Call LLM with retry logic
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1000),
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Track costs if enabled
                if self.enable_cost_tracking and hasattr(response, 'usage'):
                    cost = self._calculate_cost(response.usage)
                    self.total_cost += cost
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                # Cache result if caching enabled
                if self.enable_cache and self.cache_manager:
                    await self.cache_manager.backend.set(cache_key, result)
                
                return result
                
            except json.JSONDecodeError as e:
                # LLM returned invalid JSON
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Invalid JSON response: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                # API error or other exception
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"LLM Error: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {"score": 0.0, "reason": "Max retries exceeded"}
    
    def _calculate_cost(self, usage) -> float:
        """
        Calculate cost based on token usage.
        
        Args:
            usage: OpenAI usage object
        
        Returns:
            Cost in USD
        """
        if self.model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[self.model]
        input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost tracking statistics.
        
        Returns:
            Dict with cost and token statistics
        """
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "model": self.model
        }
    
    def reset_cost_stats(self):
        """Reset cost tracking statistics."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0