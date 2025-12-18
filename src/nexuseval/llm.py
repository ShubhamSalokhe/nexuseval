"""
Enhanced LLM client with multi-provider support, caching, and cost tracking.

This module provides a unified interface for accessing multiple LLM providers
(OpenAI, Anthropic, Google, Groq, Ollama) with intelligent caching and cost tracking.
"""

import asyncio
from typing import Dict, Any, Optional
from .providers import LLMProviderRegistry, BaseLLMProvider
from .config import LLMConfig

# Import cache if available
try:
    from .cache import CacheManager, InMemoryCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Supports: OpenAI, Anthropic (Claude), Google (Gemini), Groq, Ollama
    
    Features:
    - Multi-provider support with seamless switching
    - Intelligent caching to reduce costs
    - Automatic cost tracking
    - Retry logic with exponential backoff
    """
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        provider: str = "openai",
        config: Optional[LLMConfig] = None,
        cache_manager: Optional['CacheManager'] = None,
        enable_cache: bool = True,
        enable_cost_tracking: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier (e.g., "gpt-4-turbo", "claude-3-5-sonnet-20241022")
            provider: Provider name ("openai", "anthropic", "google", "groq", "ollama")
            config: Optional LLMConfig object (overrides model/provider if provided)
            cache_manager: Optional cache manager
            enable_cache: Whether to use caching
            enable_cost_tracking: Whether to use cost tracking
            verbose: Whether to print detailed prompts and responses
            **kwargs: Additional provider-specific parameters
        """
        # Use config if provided
        if config:
            provider = config.provider
            model = config.model
            kwargs.update({
                "api_key": config.api_key,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "retry_delay": config.retry_delay,
            })
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Create provider instance
        self.provider: BaseLLMProvider = LLMProviderRegistry.create(
            provider=provider,
            model=model,
            **kwargs
        )
        
        self.model = model
        self.provider_name = provider
        self.enable_cost_tracking = enable_cost_tracking
        self.verbose = verbose
        
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
        Generate structured JSON response from LLM.
        
        This is the main method for evaluation tasks. It ensures JSON output
        and handles caching, retries, and cost tracking automatically.
        
        Args:
            prompt: The prompt to send (should request JSON format)
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
        
        Returns:
            Dict containing the parsed JSON response
        
        Example:
            >>> client = LLMClient()
            >>> result = await client.get_score(
            ...     "Evaluate this response and return JSON with score and reason..."
            ... )
            >>> print(result["score"])
        """
        # Generate cache key if caching enabled
        if self.enable_cache and self.cache_manager:
            cache_key = self.cache_manager.generate_key(
                prompt=prompt,
                model=self.model,
                provider=self.provider_name,
                **kwargs
            )
            
            # Try cache first
            cached_result = await self.cache_manager.backend.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Call provider
        if self.verbose:
            print(f"\n[NexusEval] 📤 Request to {self.provider_name}/{self.model}:\n{prompt}\n")
            
        result = await self.provider.generate_json(prompt, **kwargs)

        if self.verbose:
            import json
            print(f"\n[NexusEval] 📥 Response:\n{json.dumps(result, indent=2)}\n")
        
        # Cache result if caching enabled
        if self.enable_cache and self.cache_manager:
            await self.cache_manager.backend.set(cache_key, result)
        
        return result
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        return await self.provider.generate(prompt, **kwargs)
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost and usage statistics.
        
        Returns:
            Dict with cost and token information
        
        Example:
            >>> stats = client.get_cost_stats()
            >>> print(f"Total cost: ${stats['total_cost_usd']:.4f}")
            >>> print(f"Total tokens: {stats['total_tokens']:,}")
        """
        return self.provider.get_usage_stats()
    
    def reset_cost_stats(self):
        """Reset cost tracking statistics."""
        self.provider.reset_usage()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache hit rate and other statistics
        """
        if self.cache_manager:
            return self.cache_manager.backend.get_stats()
        return {"enabled": False}
    
    @staticmethod
    def list_providers() -> Dict[str, bool]:
        """
        List all available providers.
        
        Returns:
            Dict mapping provider names to availability status
        
        Example:
            >>> providers = LLMClient.list_providers()
            >>> for name, available in providers.items():
            ...     print(f"{name}: {'✓' if available else '✗ (not installed)'}")
        """
        return LLMProviderRegistry.list_providers()
    
    @staticmethod
    def get_installation_instructions(provider: str) -> str:
        """
        Get installation instructions for a provider.
        
        Args:
            provider: Provider name
        
        Returns:
            Installation instructions
        """
        return LLMProviderRegistry.get_installation_instructions(provider)
    
    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider_name}', model='{self.model}')"