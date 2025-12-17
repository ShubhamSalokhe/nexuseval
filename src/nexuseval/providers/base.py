"""
Base abstraction for LLM providers.

This module defines the abstract interface that all LLM providers must implement,
inspired by LangChain's provider pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class LLMMessage(BaseModel):
    """Standard message format across all providers."""
    role: Literal["system", "user", "assistant"]
    content: str

class LLMResponse(BaseModel):
    """Standardized response from any LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # {"input_tokens": int, "output_tokens": int}
    cost: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    # Pricing per 1M tokens (to be overridden by subclasses)
    PRICING: Dict[str, Dict[str, float]] = {}
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the provider.
        
        Args:
            model: Model identifier/name
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.config = kwargs
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Input prompt (should request JSON output)
            **kwargs: Generation parameters
        
        Returns:
            Parsed JSON dict
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Default implementation uses rough estimate (4 chars per token).
        Subclasses should override with model-specific tokenizer if available.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        return len(text) // 4
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD
        """
        if self.model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[self.model]
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        
        return input_cost + output_cost
    
    def update_usage(self, input_tokens: int, output_tokens: int):
        """
        Update usage statistics.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.total_cost += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with usage and cost information
        """
        return {
            "model": self.model,
            "provider": self.__class__.__name__.replace("Provider", "").lower(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6)
        }
    
    def reset_usage(self):
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of conversation messages
            **kwargs: Generation parameters
        
        Returns:
            LLMResponse object
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
