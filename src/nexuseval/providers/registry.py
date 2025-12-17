"""
Provider registry for managing LLM providers.
"""

from typing import Dict, Type, Optional
from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider

# Lazy imports for optional providers
_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
}

# Provider availability flags
_ANTHROPIC_AVAILABLE = False
_GOOGLE_AVAILABLE = False
_GROQ_AVAILABLE = False
_OLLAMA_AVAILABLE = False

def _lazy_load_anthropic():
    """Lazy load Anthropic provider."""
    global _ANTHROPIC_AVAILABLE
    if not _ANTHROPIC_AVAILABLE:
        try:
            from .anthropic_provider import AnthropicProvider
            _PROVIDER_CLASSES["anthropic"] = AnthropicProvider
            _ANTHROPIC_AVAILABLE = True
        except ImportError:
            pass
    return _ANTHROPIC_AVAILABLE

def _lazy_load_google():
    """Lazy load Google provider."""
    global _GOOGLE_AVAILABLE
    if not _GOOGLE_AVAILABLE:
        try:
            from .google_provider import GoogleProvider
            _PROVIDER_CLASSES["google"] = GoogleProvider
            _GOOGLE_AVAILABLE = True
        except ImportError:
            pass
    return _GOOGLE_AVAILABLE

def _lazy_load_groq():
    """Lazy load Groq provider."""
    global _GROQ_AVAILABLE
    if not _GROQ_AVAILABLE:
        try:
            from .groq_provider import GroqProvider
            _PROVIDER_CLASSES["groq"] = GroqProvider
            _GROQ_AVAILABLE = True
        except ImportError:
            pass
    return _GROQ_AVAILABLE

def _lazy_load_ollama():
    """Lazy load Ollama provider."""
    global _OLLAMA_AVAILABLE
    if not _OLLAMA_AVAILABLE:
        try:
            from .ollama_provider import OllamaProvider
            _PROVIDER_CLASSES["ollama"] = OllamaProvider
            _OLLAMA_AVAILABLE = True
        except ImportError:
            pass
    return _OLLAMA_AVAILABLE

class LLMProviderRegistry:
    """
    Registry for LLM providers.
    
    Provides factory methods for creating provider instances and
    allows registration of custom providers.
    """
    
    _custom_providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """
        Register a custom provider.
        
        Args:
            name: Provider name (e.g., "my_custom_provider")
            provider_class: Provider class (must inherit from BaseLLMProvider)
        
        Example:
            >>> class MyProvider(BaseLLMProvider):
            ...     pass
            >>> LLMProviderRegistry.register("myprovider", MyProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(
                f"Provider class must inherit from BaseLLMProvider, "
                f"got {provider_class}"
            )
        cls._custom_providers[name] = provider_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseLLMProvider]:
        """
        Get provider class by name.
        
        Args:
            name: Provider name
        
        Returns:
            Provider class
        
        Raises:
            ValueError: If provider is not found
        """
        # Check custom providers first
        if name in cls._custom_providers:
            return cls._custom_providers[name]
        
        # Lazy load built-in providers
        if name == "anthropic":
            _lazy_load_anthropic()
        elif name == "google":
            _lazy_load_google()
        elif name == "groq":
            _lazy_load_groq()
        elif name == "ollama":
            _lazy_load_ollama()
        
        if name in _PROVIDER_CLASSES:
            return _PROVIDER_CLASSES[name]
        
        available = list(_PROVIDER_CLASSES.keys()) + list(cls._custom_providers.keys())
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available providers: {sorted(available)}. "
            f"Did you install the required package?"
        )
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Factory method to create provider instance.
        
        Args:
            provider: Provider name ("openai", "anthropic", "google", etc.)
            model: Model identifier
            **kwargs: Provider-specific configuration
        
        Returns:
            Initialized provider instance
        
        Example:
            >>> provider = LLMProviderRegistry.create(
            ...     "openai",
            ...     "gpt-4-turbo",
            ...     api_key="sk-..."
            ... )
        """
        provider_class = cls.get(provider)
        return provider_class(model=model, **kwargs)
    
    @classmethod
    def list_providers(cls) -> Dict[str, bool]:
        """
        List all available providers and their availability status.
        
        Returns:
            Dict mapping provider names to availability (True if installed)
        """
        providers = {
            "openai": True,  # Always available (required dependency)
            "anthropic": _lazy_load_anthropic(),
            "google": _lazy_load_google(),
            "groq": _lazy_load_groq(),
            "ollama": _lazy_load_ollama(),
        }
        
        # Add custom providers
        for name in cls._custom_providers:
            providers[name] = True
        
        return providers
    
    @classmethod
    def get_installation_instructions(cls, provider: str) -> str:
        """
        Get installation instructions for a provider.
        
        Args:
            provider: Provider name
        
        Returns:
            Installation instructions
        """
        instructions = {
            "openai": "Already installed (required dependency)",
            "anthropic": "pip install anthropic",
            "google": "pip install google-generativeai",
            "groq": "pip install groq",
            "ollama": "pip install aiohttp && ollama pull <model>",
        }
        
        return instructions.get(
            provider,
            f"Unknown provider: {provider}"
        )
