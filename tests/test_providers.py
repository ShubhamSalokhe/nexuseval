"""
Tests for multi-provider LLM support.
"""

import pytest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from nexuseval.providers import LLMProviderRegistry, OpenAIProvider
from nexuseval.llm import LLMClient
from nexuseval.config import LLMConfig


@pytest.fixture(autouse=True)
def provider_test_credentials(monkeypatch):
    # Constructor tests must run offline without using real account credentials.
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(variable, "test-api-key")

def test_provider_registry_openai():
    """Test that OpenAI provider is always available."""
    providers = LLMProviderRegistry.list_providers()
    assert "openai" in providers
    assert providers["openai"] is True

def test_provider_registry_create_openai():
    """Test creating OpenAI provider via registry (skips if no API key)."""
    try:
        provider = LLMProviderRegistry.create("openai", "gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise

def test_provider_registry_get_installation_instructions():
    """Test getting installation instructions."""
    instructions = LLMProviderRegistry.get_installation_instructions("anthropic")
    assert "pip install anthropic" in instructions
    
    instructions = LLMProviderRegistry.get_installation_instructions("google")
    assert "pip install google-generativeai" in instructions

def test_llm_client_default():
    """Test LLMClient with default OpenAI provider (skips if no API key)."""
    try:
        client = LLMClient()
        assert client.provider_name == "openai"
        assert client.model == "gpt-4-turbo"
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise

def test_llm_client_with_config():
    """Test LLMClient with LLMConfig (skips if no API key)."""
    try:
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.5
        )
        client = LLMClient(config=config)
        assert client.provider_name == "openai"
        assert client.model == "gpt-4o-mini"
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise

def test_llm_client_list_providers():
    """Test listing available providers."""
    providers = LLMClient.list_providers()
    assert isinstance(providers, dict)
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "groq" in providers
    assert "ollama" in providers

def test_llm_client_provider_switching():
    """Construct installed providers; unavailable providers give installation errors."""
    providers = LLMClient.list_providers()
    for name, model in [("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307"),
                        ("google", "gemini-1.5-flash"), ("groq", "llama-3.3-70b-versatile"),
                        ("ollama", "llama3")]:
        if providers[name]:
            client = LLMClient(provider=name, model=model)
            assert client.provider_name == name
        else:
            with pytest.raises(ImportError, match="not installed"):
                LLMClient(provider=name, model=model)

def test_llm_config_with_providers():
    """Test LLMConfig with different providers."""
    config1 = LLMConfig(provider="openai", model="gpt-4-turbo")
    assert config1.provider == "openai"
    
    config2 = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
    assert config2.provider == "anthropic"
    
    config3 = LLMConfig(provider="ollama", model="llama3", base_url="http://localhost:11434")
    assert config3.provider == "ollama"
    assert config3.base_url == "http://localhost:11434"

def test_provider_usage_tracking():
    """Test usage and cost tracking in providers (skips if no API key)."""
    try:
        provider = OpenAIProvider(model="gpt-4o-mini")
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise
    
    # Initially zero
    stats = provider.get_usage_stats()
    assert stats["total_cost_usd"] == 0.0
    assert stats["total_tokens"] == 0
    
    # Simulate usage
    provider.update_usage(input_tokens=100, output_tokens=50)
    stats = provider.get_usage_stats()
    
    assert stats["total_input_tokens"] == 100
    assert stats["total_output_tokens"] == 50
    assert stats["total_tokens"] == 150
    assert stats["total_cost_usd"] > 0.0  # Should have non-zero cost
    
    # Reset
    provider.reset_usage()
    stats = provider.get_usage_stats()
    assert stats["total_cost_usd"] == 0.0

@pytest.mark.asyncio
async def test_openai_provider_generate_json(monkeypatch):
    """Exercise JSON generation without a live API call or silently skipped errors."""
    provider = OpenAIProvider(model="gpt-4o-mini")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"test": true}'))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(provider.client.chat.completions, "create", request)
    assert await provider.generate_json("Return JSON") == {"test": True}
    assert provider.get_usage_stats()["total_tokens"] == 15

def test_llm_client_cost_stats():
    """Test cost statistics tracking (skips if no API key)."""
    try:
        client = LLMClient(enable_cost_tracking=True)
        stats = client.get_cost_stats()
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise
        
    # Continue with the rest of the test
    
    assert "total_cost_usd" in stats
    assert "total_tokens" in stats
    assert "model" in stats
    assert stats["model"] == "gpt-4-turbo"
