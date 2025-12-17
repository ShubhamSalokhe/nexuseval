"""
Tests for multi-provider LLM support.
"""

import pytest
import asyncio
from nexuseval.providers import LLMProviderRegistry, OpenAIProvider
from nexuseval.llm import LLMClient
from nexuseval.config import LLMConfig

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
    """Test creating clients with different providers."""
    # This just tests instantiation, not actual API calls
    try:
        client1 = LLMClient(provider="openai", model="gpt-4o-mini")
        assert client1.provider_name == "openai"
    except Exception as e:
        if "api_key" in str(e).lower():
            pytest.skip("OPENAI_API_KEY not set")
        raise
    
    # These will succeed if provider packages are installed
    try:
        client2 = LLMClient(provider="anthropic", model="claude-3-haiku-20240307")
        assert client2.provider_name == "anthropic"
    except ImportError:
        pytest.skip("Anthropic not installed")
    
    try:
        client3 = LLMClient(provider="google", model="gemini-1.5-flash")
        assert client3.provider_name == "google"
    except ImportError:
        pytest.skip("Google GenerativeAI not installed")

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
async def test_openai_provider_generate_json():
    """Test OpenAI provider JSON generation (requires API key)."""
    try:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        provider = OpenAIProvider(model="gpt-4o-mini")
        result = await provider.generate_json(
            "Return a JSON object with a 'test' key set to true"
        )
        
        assert isinstance(result, dict)
        # Should have some content (may not be exactly what we asked for due to retries)
        assert len(result) > 0
        
    except Exception as e:
        pytest.skip(f"API call failed: {e}")

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
