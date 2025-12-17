"""
Unit tests for configuration management.
"""

import pytest
import json
from pathlib import Path
from nexuseval.config import (
    NexusConfig,
    LLMConfig,
    CacheConfig,
    EvaluationConfig,
    ReportingConfig
)


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.0
        assert config.timeout == 30
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.7,
            max_tokens=2000
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_values(self):
        """Test default cache config."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.max_size == 1000
    
    def test_file_backend(self):
        """Test file backend configuration."""
        config = CacheConfig(
            backend="file",
            cache_dir=".custom_cache"
        )
        assert config.backend == "file"
        assert config.cache_dir == ".custom_cache"


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_default_values(self):
        """Test default evaluation config."""
        config = EvaluationConfig()
        assert config.enable_cost_tracking is False
        assert config.max_concurrency == 10
        assert config.retry_attempts == 3
        assert config.progress_bar is True
    
    def test_custom_values(self):
        """Test custom evaluation config."""
        config = EvaluationConfig(
            enable_cost_tracking=True,
            max_concurrency=20,
            rate_limit_requests_per_minute=100
        )
        assert config.enable_cost_tracking is True
        assert config.max_concurrency == 20
        assert config.rate_limit_requests_per_minute == 100


class TestReportingConfig:
    """Test reporting configuration."""
    
    def test_default_values(self):
        """Test default reporting config."""
        config = ReportingConfig()
        assert config.output_dir == "./results"
        assert config.save_results is True
        assert "json" in config.formats
    
    def test_custom_formats(self):
        """Test custom output formats."""
        config = ReportingConfig(
            formats=["json", "csv", "html"],
            include_charts=True
        )
        assert len(config.formats) == 3
        assert config.include_charts is True


class TestNexusConfig:
    """Test main configuration."""
    
    def test_default_creation(self):
        """Test creating config with defaults."""
        config = NexusConfig()
        assert config.llm.provider == "openai"
        assert config.cache.enabled is True
        assert config.evaluation.max_concurrency == 10
    
    def test_custom_creation(self):
        """Test creating custom config."""
        config = NexusConfig(
            llm=LLMConfig(model="gpt-4o-mini"),
            cache=CacheConfig(backend="file"),
            evaluation=EvaluationConfig(enable_cost_tracking=True)
        )
        assert config.llm.model == "gpt-4o-mini"
        assert config.cache.backend == "file"
        assert config.evaluation.enable_cost_tracking is True
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "llm": {"model": "gpt-4o", "temperature": 0.5},
            "cache": {"enabled": False}
        }
        config = NexusConfig.from_dict(data)
        assert config.llm.model == "gpt-4o"
        assert config.llm.temperature == 0.5
        assert config.cache.enabled is False
    
    def test_to_file_and_from_file(self, tmp_path):
        """Test saving and loading config from file."""
        config = NexusConfig(
            llm=LLMConfig(model="gpt-4o-mini"),
            cache=CacheConfig(backend="file")
        )
        
        # Save to file
        config_file = tmp_path / "config.json"
        config.to_file(str(config_file))
        
        # Load from file
        loaded_config = NexusConfig.from_file(str(config_file))
        assert loaded_config.llm.model == "gpt-4o-mini"
        assert loaded_config.cache.backend == "file"
    
    def test_preset_development(self):
        """Test development preset."""
        config = NexusConfig.preset_development()
        assert config.llm.model == "gpt-4o-mini"  # Cheaper model
        assert config.cache.enabled is True
        assert config.evaluation.enable_cost_tracking is True
        assert config.reporting.verbose is True
    
    def test_preset_production(self):
        """Test production preset."""
        config = NexusConfig.preset_production()
        assert config.llm.model == "gpt-4-turbo"  # Best model
        assert config.cache.backend == "redis"
        assert config.evaluation.max_concurrency == 20
        assert config.reporting.verbose is False
    
    def test_preset_fast(self):
        """Test fast preset."""
        config = NexusConfig.preset_fast()
        assert config.llm.model == "gpt-3.5-turbo"  # Fastest model
        assert config.cache.backend == "memory"
        assert config.evaluation.max_concurrency == 30
        assert config.evaluation.retry_attempts == 2
