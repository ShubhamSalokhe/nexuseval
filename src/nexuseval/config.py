"""
Configuration management for NexusEval.

Provides centralized configuration with support for:
- Loading from YAML/JSON files
- Environment variable overrides
- Validation with Pydantic
- Preset configurations
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    provider: Literal["openai", "anthropic", "google", "groq", "ollama"] = "openai"
    model: str = "gpt-4-turbo"
    api_key: Optional[str] = None
    
    # Provider-specific options
    base_url: Optional[str] = None  # For Ollama, Azure OpenAI
    region: Optional[str] = None  # For AWS Bedrock (future)
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 30
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    class Config:
        extra = "allow"  # Allow provider-specific extras


class CacheConfig(BaseModel):
    """
    Configuration for caching system.
    """
    enabled: bool = Field(
        default=True,
        description="Whether caching is enabled"
    )
    backend: Literal["memory", "file", "redis", "none"] = Field(
        default="memory",
        description="Cache backend to use"
    )
    max_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of cached entries"
    )
    ttl: Optional[int] = Field(
        default=None,
        description="Time to live in seconds (None = no expiration)"
    )
    cache_dir: str = Field(
        default=".nexuseval_cache",
        description="Directory for file-based cache"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (for redis backend)"
    )


class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation runs.
    """
    enable_cost_tracking: bool = Field(
        default=False,
        description="Track API costs for evaluations"
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent evaluations"
    )
    rate_limit_requests_per_minute: Optional[int] = Field(
        default=None,
        description="Rate limit for API requests (None = no limit)"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts for failed requests"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial delay between retries (seconds, with exponential backoff)"
    )
    progress_bar: bool = Field(
        default=True,
        description="Show progress bar during evaluation"
    )


class ReportingConfig(BaseModel):
    """
    Configuration for reporting and output.
    """
    output_dir: str = Field(
        default="./results",
        description="Directory for output files"
    )
    save_results: bool = Field(
        default=True,
        description="Automatically save results to files"
    )
    formats: list[str] = Field(
        default=["json"],
        description="Output formats (json, csv, html)"
    )
    include_charts: bool = Field(
        default=False,
        description="Include visualizations in HTML reports"
    )
    verbose: bool = Field(
        default=True,
        description="Verbose output during evaluation"
    )


class NexusConfig(BaseModel):
    """
    Main configuration object for NexusEval.
    """
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    
    @classmethod
    def from_file(cls, path: str) -> "NexusConfig":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
        
        Returns:
            NexusConfig object
        
        Example:
            >>> config = NexusConfig.from_file("config.json")
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path_obj, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NexusConfig":
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
        
        Returns:
            NexusConfig object
        """
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "NexusConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
        - NEXUSEVAL_LLM_PROVIDER
        - NEXUSEVAL_LLM_MODEL
        - NEXUSEVAL_CACHE_BACKEND
        - NEXUSEVAL_CACHE_ENABLED
        - etc.
        
        Returns:
            NexusConfig object
        """
        config_data: Dict[str, Any] = {
            "llm": {},
            "cache": {},
            "evaluation": {},
            "reporting": {}
        }
        
        # LLM config from env
        if os.getenv("NEXUSEVAL_LLM_PROVIDER"):
            config_data["llm"]["provider"] = os.getenv("NEXUSEVAL_LLM_PROVIDER")
        if os.getenv("NEXUSEVAL_LLM_MODEL"):
            config_data["llm"]["model"] = os.getenv("NEXUSEVAL_LLM_MODEL")
        if os.getenv("NEXUSEVAL_LLM_TEMPERATURE"):
            config_data["llm"]["temperature"] = float(os.getenv("NEXUSEVAL_LLM_TEMPERATURE"))
        
        # Cache config from env
        if os.getenv("NEXUSEVAL_CACHE_ENABLED"):
            config_data["cache"]["enabled"] = os.getenv("NEXUSEVAL_CACHE_ENABLED").lower() == "true"
        if os.getenv("NEXUSEVAL_CACHE_BACKEND"):
            config_data["cache"]["backend"] = os.getenv("NEXUSEVAL_CACHE_BACKEND")
        
        # Evaluation config from env
        if os.getenv("NEXUSEVAL_COST_TRACKING"):
            config_data["evaluation"]["enable_cost_tracking"] = os.getenv("NEXUSEVAL_COST_TRACKING").lower() == "true"
        
        return cls(**config_data)
    
    def to_file(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path: Output path for configuration file
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def preset_development(cls) -> "NexusConfig":
        """
        Preset configuration for development.
        - Uses cheaper models
        - Enables caching
        - Verbose output
        """
        return cls(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",  # Cheaper model
                temperature=0.0
            ),
            cache=CacheConfig(
                enabled=True,
                backend="file",
                max_size=500
            ),
            evaluation=EvaluationConfig(
                enable_cost_tracking=True,
                max_concurrency=5,
                retry_attempts=3,
                progress_bar=True
            ),
            reporting=ReportingConfig(
                verbose=True,
                formats=["json"],
                include_charts=False
            )
        )
    
    @classmethod
    def preset_production(cls) -> "NexusConfig":
        """
        Preset configuration for production.
        - Uses best models
        - Redis caching (if available)
        - Higher concurrency
        - Minimal output
        """
        return cls(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.0
            ),
            cache=CacheConfig(
                enabled=True,
                backend="redis",
                max_size=10000,
                ttl=86400  # 24 hours
            ),
            evaluation=EvaluationConfig(
                enable_cost_tracking=True,
                max_concurrency=20,
                rate_limit_requests_per_minute=500,
                retry_attempts=5,
                progress_bar=False
            ),
            reporting=ReportingConfig(
                verbose=False,
                save_results=True,
                formats=["json", "csv"],
                include_charts=True
            )
        )
    
    @classmethod
    def preset_fast(cls) -> "NexusConfig":
        """
        Preset configuration optimized for speed.
        - Fast model
        - In-memory caching
        - High concurrency
        """
        return cls(
            llm=LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=500  # Shorter responses
            ),
            cache=CacheConfig(
                enabled=True,
                backend="memory",
                max_size=2000
            ),
            evaluation=EvaluationConfig(
                enable_cost_tracking=False,
                max_concurrency=30,
                retry_attempts=2,
                progress_bar=True
            ),
            reporting=ReportingConfig(
                verbose=False,
                save_results=False,
                formats=["json"]
            )
        )
