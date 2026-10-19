"""
Ollama provider implementation for local models.
"""

from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse, retry_provider_request

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local models.
    
    Supports: Llama, Mistral, Phi, Gemma, and many others running locally
    
    Requires: 
    - Ollama installed and running (https://ollama.ai)
    - pip install aiohttp
    """
    
    # Local models are free!
    PRICING = {}
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,  # Local models can be slower
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name (e.g., "llama3", "mistral", "phi")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp package not installed. "
                "Install with: pip install aiohttp"
            )
        
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
    
    @retry_provider_request
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        kwargs = {**self.config, **kwargs}
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),
                    "num_predict": kwargs.get("max_tokens", 1024),
                }
            }
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Track usage (approximate)
                if "prompt_eval_count" in result and "eval_count" in result:
                    self.update_usage(
                        result.get("prompt_eval_count", 0),
                        result.get("eval_count", 0)
                    )
                
                return result.get("response", "")
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a JSON object, raising on HTTP errors or exhausted retries."""
        kwargs = {**self.config, **kwargs}

        async def request():
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.0),
                        "num_predict": kwargs.get("max_tokens", 1024),
                    },
                }
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    self.update_usage(result.get("prompt_eval_count", 0), result.get("eval_count", 0))
                    return result["response"]

        return await self._generate_json_with_retry(request)

    @retry_provider_request
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        kwargs = {**self.config, **kwargs}
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),
                    "num_predict": kwargs.get("max_tokens", 1024),
                }
            }
            
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                content = result.get("message", {}).get("content", "")
                
                usage_dict = None
                if "prompt_eval_count" in result and "eval_count" in result:
                    usage_dict = {
                        "input_tokens": result.get("prompt_eval_count", 0),
                        "output_tokens": result.get("eval_count", 0)
                    }
                    self.update_usage(
                        result.get("prompt_eval_count", 0),
                        result.get("eval_count", 0)
                    )
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage_dict,
                    cost=0.0,  # Local is free!
                    metadata={"local": True}
                )
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free!"""
        return 0.0
