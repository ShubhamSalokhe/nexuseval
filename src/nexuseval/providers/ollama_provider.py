"""
Ollama provider implementation for local models.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse

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
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
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
                result = await response.json()
                
                # Track usage (approximate)
                if "prompt_eval_count" in result and "eval_count" in result:
                    self.update_usage(
                        result.get("prompt_eval_count", 0),
                        result.get("eval_count", 0)
                    )
                
                return result.get("response", "")
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "format": "json",  # Ollama's JSON mode
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),
                    "num_predict": kwargs.get("max_tokens", 1024),
                }
            }
            
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    result = await response.json()
                    
                    content = result.get("response", "{}")
                    parsed = json.loads(content)
                    
                    # Track usage (approximate)
                    if "prompt_eval_count" in result and "eval_count" in result:
                        self.update_usage(
                            result.get("prompt_eval_count", 0),
                            result.get("eval_count", 0)
                        )
                    
                    return parsed
                    
            except json.JSONDecodeError as e:
                return {"score": 0.0, "reason": f"Invalid JSON response: {str(e)}"}
            except Exception as e:
                return {"score": 0.0, "reason": f"Ollama Error: {str(e)}"}
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
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
