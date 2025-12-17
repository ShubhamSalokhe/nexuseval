"""
Google Gemini provider implementation.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMMessage, LLMResponse

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

class GoogleProvider(BaseLLMProvider):
    """
    Google provider for Gemini models.
    
    Supports: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro
    
    Requires: pip install google-generativeai
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
        "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Google provider.
        
        Args:
            model: Gemini model name
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            max_retries: Number of retry attempts
            retry_delay: Initial delay between retries
            **kwargs: Additional configuration
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google GenerativeAI package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        super().__init__(model, **kwargs)
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        
        response = await self.client.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        # Track usage (approximate)
        if response.text:
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response.text)
            self.update_usage(input_tokens, output_tokens)
        
        return response.text
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output."""
        for attempt in range(self.max_retries):
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.0),
                    max_output_tokens=kwargs.get("max_tokens", 1024),
                    response_mime_type="application/json"
                )
                
                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                
                content = response.text
                result = json.loads(content)
                
                # Track usage (approximate)
                input_tokens = self.count_tokens(prompt)
                output_tokens = self.count_tokens(content)
                self.update_usage(input_tokens, output_tokens)
                
                return result
                
            except json.JSONDecodeError as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Invalid JSON response: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"score": 0.0, "reason": f"Google Error: {str(e)}"}
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {"score": 0.0, "reason": "Max retries exceeded"}
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        # Convert to Gemini format
        history = []
        last_user_message = None
        
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have system role, prepend to first user message
                continue
            elif msg.role == "user":
                last_user_message = msg.content
            else:  # assistant
                if last_user_message:
                    history.append({
                        "role": "user",
                        "parts": [last_user_message]
                    })
                    last_user_message = None
                history.append({
                    "role": "model",
                    "parts": [msg.content]
                })
        
        # Start chat session
        chat = self.client.start_chat(history=history)
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        
        if last_user_message:
            response = await chat.send_message_async(
                last_user_message,
                generation_config=generation_config
            )
        else:
            # If no pending user message, use last message from list
            response = await chat.send_message_async(
                messages[-1].content if messages else "",
                generation_config=generation_config
            )
        
        # Track usage (approximate)
        input_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        output_tokens = self.count_tokens(response.text)
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.update_usage(input_tokens, output_tokens)
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            cost=cost
        )
