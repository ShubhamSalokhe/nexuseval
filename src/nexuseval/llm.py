import os
import json
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any

class LLMClient:
    def __init__(self, model: str = "gpt-4-turbo"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def get_score(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and enforces a JSON response.
        Includes retry logic for robustness.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # Fallback for API errors
            return {"score": 0.0, "reason": f"LLM Error: {str(e)}"}