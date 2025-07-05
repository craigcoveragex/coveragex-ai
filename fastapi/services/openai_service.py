"""
🤖 OpenAI Service
Integration with OpenAI's GPT models
"""

import os
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for interacting with OpenAI's API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.client)
    
    async def health_check(self) -> bool:
        """Check service health"""
        if not self.client:
            return False
        
        try:
            # Simple test to verify API key
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    async def chat(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Send chat request to OpenAI"""
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        try:
            # Build messages
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=4000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise
    
    async def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]