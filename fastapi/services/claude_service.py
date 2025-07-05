"""
🧠 Claude Service
Integration with Anthropic's Claude API
"""

import os
import logging
from typing import List, Dict, Optional
import anthropic
import asyncio

logger = logging.getLogger(__name__)

class ClaudeService:
    """Service for interacting with Anthropic's Claude API"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not found")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.client)
    
    async def health_check(self) -> bool:
        """Check service health"""
        if not self.client:
            return False
        
        try:
            # Simple test to verify API key
            # Using async wrapper for sync client
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Claude health check failed: {e}")
            return False
    
    async def chat(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Send chat request to Claude"""
        if not self.client:
            raise Exception("Claude API key not configured")
        
        try:
            # Build messages
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            
            # Use async wrapper for sync client
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=messages
            )
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            raise
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Claude models"""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]