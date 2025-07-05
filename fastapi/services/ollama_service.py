"""
🏠 Ollama Service
Integration with local Ollama instance for running models locally
"""

import os
import aiohttp
import logging
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for interacting with local Ollama instance"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return True  # Always available if configured
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def chat(self, prompt: str, context: Optional[List[Dict]] = None, model: str = "llama3") -> str:
        """Send chat request to Ollama"""
        try:
            # Build messages
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4000
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}")
            raise
    
    async def ensure_model_available(self, model: str) -> bool:
        """Ensure a model is available (pull if necessary)"""
        try:
            # Check if model exists
            models = await self.get_available_models()
            if model in models:
                return True
            
            # Pull the model
            logger.info(f"Pulling model {model}...")
            async with aiohttp.ClientSession() as session:
                payload = {"name": model}
                
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes
                ) as response:
                    if response.status == 200:
                        # Read streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "status" in data:
                                        logger.info(f"Pull status: {data['status']}")
                                except:
                                    pass
                        return True
                    else:
                        return False
                        
        except Exception as e:
            logger.error(f"Error ensuring model {model}: {e}")
            return False
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    else:
                        return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []