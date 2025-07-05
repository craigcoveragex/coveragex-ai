"""
🤗 HuggingFace Service
Integration with HuggingFace Inference API
"""

import os
import aiohttp
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class HuggingFaceService:
    """Service for interacting with HuggingFace Inference API"""
    
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Default models for different tasks
        self.models = {
            "chat": "microsoft/DialoGPT-medium",
            "code": "Salesforce/codegen-2B-mono",
            "creative": "EleutherAI/gpt-neo-2.7B",
            "analysis": "google/flan-t5-large"
        }
        
        if not self.api_key:
            logger.warning("⚠️ HUGGINGFACE_API_KEY not found")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.api_key)
    
    async def health_check(self) -> bool:
        """Check service health"""
        if not self.api_key:
            return False
        
        try:
            # Try a simple request to check API key validity
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/bert-base-uncased",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status < 500
        except:
            return False
    
    async def chat(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Send chat request to HuggingFace"""
        if not self.api_key:
            raise Exception("HuggingFace API key not configured")
        
        try:
            # Determine best model based on prompt
            model = self._select_model(prompt)
            
            # Build the input
            if context:
                # Build conversation history
                conversation = []
                for msg in context:
                    conversation.append(f"{msg['role']}: {msg['content']}")
                conversation.append(f"user: {prompt}")
                full_input = "\n".join(conversation)
            else:
                full_input = prompt
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": full_input,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/{model}",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Handle different response formats
                        if isinstance(data, list) and len(data) > 0:
                            return data[0].get("generated_text", "").strip()
                        elif isinstance(data, dict):
                            return data.get("generated_text", "").strip()
                        else:
                            return str(data)
                    else:
                        error_text = await response.text()
                        raise Exception(f"HuggingFace API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ HuggingFace API error: {e}")
            raise
    
    def _select_model(self, prompt: str) -> str:
        """Select appropriate model based on prompt"""
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ["code", "program", "function", "debug"]):
            return self.models["code"]
        elif any(keyword in prompt_lower for keyword in ["story", "creative", "write"]):
            return self.models["creative"]
        elif any(keyword in prompt_lower for keyword in ["analyze", "compare", "evaluate"]):
            return self.models["analysis"]
        else:
            return self.models["chat"]
    
    async def get_available_models(self) -> List[str]:
        """Get list of available HuggingFace models"""
        return list(self.models.values())