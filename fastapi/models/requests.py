"""
📝 Request Models
Pydantic models for API requests
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChatRequest(BaseModel):
    """Chat request model"""
    prompt: str = Field(..., description="The user's prompt or question")
    model: Optional[str] = Field(None, description="Specific model to use (optional)")
    context: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Conversation context as list of messages"
    )
    user_id: Optional[str] = Field("default", description="User identifier")
    save_to_memory: bool = Field(True, description="Whether to save conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4000, ge=1, le=8000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a Python function to calculate fibonacci numbers",
                "context": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! How can I help?"}
                ],
                "save_to_memory": True
            }
        }

class MultiModelRequest(BaseModel):
    """Request for comparing multiple models"""
    prompt: str
    models: List[str] = Field(
        default=["openai", "claude", "ollama"],
        description="List of models to compare"
    )
    context: Optional[List[Dict[str, str]]] = None
    
class RoutingExplanationRequest(BaseModel):
    """Request for routing explanation"""
    prompt: str = Field(..., description="The prompt to analyze")