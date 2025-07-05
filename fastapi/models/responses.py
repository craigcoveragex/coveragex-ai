"""
📝 Response Models
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ChatResponse(BaseModel):
    """Standard chat response"""
    response: str = Field(..., description="The model's response")
    model_used: str = Field(..., description="Which model was used")
    processing_time: float = Field(..., description="Time taken to process request")
    tokens_used: int = Field(..., description="Approximate token count")
    timestamp: float = Field(..., description="Response timestamp")
    routing_reason: Optional[str] = Field(
        None, 
        description="Explanation of why this model was chosen"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Here's a Python function for fibonacci...",
                "model_used": "openai",
                "processing_time": 1.23,
                "tokens_used": 156,
                "timestamp": 1704067200.0,
                "routing_reason": "Best performer for coding tasks"
            }
        }

class MultiModelResponse(BaseModel):
    """Response from multiple models"""
    results: Dict[str, Dict[str, Any]]
    prompt: str
    comparison_summary: Optional[str] = None

class RoutingStatsResponse(BaseModel):
    """Routing statistics response"""
    models: Dict[str, Dict[str, Any]]
    task_types: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    services: Dict[str, str]