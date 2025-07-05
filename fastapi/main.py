"""
🚀 AI Dev Superstack - FastAPI Main Application
Intelligent Multi-LLM Routing with Performance-Based Selection
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# Import services
from services.claude_service import ClaudeService
from services.openai_service import OpenAIService
from services.huggingface_service import HuggingFaceService
from services.ollama_service import OllamaService
from services.model_performance import ModelPerformanceTracker
from models.requests import ChatRequest, RoutingExplanationRequest
from models.responses import ChatResponse
from routers.llm_router import IntelligentRouter
from config.model_benchmarks import MODEL_BENCHMARKS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting AI Dev Superstack - Intelligent Routing Mode...")
    
    # Initialize services
    app.state.services = {
        "openai": OpenAIService(),
        "claude": ClaudeService(),
        "huggingface": HuggingFaceService(),
        "ollama": OllamaService()
    }
    
    # Initialize performance tracker
    app.state.performance_tracker = ModelPerformanceTracker()
    
    # Initialize intelligent router
    app.state.router = IntelligentRouter(
        services=app.state.services,
        performance_tracker=app.state.performance_tracker,
        benchmarks=MODEL_BENCHMARKS
    )
    
    # Setup default Ollama models
    asyncio.create_task(setup_ollama_models())
    
    logger.info("✅ AI Dev Superstack ready with Intelligent Routing!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Dev Superstack...")

app = FastAPI(
    title="AI Dev Superstack - Intelligent Routing",
    description="Performance-based multi-LLM orchestration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def setup_ollama_models():
    """Setup default Ollama models"""
    try:
        ollama = OllamaService()
        default_models = ["llama3", "mistral"]
        
        for model in default_models:
            logger.info(f"📥 Checking/Pulling {model}...")
            success = await ollama.ensure_model_available(model)
            if success:
                logger.info(f"✅ {model} ready")
            else:
                logger.warning(f"⚠️ Could not setup {model}")
    except Exception as e:
        logger.error(f"❌ Error setting up Ollama models: {e}")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with intelligent routing based on model performance
    """
    start_time = time.time()
    
    try:
        # Use intelligent router to select best model
        selected_model, selected_service = await app.state.router.route_request(request)
        
        logger.info(f"🎯 Intelligently routed to {selected_model} for: {request.prompt[:50]}...")
        
        # Make the request
        response = await selected_service.chat(request.prompt, request.context)
        
        # Track performance
        processing_time = time.time() - start_time
        await app.state.performance_tracker.record_performance(
            model=selected_model,
            task_type=app.state.router.classify_task(request.prompt),
            response_time=processing_time,
            success=True
        )
        
        return ChatResponse(
            response=response,
            model_used=selected_model,
            processing_time=processing_time,
            tokens_used=len(response.split()),  # Simplified
            timestamp=time.time(),
            routing_reason=f"Best performer for {app.state.router.classify_task(request.prompt)} tasks"
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        # Track failure
        if 'selected_model' in locals():
            await app.state.performance_tracker.record_performance(
                model=selected_model,
                task_type=app.state.router.classify_task(request.prompt),
                response_time=time.time() - start_time,
                success=False
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/routing/stats")
async def get_routing_stats():
    """Get routing statistics and model performance"""
    return await app.state.performance_tracker.get_stats()

@app.get("/api/v1/models/benchmarks")
async def get_model_benchmarks():
    """Get current model benchmark scores"""
    return MODEL_BENCHMARKS

@app.post("/api/v1/routing/explain")
async def explain_routing(request: RoutingExplanationRequest):
    """Get explanation for routing decision for a given prompt"""
    return await app.state.router.get_routing_explanation(request.prompt)

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "🧠 AI Dev Superstack - Intelligent Routing Edition",
        "version": "2.0.0",
        "docs": "/docs",
        "features": {
            "intelligent_routing": "Active",
            "performance_tracking": "Active",
            "active_providers": list(app.state.services.keys()),
            "backup_retention": "180 days"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    service_status = {}
    for name, service in app.state.services.items():
        try:
            service_status[name] = "healthy" if await service.health_check() else "unhealthy"
        except Exception as e:
            service_status[name] = "unavailable"
            logger.error(f"Health check failed for {name}: {e}")
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": service_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )