# 🧠 AI Dev Superstack - Intelligent Multi-LLM Orchestration

## Project Structure
```
ai-dev-superstack/
├── docker-compose.yml
├── .env.example
├── Makefile
├── setup.sh
├── README.md
├── .gitignore
├── .dockerignore
│
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── llm_router.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── claude_service.py
│   │   ├── openai_service.py
│   │   ├── huggingface_service.py
│   │   ├── ollama_service.py
│   │   └── model_performance.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   └── config/
│       ├── __init__.py
│       └── model_benchmarks.py
│
├── monitoring/
│   └── prometheus.yml
│
└── scripts/
    ├── backup_to_drive.py
    └── cleanup_old_backups.py
```

## 📄 docker-compose.yml
```yaml
version: '3.9'

# Development environment - Resource-conscious configuration
services:
  # 🚀 FastAPI - Main Application with Intelligent Routing
  fastapi:
    build:
      context: ./fastapi
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=info
      # Active API Keys
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
      # Placeholder API Keys (Uncomment and fill when ready)
      # - COHERE_API_KEY=${COHERE_API_KEY}
      # - AI21_API_KEY=${AI21_API_KEY}
      # - GOOGLE_AI_API_KEY=${GOOGLE_AI_API_KEY}
      # - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      # - REPLICATE_API_KEY=${REPLICATE_API_KEY}
      # - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      # Google Drive Backup
      - GOOGLE_DRIVE_FOLDER_ID=${GOOGLE_DRIVE_FOLDER_ID}
      - BACKUP_RETENTION_DAYS=180
    volumes:
      - ./fastapi:/app
      - ./data/conversations:/app/data/conversations
      - ./credentials:/app/credentials:ro
    depends_on:
      - redis
      - ollama
    networks:
      - ai-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # 🧠 Ollama - Local LLM Server (Llama3, Mistral)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_KEEP_ALIVE=5m
      - OLLAMA_NUM_PARALLEL=2
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - ai-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          memory: 8G

  # 🔄 Redis - Caching for intelligent routing decisions
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - ai-network
    restart: unless-stopped
    command: redis-server --maxmemory 512mb --maxmemory-policy lru
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G

  # 📊 Prometheus - Lightweight monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - ai-network
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=7d'
      - '--storage.tsdb.retention.size=2GB'
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G

  # 🔄 Backup Service - Google Drive Integration
  backup-service:
    build:
      context: ./scripts
      dockerfile: Dockerfile.backup
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json
      - GOOGLE_DRIVE_FOLDER_ID=${GOOGLE_DRIVE_FOLDER_ID}
      - BACKUP_RETENTION_DAYS=180
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data:ro
      - ./credentials:/app/credentials:ro
    networks:
      - ai-network
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G

volumes:
  ollama-data:
  redis-data:
  prometheus-data:

networks:
  ai-network:
    driver: bridge
```

## 📄 .env.example
```bash
# ===================================
# 🔑 API KEYS - Active Providers
# ===================================
# OpenAI - GPT Models
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic - Claude Models
ANTHROPIC_API_KEY=sk-ant-your-claude-key-here

# HuggingFace - Open Source Models
HUGGINGFACE_API_KEY=hf_your-huggingface-key-here

# ===================================
# 🔑 API KEYS - Future Providers (Uncomment when ready)
# ===================================
# Cohere - Command Models
# COHERE_API_KEY=your-cohere-key-here

# AI21 Labs - Jurassic Models
# AI21_API_KEY=your-ai21-key-here

# Google AI - PaLM/Gemini Models
# GOOGLE_AI_API_KEY=your-google-ai-key-here

# Azure OpenAI Service
# AZURE_OPENAI_API_KEY=your-azure-openai-key-here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Replicate - Various Models
# REPLICATE_API_KEY=your-replicate-key-here

# Together AI - Open Models
# TOGETHER_API_KEY=your-together-key-here

# ===================================
# 📁 Google Drive Backup Configuration
# ===================================
GOOGLE_DRIVE_FOLDER_ID=your-google-drive-folder-id-here

# ===================================
# 🔧 Application Settings
# ===================================
ENVIRONMENT=development
LOG_LEVEL=info
DEBUG=true

# ===================================
# 🚀 Service Configuration
# ===================================
REDIS_URL=redis://redis:6379
OLLAMA_HOST=http://ollama:11434

# ===================================
# 🔒 Security (Change in production)
# ===================================
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET=dev-jwt-secret-change-in-production
```

## 📄 fastapi/main.py
```python
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
from models.requests import ChatRequest
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
        except:
            service_status[name] = "unavailable"
    
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
```

## 📄 fastapi/routers/llm_router.py
```python
"""
🧠 Intelligent LLM Router
Routes requests to the best-performing model based on task type and historical performance
"""

import logging
from typing import Dict, Tuple, Optional, List
import re
from models.requests import ChatRequest

logger = logging.getLogger(__name__)

class IntelligentRouter:
    """Routes requests to models based on performance benchmarks and real-time metrics"""
    
    def __init__(self, services: Dict, performance_tracker, benchmarks: Dict):
        self.services = services
        self.performance_tracker = performance_tracker
        self.benchmarks = benchmarks
        
        # Task classification patterns
        self.task_patterns = {
            "coding": [
                r"code", r"program", r"function", r"debug", r"implement",
                r"python", r"javascript", r"java", r"c\+\+", r"sql", r"api",
                r"algorithm", r"data structure", r"class", r"method"
            ],
            "analysis": [
                r"analyze", r"compare", r"evaluate", r"assess", r"review",
                r"pros and cons", r"advantages", r"disadvantages", r"benchmark",
                r"performance", r"metrics", r"statistics", r"data"
            ],
            "math": [
                r"calculate", r"solve", r"equation", r"formula", r"mathematical",
                r"algebra", r"calculus", r"geometry", r"statistics", r"probability",
                r"integral", r"derivative", r"matrix", r"vector"
            ],
            "creative": [
                r"write", r"story", r"poem", r"creative", r"fiction", r"narrative",
                r"character", r"plot", r"dialogue", r"scene", r"article", r"blog",
                r"essay", r"content", r"copy"
            ],
            "reasoning": [
                r"explain", r"why", r"how", r"reason", r"logic", r"deduce",
                r"infer", r"conclude", r"argument", r"premise", r"hypothesis",
                r"theory", r"proof", r"demonstrate"
            ],
            "qa": [
                r"what is", r"who is", r"when", r"where", r"which", r"how many",
                r"how much", r"define", r"describe", r"tell me about", r"fact"
            ],
            "translation": [
                r"translate", r"translation", r"french", r"spanish", r"german",
                r"chinese", r"japanese", r"language", r"convert to", r"in .+ language"
            ],
            "summarization": [
                r"summarize", r"summary", r"brief", r"overview", r"key points",
                r"main ideas", r"tldr", r"abstract", r"condensed", r"highlights"
            ]
        }
    
    def classify_task(self, prompt: str) -> str:
        """Classify the task type based on the prompt"""
        prompt_lower = prompt.lower()
        
        # Count matches for each category
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, prompt_lower))
            if score > 0:
                scores[task_type] = score
        
        # Return the task type with highest score, or 'general' if no matches
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    async def route_request(self, request: ChatRequest) -> Tuple[str, object]:
        """
        Route request to the best-performing model for the task type
        Returns: (model_name, service_instance)
        """
        # Classify the task
        task_type = self.classify_task(request.prompt)
        logger.info(f"📊 Classified task as: {task_type}")
        
        # Get performance data
        performance_data = await self.performance_tracker.get_model_scores(task_type)
        
        # Combine benchmark scores with real-time performance
        model_scores = {}
        
        for provider, service in self.services.items():
            if not service.is_available():
                continue
                
            # Get benchmark score for this task type
            benchmark_score = self.benchmarks.get(provider, {}).get(task_type, 0.5)
            
            # Get real-time performance score
            perf_score = performance_data.get(provider, {}).get('score', benchmark_score)
            
            # Weighted combination (70% real-time, 30% benchmark)
            if provider in performance_data:
                final_score = 0.7 * perf_score + 0.3 * benchmark_score
            else:
                # If no real-time data, use benchmark only
                final_score = benchmark_score
            
            model_scores[provider] = final_score
            logger.info(f"  {provider}: score={final_score:.3f}")
        
        # Select the best model
        if not model_scores:
            raise HTTPException(status_code=503, detail="No models available")
        
        best_model = max(model_scores, key=model_scores.get)
        logger.info(f"✅ Selected {best_model} (score: {model_scores[best_model]:.3f})")
        
        return best_model, self.services[best_model]
    
    async def get_routing_explanation(self, prompt: str) -> Dict:
        """Explain why a particular model would be chosen"""
        task_type = self.classify_task(prompt)
        performance_data = await self.performance_tracker.get_model_scores(task_type)
        
        explanation = {
            "task_type": task_type,
            "model_scores": {},
            "recommendation": None,
            "reasoning": ""
        }
        
        # Calculate scores
        model_scores = {}
        for provider, service in self.services.items():
            if not service.is_available():
                continue
            
            benchmark_score = self.benchmarks.get(provider, {}).get(task_type, 0.5)
            perf_data = performance_data.get(provider, {})
            
            if perf_data:
                real_time_score = perf_data.get('score', benchmark_score)
                final_score = 0.7 * real_time_score + 0.3 * benchmark_score
                
                model_scores[provider] = {
                    "final_score": final_score,
                    "benchmark_score": benchmark_score,
                    "real_time_score": real_time_score,
                    "sample_size": perf_data.get('count', 0)
                }
            else:
                model_scores[provider] = {
                    "final_score": benchmark_score,
                    "benchmark_score": benchmark_score,
                    "real_time_score": None,
                    "sample_size": 0
                }
        
        explanation["model_scores"] = model_scores
        
        if model_scores:
            best_model = max(model_scores, key=lambda x: model_scores[x]["final_score"])
            explanation["recommendation"] = best_model
            explanation["reasoning"] = (
                f"For {task_type} tasks, {best_model} has the best performance "
                f"with a score of {model_scores[best_model]['final_score']:.3f}"
            )
        
        return explanation
```

## 📄 fastapi/config/model_benchmarks.py
```python
"""
📊 Model Benchmark Scores
Based on common benchmarks and task-specific performance metrics
Scores range from 0.0 to 1.0 (higher is better)
"""

MODEL_BENCHMARKS = {
    "openai": {
        "coding": 0.95,        # Excellent at code generation and debugging
        "analysis": 0.90,      # Strong analytical capabilities
        "math": 0.92,          # Very good at mathematical problems
        "creative": 0.88,      # Good creative writing
        "reasoning": 0.94,     # Excellent logical reasoning
        "qa": 0.91,           # Very good factual Q&A
        "translation": 0.89,   # Good translation capabilities
        "summarization": 0.90, # Strong summarization
        "general": 0.91        # Overall strong performance
    },
    "claude": {
        "coding": 0.93,        # Excellent coding with good explanations
        "analysis": 0.95,      # Outstanding analytical thinking
        "math": 0.88,          # Good mathematical reasoning
        "creative": 0.92,      # Excellent creative writing
        "reasoning": 0.96,     # Best-in-class reasoning
        "qa": 0.90,           # Very accurate Q&A
        "translation": 0.87,   # Good translations
        "summarization": 0.94, # Excellent summarization
        "general": 0.92        # Overall excellent performance
    },
    "huggingface": {
        "coding": 0.75,        # Decent for simpler code
        "analysis": 0.70,      # Basic analysis capabilities
        "math": 0.65,          # Limited math capabilities
        "creative": 0.80,      # Good for creative tasks
        "reasoning": 0.68,     # Basic reasoning
        "qa": 0.72,           # Decent for simple Q&A
        "translation": 0.82,   # Good with dedicated models
        "summarization": 0.78, # Good with dedicated models
        "general": 0.74        # Good for specific tasks
    },
    "ollama": {
        # Scores for Llama3 (can be adjusted based on specific model)
        "coding": 0.82,        # Good coding capabilities
        "analysis": 0.78,      # Decent analysis
        "math": 0.75,          # Good basic math
        "creative": 0.84,      # Good creative writing
        "reasoning": 0.80,     # Good reasoning for local model
        "qa": 0.79,           # Good Q&A performance
        "translation": 0.73,   # Basic translation
        "summarization": 0.81, # Good summarization
        "general": 0.79        # Solid all-around performance
    }
}

# Task complexity weights (for cost optimization)
TASK_COMPLEXITY = {
    "coding": 0.9,        # Complex tasks
    "analysis": 0.85,     # Complex analysis
    "math": 0.8,          # Moderate to complex
    "creative": 0.7,      # Moderate complexity
    "reasoning": 0.85,    # Complex reasoning
    "qa": 0.4,           # Simple tasks
    "translation": 0.6,   # Moderate complexity
    "summarization": 0.65, # Moderate complexity
    "general": 0.6        # Average complexity
}
```

## 📄 fastapi/services/model_performance.py
```python
"""
📊 Model Performance Tracker
Tracks real-time performance metrics for intelligent routing decisions
"""

import asyncio
import time
from typing import Dict, List, Optional
from collections import defaultdict
import statistics
import redis.asyncio as redis
import json
import logging

logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    """Tracks and analyzes model performance for intelligent routing"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.performance_window = 3600  # 1 hour window for performance tracking
        self.min_samples = 5  # Minimum samples before considering real-time data
        
    async def _get_redis(self):
        """Get Redis connection"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(self.redis_url)
        return self.redis_client
    
    async def record_performance(
        self,
        model: str,
        task_type: str,
        response_time: float,
        success: bool,
        tokens_used: Optional[int] = None
    ):
        """Record performance metrics for a model"""
        try:
            r = await self._get_redis()
            
            # Create performance record
            record = {
                "timestamp": time.time(),
                "model": model,
                "task_type": task_type,
                "response_time": response_time,
                "success": success,
                "tokens_used": tokens_used or 0
            }
            
            # Store in Redis with expiration
            key = f"perf:{model}:{task_type}"
            await r.rpush(key, json.dumps(record))
            await r.expire(key, self.performance_window)
            
            # Also update aggregated stats
            await self._update_stats(model, task_type, response_time, success)
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    async def _update_stats(self, model: str, task_type: str, response_time: float, success: bool):
        """Update aggregated statistics"""
        try:
            r = await self._get_redis()
            stats_key = f"stats:{model}:{task_type}"
            
            # Get current stats
            stats_data = await r.get(stats_key)
            if stats_data:
                stats = json.loads(stats_data)
            else:
                stats = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0
                }
            
            # Update stats
            stats["count"] += 1
            if success:
                stats["success_count"] += 1
            stats["total_time"] += response_time
            stats["min_time"] = min(stats["min_time"], response_time)
            stats["max_time"] = max(stats["max_time"], response_time)
            
            # Save updated stats
            await r.set(stats_key, json.dumps(stats), ex=self.performance_window)
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    async def get_model_scores(self, task_type: str) -> Dict[str, Dict]:
        """Get performance scores for all models for a specific task type"""
        try:
            r = await self._get_redis()
            scores = {}
            
            # Get all models
            models = ["openai", "claude", "huggingface", "ollama"]
            
            for model in models:
                stats_key = f"stats:{model}:{task_type}"
                stats_data = await r.get(stats_key)
                
                if stats_data:
                    stats = json.loads(stats_data)
                    
                    if stats["count"] >= self.min_samples:
                        # Calculate performance score
                        success_rate = stats["success_count"] / stats["count"]
                        avg_time = stats["total_time"] / stats["count"]
                        
                        # Score formula: 70% success rate, 30% speed (normalized)
                        # Speed score: faster is better, normalized to 0-1
                        speed_score = 1.0 / (1.0 + avg_time)  # Inverse time, bounded
                        
                        overall_score = 0.7 * success_rate + 0.3 * speed_score
                        
                        scores[model] = {
                            "score": overall_score,
                            "success_rate": success_rate,
                            "avg_response_time": avg_time,
                            "count": stats["count"]
                        }
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting model scores: {e}")
            return {}
    
    async def get_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        try:
            r = await self._get_redis()
            all_stats = {
                "models": {},
                "task_types": defaultdict(dict),
                "summary": {
                    "total_requests": 0,
                    "total_successes": 0,
                    "avg_response_time": 0
                }
            }
            
            models = ["openai", "claude", "huggingface", "ollama"]
            task_types = ["coding", "analysis", "math", "creative", "reasoning", 
                         "qa", "translation", "summarization", "general"]
            
            total_time = 0
            total_count = 0
            
            for model in models:
                all_stats["models"][model] = {}
                
                for task_type in task_types:
                    stats_key = f"stats:{model}:{task_type}"
                    stats_data = await r.get(stats_key)
                    
                    if stats_data:
                        stats = json.loads(stats_data)
                        all_stats["models"][model][task_type] = stats
                        all_stats["task_types"][task_type][model] = stats
                        
                        total_count += stats["count"]
                        all_stats["summary"]["total_successes"] += stats["success_count"]
                        total_time += stats["total_time"]
            
            all_stats["summary"]["total_requests"] = total_count
            if total_count > 0:
                all_stats["summary"]["avg_response_time"] = total_time / total_count
                all_stats["summary"]["overall_success_rate"] = (
                    all_stats["summary"]["total_successes"] / total_count
                )
            
            return all_stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
```

## 📄 fastapi/services/huggingface_service.py
```python
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
```

## 📄 scripts/backup_to_drive.py
```python
#!/usr/bin/env python3
"""
📁 Google Drive Backup Service
Backs up conversation data to Google Drive with 180-day retention
"""

import os
import json
import time
import tarfile
import logging
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveBackup:
    def __init__(self):
        self.folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", 180))
        self.data_dir = "/app/data"
        self.credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Initialize Google Drive service
        self.service = self._init_drive_service()
    
    def _init_drive_service(self):
        """Initialize Google Drive API service"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            return None
    
    def create_backup(self):
        """Create a backup of conversation data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"ai_superstack_backup_{timestamp}.tar.gz"
            backup_path = f"/tmp/{backup_filename}"
            
            # Create tar.gz archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(self.data_dir, arcname="conversations")
            
            logger.info(f"Created backup: {backup_filename}")
            return backup_path, backup_filename
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None, None
    
    def upload_to_drive(self, file_path, file_name):
        """Upload backup to Google Drive"""
        if not self.service:
            logger.error("Google Drive service not initialized")
            return False
        
        try:
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(
                file_path,
                mimetype='application/gzip',
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"Uploaded backup to Google Drive: {file.get('id')}")
            
            # Clean up local file
            os.remove(file_path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        if not self.service:
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%dT%H:%M:%S")
            
            # Query for old files
            query = (
                f"'{self.folder_id}' in parents and "
                f"createdTime < '{cutoff_str}' and "
                f"name contains 'ai_superstack_backup_'"
            )
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, createdTime)"
            ).execute()
            
            files = results.get('files', [])
            
            for file in files:
                try:
                    self.service.files().delete(fileId=file['id']).execute()
                    logger.info(f"Deleted old backup: {file['name']}")
                except Exception as e:
                    logger.error(f"Failed to delete {file['name']}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def run_backup(self):
        """Main backup process"""
        logger.info("Starting backup process...")
        
        # Create backup
        backup_path, backup_name = self.create_backup()
        if not backup_path:
            return
        
        # Upload to Drive
        if self.upload_to_drive(backup_path, backup_name):
            logger.info("Backup completed successfully")
        
        # Cleanup old backups
        self.cleanup_old_backups()

def main():
    """Main entry point"""
    backup = GoogleDriveBackup()
    
    # Run backup immediately
    backup.run_backup()
    
    # Schedule daily backups at 2 AM
    schedule.every().day.at("02:00").do(backup.run_backup)
    
    logger.info("Backup service started. Running daily at 2 AM.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
```

## 📄 Makefile
```makefile
.PHONY: help setup start stop restart logs status test clean backup

# Colors
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[1;33m
NC=\033[0m

help: ## Show this help message
	@echo "$(BLUE)🧠 AI Dev Superstack - Intelligent Routing Edition$(NC)"
	@echo "$(GREEN)============================================$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## 🚀 Initial setup and configuration
	@echo "$(BLUE)🚀 Setting up AI Dev Superstack...$(NC)"
	@chmod +x setup.sh
	@./setup.sh

start: ## ▶️ Start all services
	@echo "$(BLUE)▶️ Starting services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started!$(NC)"
	@make status

stop: ## ⏹️ Stop all services
	@echo "$(BLUE)⏹️ Stopping services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

restart: ## 🔄 Restart all services
	@echo "$(BLUE)🔄 Restarting services...$(NC)"
	@docker-compose restart
	@echo "$(GREEN)✅ Services restarted!$(NC)"

logs: ## 📋 Show logs from all services
	@docker-compose logs -f --tail=100

logs-api: ## 📋 Show FastAPI logs
	@docker-compose logs -f fastapi --tail=100

status: ## 📊 Show service status
	@echo "$(BLUE)📊 Service Status:$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(BLUE)🔍 Health Checks:$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "$(RED)❌ FastAPI not responding$(NC)"

test: ## 🧪 Run a test request
	@echo "$(BLUE)🧪 Testing intelligent routing...$(NC)"
	@curl -s -X POST http://localhost:8000/api/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Write a Python function to calculate fibonacci numbers"}' \
		| python -m json.tool

routing-stats: ## 📊 Show routing statistics
	@echo "$(BLUE)📊 Routing Statistics:$(NC)"
	@curl -s http://localhost:8000/api/v1/routing/stats | python -m json.tool

benchmarks: ## 📊 Show model benchmarks
	@echo "$(BLUE)📊 Model Benchmarks:$(NC)"
	@curl -s http://localhost:8000/api/v1/models/benchmarks | python -m json.tool

models: ## 🧠 List available models
	@echo "$(BLUE)🧠 Ollama Models:$(NC)"
	@docker-compose exec ollama ollama list

pull-model: ## 📥 Pull an Ollama model (usage: make pull-model MODEL=llama3)
	@echo "$(BLUE)📥 Pulling model: $(MODEL)$(NC)"
	@docker-compose exec ollama ollama pull $(MODEL)

clean: ## 🧹 Clean up containers and volumes
	@echo "$(RED)🧹 Cleaning up...$(NC)"
	@docker-compose down -v
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

backup: ## 💾 Create manual backup
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@docker-compose exec backup-service python /app/backup_to_drive.py
	@echo "$(GREEN)✅ Backup complete!$(NC)"
```

## 📄 setup.sh
```bash
#!/bin/bash
set -e

echo "🧠 AI Dev Superstack - Intelligent Routing Setup"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose detected${NC}"

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p fastapi/{services,models,routers,config}
mkdir -p data/conversations
mkdir -p credentials
mkdir -p monitoring
mkdir -p scripts

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}🔧 Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Please edit .env file with your API keys${NC}"
    echo -e "${YELLOW}   Required: OPENAI_API_KEY, ANTHROPIC_API_KEY${NC}"
    echo -e "${YELLOW}   Optional: HUGGINGFACE_API_KEY, GOOGLE_DRIVE_FOLDER_ID${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Check for Google Drive credentials
if [ ! -f credentials/service-account.json ]; then
    echo -e "${YELLOW}ℹ️ Google Drive backup: Add service account JSON to credentials/service-account.json${NC}"
fi

# Create monitoring config
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

# Pull images
echo -e "${BLUE}📥 Pulling Docker images...${NC}"
docker-compose pull

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
docker-compose up -d

# Wait for services
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 10

# Setup Ollama models
echo -e "${BLUE}📥 Setting up Ollama models...${NC}"
docker-compose exec -T ollama ollama pull llama3 &
docker-compose exec -T ollama ollama pull mistral &
wait

# Final message
echo -e "${GREEN}✅ AI Dev Superstack is ready!${NC}"
echo ""
echo -e "${BLUE}🌐 Access your services:${NC}"
echo -e "  API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  Health: ${GREEN}http://localhost:8000/health${NC}"
echo -e "  Prometheus: ${GREEN}http://localhost:9090${NC}"
echo ""
echo -e "${BLUE}📊 View routing stats:${NC}"
echo -e "  ${YELLOW}make routing-stats${NC}"
echo ""
echo -e "${BLUE}🧪 Test the system:${NC}"
echo -e "  ${YELLOW}make test${NC}"
echo ""
echo -e "${YELLOW}⚠️ Next steps:${NC}"
echo -e "  1. Add your API keys to .env file"
echo -e "  2. (Optional) Add Google Drive service account for backups"
echo -e "  3. Test different prompts to see intelligent routing in action"
```

## 📄 README.md
```markdown
# 🧠 AI Dev Superstack - Intelligent Routing Edition

A development and testing environment for multi-LLM applications with performance-based intelligent routing.

## 🚀 Features

### Core Features (Implemented)
- **🎯 Intelligent Routing**: Automatically routes requests to the best-performing model based on:
  - Task type classification (coding, analysis, creative, etc.)
  - Historical performance metrics
  - Model benchmark scores
- **📊 Performance Tracking**: Real-time tracking of model performance
- **🔄 Multi-Provider Support**: OpenAI, Claude, HuggingFace, and local Ollama models
- **💾 Google Drive Backup**: Automated backups with 180-day retention
- **📈 Monitoring**: Prometheus metrics for all services

### Supported Providers
- ✅ **OpenAI** (GPT-4, GPT-3.5)
- ✅ **Anthropic** (Claude)
- ✅ **HuggingFace** (Open source models)
- ✅ **Ollama** (Local models: Llama3, Mistral)
- 🔜 **Cohere** (Ready to enable)
- 🔜 **AI21 Labs** (Ready to enable)
- 🔜 **Google AI** (Ready to enable)
- 🔜 **Azure OpenAI** (Ready to enable)
- 🔜 **Replicate** (Ready to enable)
- 🔜 **Together AI** (Ready to enable)

## 📋 Requirements

- Docker & Docker Compose
- 32GB RAM (for local models)
- API keys for OpenAI and Claude (required)
- HuggingFace API key (optional)
- Google Cloud service account (optional, for backups)

## ⚡ Quick Start

1. **Clone and setup**
   ```bash
   git clone <repository>
   cd ai-dev-superstack
   make setup
   ```

2. **Configure API keys**
   ```bash
   nano .env
   # Add your OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```

3. **Start the system**
   ```bash
   make start
   ```

4. **Test intelligent routing**
   ```bash
   make test
   ```

## 🎯 How Intelligent Routing Works

The system analyzes each request and routes it to the best model based on:

1. **Task Classification**: Identifies the type of task (coding, analysis, creative, etc.)
2. **Performance Metrics**: Considers real-time success rates and response times
3. **Benchmark Scores**: Uses pre-configured benchmark scores for each model
4. **Availability**: Routes only to available services

### Example Routing Decisions

- **Coding Task** → Likely routes to OpenAI (GPT-4) or Claude
- **Creative Writing** → May choose Claude or HuggingFace
- **Simple Q&A** → Could use Ollama for cost efficiency
- **Complex Analysis** → Typically selects Claude for best reasoning

## 📊 Monitoring Performance

View routing statistics:
```bash
make routing-stats
```

View model benchmarks:
```bash
make benchmarks
```

## 🛠️ Development Commands

```bash
make help           # Show all commands
make start          # Start services
make stop           # Stop services
make logs           # View logs
make status         # Check service health
make test           # Run test request
make routing-stats  # View routing statistics
make benchmarks     # View model benchmarks
make backup         # Create manual backup
```

## 📁 Project Structure

```
ai-dev-superstack/
├── fastapi/           # Main API application
│   ├── services/      # LLM service integrations
│   ├── routers/       # Intelligent routing logic
│   └── config/        # Model benchmarks
├── data/              # Conversation storage
├── credentials/       # Service account keys
└── monitoring/        # Prometheus config
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ANTHROPIC_API_KEY`: Your Claude API key (required)
- `HUGGINGFACE_API_KEY`: HuggingFace API key (optional)
- `GOOGLE_DRIVE_FOLDER_ID`: Folder ID for backups (optional)

### Adding New Providers

Uncomment the desired provider in `.env`:
```bash
# COHERE_API_KEY=your-key-here
# AI21_API_KEY=your-key-here
```

## 📈 Performance Optimization

The system is optimized for development/testing with:
- Resource limits on containers
- Efficient caching with Redis
- Smart model selection to minimize costs
- Local models for frequent testing

## 🆘 Troubleshooting

**Services not starting?**
```bash
make status
make logs
```

**Models not routing correctly?**
```bash
make routing-stats  # Check performance metrics
make benchmarks     # View model scores
```

**Need more models?**
```bash
make pull-model MODEL=codellama
```

## 📄 License

MIT License - See LICENSE file for details

---

Built with ❤️ for efficient multi-LLM development and testing
```

# 🧠 AI Dev Superstack - Service Implementation Files

## 📄 fastapi/services/claude_service.py
```python
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
```

## 📄 fastapi/services/openai_service.py
```python
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
```

## 📄 fastapi/services/ollama_service.py
```python
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
```

## 📄 fastapi/services/__init__.py
```python
"""
Service modules for AI Dev Superstack
"""

from .claude_service import ClaudeService
from .openai_service import OpenAIService
from .huggingface_service import HuggingFaceService
from .ollama_service import OllamaService
from .model_performance import ModelPerformanceTracker

__all__ = [
    "ClaudeService",
    "OpenAIService", 
    "HuggingFaceService",
    "OllamaService",
    "ModelPerformanceTracker"
]
```

## 📄 fastapi/models/requests.py
```python
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
```

## 📄 fastapi/models/responses.py
```python
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
```

## 📄 fastapi/models/__init__.py
```python
"""
Model definitions for AI Dev Superstack
"""

from .requests import ChatRequest, MultiModelRequest, RoutingExplanationRequest
from .responses import ChatResponse, MultiModelResponse, RoutingStatsResponse, HealthCheckResponse

__all__ = [
    "ChatRequest",
    "MultiModelRequest", 
    "RoutingExplanationRequest",
    "ChatResponse",
    "MultiModelResponse",
    "RoutingStatsResponse",
    "HealthCheckResponse"
]
```

## 📄 fastapi/routers/__init__.py
```python
"""
Router modules for AI Dev Superstack
"""

from .llm_router import IntelligentRouter

__all__ = ["IntelligentRouter"]
```

## 📄 fastapi/config/__init__.py
```python
"""
Configuration modules for AI Dev Superstack
"""

from .model_benchmarks import MODEL_BENCHMARKS, TASK_COMPLEXITY

__all__ = ["MODEL_BENCHMARKS", "TASK_COMPLEXITY"]
```

## 📄 fastapi/requirements.txt
```
# FastAPI and ASGI
fastapi==0.109.0
uvicorn[standard]==0.25.0
python-multipart==0.0.6

# LLM APIs
anthropic==0.15.0
openai==1.10.0
aiohttp==3.9.1

# Redis for caching and performance tracking
redis[hiredis]==5.0.1

# Data models and validation
pydantic==2.5.3
pydantic-settings==2.1.0

# Utilities
python-dotenv==1.0.0
python-json-logger==2.0.7

# Monitoring
prometheus-client==0.19.0

# Google Drive backup
google-api-python-client==2.114.0
google-auth==2.26.1
google-auth-httplib2==0.2.0

# Async support
asyncio==3.4.3
aiofiles==23.2.1

# Development
pytest==7.4.4
pytest-asyncio==0.21.1
httpx==0.26.0
```

## 📄 fastapi/Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📄 scripts/Dockerfile.backup
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    google-api-python-client==2.114.0 \
    google-auth==2.26.1 \
    schedule==1.2.0

# Copy backup script
COPY backup_to_drive.py .
COPY cleanup_old_backups.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run backup service
CMD ["python", "backup_to_drive.py"]
```

## 📄 scripts/cleanup_old_backups.py
```python
#!/usr/bin/env python3
"""
🧹 Cleanup Old Backups
Removes backups older than retention period
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_local_backups(backup_dir: str, retention_days: int):
    """Clean up local backup files older than retention period"""
    try:
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return
        
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        for file in backup_path.glob("ai_superstack_backup_*.tar.gz"):
            if file.stat().st_mtime < cutoff_time:
                file.unlink()
                logger.info(f"Deleted old backup: {file.name}")
                
    except Exception as e:
        logger.error(f"Error cleaning up local backups: {e}")

if __name__ == "__main__":
    retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", 180))
    backup_dir = "/app/data/backups"
    
    logger.info(f"Cleaning up backups older than {retention_days} days...")
    cleanup_local_backups(backup_dir, retention_days)
```

## 📄 .gitignore
```
# Environment variables
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
*.log

# Data and credentials
data/
credentials/
*.json

# Backups
backups/
*.tar.gz

# Monitoring data
prometheus-data/
grafana-data/
```

## 📄 .dockerignore
```
# Git
.git
.gitignore
README.md
*.md

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv

# IDE
.vscode
.idea

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.*

# Logs
*.log
logs/

# Test
tests/
pytest_cache/

# Documentation
docs/
```

## 📄 monitoring/prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
```

## 📄 Example API Usage Script (examples/test_routing.py)
```python
#!/usr/bin/env python3
"""
🧪 Test Intelligent Routing
Example script to test the AI Dev Superstack routing
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_routing():
    """Test various prompts to see routing decisions"""
    
    test_prompts = [
        {
            "category": "Coding",
            "prompt": "Write a Python function to implement quicksort algorithm with detailed comments"
        },
        {
            "category": "Analysis",
            "prompt": "Analyze the pros and cons of microservices vs monolithic architecture"
        },
        {
            "category": "Creative",
            "prompt": "Write a short story about a robot learning to paint"
        },
        {
            "category": "Math",
            "prompt": "Solve the integral of x^2 * sin(x) dx"
        },
        {
            "category": "Q&A",
            "prompt": "What is the capital of France?"
        },
        {
            "category": "Reasoning",
            "prompt": "Explain why the sky appears blue using principles of physics"
        }
    ]
    
    print("🧪 Testing Intelligent Routing\n")
    
    for test in test_prompts:
        print(f"📝 Category: {test['category']}")
        print(f"   Prompt: {test['prompt'][:50]}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"prompt": test['prompt']},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Routed to: {data['model_used']}")
                print(f"   ⏱️  Time: {data['processing_time']:.2f}s")
                print(f"   📊 Reason: {data.get('routing_reason', 'N/A')}")
                print(f"   💬 Response preview: {data['response'][:100]}...")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        print()
        time.sleep(1)  # Be nice to the API

def check_stats():
    """Check routing statistics"""
    print("\n📊 Routing Statistics\n")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/routing/stats")
        if response.status_code == 200:
            stats = response.json()
            print(json.dumps(stats, indent=2))
        else:
            print(f"Error getting stats: {response.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_routing()
    check_stats()
```

# 🚀 AI Dev Superstack - Complete Setup Guide

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Docker Desktop installed (with Docker Compose)
- ✅ 32GB RAM available
- ✅ At least 20GB free disk space
- ✅ API keys for OpenAI and Claude
- ✅ (Optional) HuggingFace API key
- ✅ (Optional) Google Cloud service account for backups

## 🔧 Step-by-Step Setup

### 1. Create Project Directory Structure

```bash
# Create main project directory
mkdir ai-dev-superstack
cd ai-dev-superstack

# Create all necessary subdirectories
mkdir -p fastapi/{services,models,routers,config}
mkdir -p data/conversations
mkdir -p credentials
mkdir -p monitoring
mkdir -p scripts
mkdir -p examples
```

### 2. Download All Files

Save all the files from the artifacts above into their respective directories:

- `docker-compose.yml` → Root directory
- `.env.example` → Root directory
- `Makefile` → Root directory
- `setup.sh` → Root directory
- `README.md` → Root directory
- All FastAPI files → `fastapi/` directory
- All script files → `scripts/` directory
- Configuration files → Their respective directories

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
```bash
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-claude-key-here
```

**Optional API Keys:**
```bash
HUGGINGFACE_API_KEY=hf_your-huggingface-key-here
# Leave commented if you don't have these yet:
# COHERE_API_KEY=your-cohere-key-here
# AI21_API_KEY=your-ai21-key-here
# etc...
```

### 4. Google Drive Backup Setup (Optional)

If you want automated backups to Google Drive:

#### a. Create a Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create a service account:
   ```
   IAM & Admin → Service Accounts → Create Service Account
   ```
5. Download the JSON key file
6. Save it as `credentials/service-account.json`

#### b. Set Up Google Drive Folder

1. Create a folder in Google Drive for backups
2. Share the folder with your service account email
3. Copy the folder ID from the URL:
   ```
   https://drive.google.com/drive/folders/[FOLDER_ID_HERE]
   ```
4. Add to `.env`:
   ```bash
   GOOGLE_DRIVE_FOLDER_ID=your-folder-id-here
   ```

### 5. Make Scripts Executable

```bash
chmod +x setup.sh
chmod +x scripts/*.py
```

### 6. Run the Setup Script

```bash
# Using the setup script
./setup.sh

# Or using Make
make setup
```

This will:
- ✅ Check Docker installation
- ✅ Create remaining directories
- ✅ Pull Docker images
- ✅ Build custom images
- ✅ Start all services
- ✅ Pull Llama3 and Mistral models

### 7. Verify Installation

```bash
# Check service status
make status

# View logs
make logs

# Test the API
make test
```

## 🧪 Testing the System

### Basic Health Check

```bash
curl http://localhost:8000/health | python -m json.tool
```

### Test Intelligent Routing

```bash
# Test a coding request
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list"}' \
  | python -m json.tool
```

### View Routing Statistics

```bash
make routing-stats
```

### Run Test Suite

Create `examples/test_routing.py` with the test script provided, then:

```bash
python examples/test_routing.py
```

## 📊 Understanding Intelligent Routing

The system routes requests based on:

1. **Task Classification**: Analyzes your prompt to determine the type
   - Coding → Prefers OpenAI or Claude
   - Analysis → Prefers Claude
   - Creative → Balanced selection
   - Simple Q&A → May use Ollama for efficiency

2. **Performance Metrics**: Tracks real-time performance
   - Success rates
   - Response times
   - Model availability

3. **Smart Selection**: Combines benchmarks with real performance

## 🛠️ Common Operations

### Adding New Models

```bash
# Pull a new Ollama model
make pull-model MODEL=codellama

# Pull specific model version
docker-compose exec ollama ollama pull llama2:13b
```

### Viewing Logs

```bash
# All logs
make logs

# Specific service
make logs-api

# Follow logs
docker-compose logs -f fastapi
```

### Manual Backup

```bash
make backup
```

### Stopping Services

```bash
# Stop all services
make stop

# Stop and remove volumes (careful!)
make clean
```

## 🔧 Troubleshooting

### Services Won't Start

```bash
# Check Docker
docker --version
docker-compose --version

# Check port availability
lsof -i :8000  # FastAPI
lsof -i :11434  # Ollama

# Reset everything
make clean
make setup
```

### API Keys Not Working

1. Check `.env` file formatting:
   ```bash
   # Correct
   OPENAI_API_KEY=sk-abcd1234
   
   # Wrong (no quotes needed)
   OPENAI_API_KEY="sk-abcd1234"
   ```

2. Restart services after changing:
   ```bash
   make restart
   ```

### Ollama Models Not Loading

```bash
# Check Ollama status
docker-compose exec ollama ollama list

# Manually pull models
docker-compose exec ollama ollama pull llama3
docker-compose exec ollama ollama pull mistral
```

### Out of Memory

1. Check Docker resources:
   - Docker Desktop → Settings → Resources
   - Allocate at least 16GB RAM

2. Reduce parallel models:
   - Edit `docker-compose.yml`
   - Adjust `OLLAMA_NUM_PARALLEL=1`

### Redis Connection Issues

```bash
# Check Redis
docker-compose exec redis redis-cli ping
# Should return: PONG
```

## 📈 Performance Tuning

### For Development (Resource-Conscious)

Current configuration is optimized for development with:
- CPU limits on containers
- Memory limits to prevent runaway usage
- Reduced model parallelism

### For Production

Modify `docker-compose.yml`:
```yaml
# Remove resource limits
deploy:
  resources:
    limits:
      cpus: '8'      # Increase
      memory: 32G    # Increase
```

### Model Selection Tuning

Edit `fastapi/config/model_benchmarks.py` to adjust model preferences for different tasks.

## 🔗 Useful Commands Reference

```bash
# Start/Stop
make start          # Start all services
make stop           # Stop all services
make restart        # Restart all services

# Monitoring
make status         # Check service health
make logs           # View all logs
make logs-api       # View API logs only
make routing-stats  # View routing statistics
make benchmarks     # View model benchmarks

# Models
make models         # List Ollama models
make pull-model MODEL=name  # Pull new model

# Maintenance
make backup         # Manual backup
make clean          # Clean everything
make test           # Run test request
```

## 🎯 Next Steps

1. **Test Different Prompts**: See how the router selects models
2. **Monitor Performance**: Check `make routing-stats` regularly
3. **Add More Providers**: Uncomment providers in `.env` as needed
4. **Customize Routing**: Modify `model_benchmarks.py` for your needs
5. **Set Up Backups**: Configure Google Drive for automated backups

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Model Library](https://ollama.ai/library)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Claude API Reference](https://docs.anthropic.com/)

---

## 🎉 Congratulations!

Your AI Dev Superstack is now ready! The intelligent routing system will automatically select the best model for each request based on performance data.

Start experimenting with different prompts to see the routing in action:

```bash
# Watch the logs in one terminal
make logs-api

# Send requests in another
make test
```

Happy coding! 🚀



