# API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Chat Endpoint
**POST** `/api/v1/chat`

Intelligent chat endpoint that automatically routes to the best model.

**Request Body:**
```json
{
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "context": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"}
  ],
  "model": "openai",
  "user_id": "user123",
  "save_to_memory": true,
  "temperature": 0.7,
  "max_tokens": 4000
}
```

**Response:**
```json
{
  "response": "Here's a Python function for fibonacci...",
  "model_used": "openai",
  "processing_time": 1.23,
  "tokens_used": 156,
  "timestamp": 1704067200.0,
  "routing_reason": "Best performer for coding tasks"
}
```

### 2. Routing Statistics
**GET** `/api/v1/routing/stats`

Get comprehensive routing and performance statistics.

**Response:**
```json
{
  "models": {
    "openai": {
      "coding": {"count": 45, "success_count": 44, "total_time": 67.8},
      "analysis": {"count": 23, "success_count": 23, "total_time": 34.1}
    }
  },
  "task_types": {
    "coding": {
      "openai": {"count": 45, "success_rate": 0.98}
    }
  },
  "summary": {
    "total_requests": 150,
    "total_successes": 147,
    "avg_response_time": 1.45,
    "overall_success_rate": 0.98
  }
}
```

### 3. Model Benchmarks
**GET** `/api/v1/models/benchmarks`

Get current model benchmark scores for different task types.

**Response:**
```json
{
  "openai": {
    "coding": 0.95,
    "analysis": 0.90,
    "math": 0.92,
    "creative": 0.88,
    "reasoning": 0.94,
    "qa": 0.91,
    "translation": 0.89,
    "summarization": 0.90,
    "general": 0.91
  },
  "claude": {
    "coding": 0.93,
    "analysis": 0.95,
    "math": 0.88,
    "creative": 0.92,
    "reasoning": 0.96,
    "qa": 0.90,
    "translation": 0.87,
    "summarization": 0.94,
    "general": 0.92
  }
}
```

### 4. Routing Explanation
**POST** `/api/v1/routing/explain`

Get explanation for why a specific model would be chosen for a prompt.

**Request Body:**
```json
{
  "prompt": "Write a Python function to sort data"
}
```

**Response:**
```json
{
  "task_type": "coding",
  "model_scores": {
    "openai": {
      "final_score": 0.94,
      "benchmark_score": 0.95,
      "real_time_score": 0.93,
      "sample_size": 45
    },
    "claude": {
      "final_score": 0.92,
      "benchmark_score": 0.93,
      "real_time_score": 0.91,
      "sample_size": 38
    }
  },
  "recommendation": "openai",
  "reasoning": "For coding tasks, openai has the best performance with a score of 0.94"
}
```

### 5. Health Check
**GET** `/health`

Check the health status of all services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200.0,
  "services": {
    "openai": "healthy",
    "claude": "healthy",
    "huggingface": "unavailable",
    "ollama": "healthy"
  }
}
```

### 6. Root Information
**GET** `/`

Get API information and feature status.

**Response:**
```json
{
  "message": "🧠 AI Dev Superstack - Intelligent Routing Edition",
  "version": "2.0.0",
  "docs": "/docs",
  "features": {
    "intelligent_routing": "Active",
    "performance_tracking": "Active",
    "active_providers": ["openai", "claude", "huggingface", "ollama"],
    "backup_retention": "180 days"
  }
}
```

## Task Types

The intelligent router classifies prompts into these task types:

- **coding**: Programming, debugging, algorithms
- **analysis**: Comparing, evaluating, reviewing
- **math**: Mathematical calculations and problems
- **creative**: Writing, storytelling, content creation
- **reasoning**: Logical thinking, explanations
- **qa**: Questions and answers, facts
- **translation**: Language translation
- **summarization**: Text summarization
- **general**: Default category for unmatched prompts

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `400`: Bad Request - Invalid input
- `500`: Internal Server Error - Service error
- `503`: Service Unavailable - No models available

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation.