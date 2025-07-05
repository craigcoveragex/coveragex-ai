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