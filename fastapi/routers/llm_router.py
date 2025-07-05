"""
🧠 Intelligent LLM Router
Routes requests to the best-performing model based on task type and historical performance
"""

import logging
from typing import Dict, Tuple, Optional, List
import re
from fastapi import HTTPException
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