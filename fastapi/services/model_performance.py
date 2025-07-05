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