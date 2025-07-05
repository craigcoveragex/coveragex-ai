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