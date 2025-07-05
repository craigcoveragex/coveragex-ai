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