import os
import logging
import tempfile
from pathlib import Path
from typing import Optional
from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)

class GCSFileStorage:
    """Google Cloud Storage file storage implementation."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize GCS storage.
        
        Args:
            bucket_name: GCS bucket name. If None, uses GCS_BUCKET_NAME env var.
        """
        self.bucket_name = bucket_name or os.environ.get('GCS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable must be set")
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        # Ensure bucket exists
        if not self.bucket.exists():
            logger.info(f"Creating bucket: {self.bucket_name}")
            self.bucket.create()
    
    def upload_file(self, local_path: str, gcs_path: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_path: Destination path in GCS
            content_type: MIME type of the file
            
        Returns:
            GCS path of uploaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            
            if content_type:
                blob.content_type = content_type
            
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to GCS: {e}")
            raise
    
    def upload_fileobj(self, file_obj, gcs_path: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file object to GCS.
        
        Args:
            file_obj: File-like object to upload
            gcs_path: Destination path in GCS
            content_type: MIME type of the file
            
        Returns:
            GCS path of uploaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            
            if content_type:
                blob.content_type = content_type
            
            blob.upload_from_file(file_obj)
            logger.info(f"Uploaded file object to gs://{self.bucket_name}/{gcs_path}")
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload file object to GCS: {e}")
            raise
    
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """
        Download a file from GCS.
        
        Args:
            gcs_path: Source path in GCS
            local_path: Destination local path
            
        Returns:
            Local path of downloaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
            return local_path
            
        except NotFound:
            logger.error(f"File not found in GCS: gs://{self.bucket_name}/{gcs_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            raise
    
    def delete_file(self, gcs_path: str) -> bool:
        """
        Delete a file from GCS.
        
        Args:
            gcs_path: Path in GCS to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{gcs_path}")
            return True
            
        except NotFound:
            logger.warning(f"File not found for deletion: gs://{self.bucket_name}/{gcs_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete from GCS: {e}")
            raise
    
    def file_exists(self, gcs_path: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            gcs_path: Path in GCS to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check file existence in GCS: {e}")
            return False
    
    def generate_download_url(self, gcs_path: str, expiration: int = 3600) -> str:
        """
        Generate a signed download URL for a file.
        
        Args:
            gcs_path: Path in GCS
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Signed download URL
        """
        try:
            blob = self.bucket.blob(gcs_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate download URL for {gcs_path}: {e}")
            raise
    
    def get_file_metadata(self, gcs_path: str) -> Optional[dict]:
        """
        Get metadata for a file in GCS.
        
        Args:
            gcs_path: Path in GCS
            
        Returns:
            File metadata dict or None if not found
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.reload()
            return {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_type,
                'created': blob.time_created.isoformat() if blob.time_created else None,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'md5_hash': blob.md5_hash
            }
        except NotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for {gcs_path}: {e}")
            return None

# Legacy S3FileStorage class for backward compatibility
class S3FileStorage:
    """Legacy S3 storage class - now redirects to GCS."""
    
    def __init__(self):
        self.gcs_storage = GCSFileStorage()
    
    def upload_file(self, local_path: str, s3_key: str, content_type: Optional[str] = None) -> str:
        """Upload file to GCS (using S3-style key as GCS path)."""
        return self.gcs_storage.upload_file(local_path, s3_key, content_type)
    
    def generate_download_url(self, s3_key: str) -> str:
        """Generate download URL for GCS file."""
        return self.gcs_storage.generate_download_url(s3_key) 