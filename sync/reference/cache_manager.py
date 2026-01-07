import redis
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Optional, Any, Dict, List
from functools import wraps
from webserver.storage import GCSFileStorage

logger = logging.getLogger(__name__)

class CacheManager:
    """Comprehensive cache manager for GCS files, database queries, and task results."""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
        
        self._gcs_storage = None  # Lazy load GCS storage
        
        # Cache TTLs (in seconds)
        self.FILE_CONTENT_TTL = 3600  # 1 hour
        self.METADATA_TTL = 300       # 5 minutes
        self.TASK_RESULT_TTL = 86400  # 24 hours
        self.QUERY_TTL = 300          # 5 minutes
    
    def _serialize_for_cache(self, obj: Any) -> str:
        """Serialize object for caching, handling UUIDs and other special types."""
        def json_serializer(obj):
            if isinstance(obj, uuid.UUID):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(obj, default=json_serializer)
    
    def _redis_available(self) -> bool:
        """Check if Redis is available."""
        return self.redis_client is not None
    
    @property
    def gcs_storage(self):
        if self._gcs_storage is None:
            self._gcs_storage = GCSFileStorage()
        return self._gcs_storage
    
    def _get_md5_hash(self, content: str) -> str:
        """Generate MD5 hash for content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate consistent cache key."""
        return f"{prefix}:{identifier}"
    

    # Processed PDB Caching (for MM/GBSA determinism)
    # No TTL - permanent cache since processed structures are stable
    
    def get_processed_pdb_cache_key(self, pdb_content_hash: str, ph: float = 7.0) -> str:
        """Generate cache key based on PDB content hash and processing parameters."""
        return f"mmgbsa:processed_pdb:{pdb_content_hash[:16]}:ph{ph}"
    
    def get_cached_processed_pdb(self, original_pdb_content: str, ph: float = 7.0) -> Optional[str]:
        """Get cached processed PDB content if available."""
        try:
            content_hash = self._get_md5_hash(original_pdb_content)
            cache_key = self.get_processed_pdb_cache_key(content_hash, ph)
            
            if self._redis_available():
                gcs_path = self.redis_client.get(cache_key)
                if gcs_path:
                    logger.info(f"Cache HIT for processed PDB: {cache_key}")
                    temp_path = self.gcs_storage.download_file(gcs_path, "/tmp/processed_pdb_cache")
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    os.unlink(temp_path)  # Clean up temp file
                    return content
            
            logger.info(f"Cache MISS for processed PDB: {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Error getting cached processed PDB: {e}")
            return None
    
    def cache_processed_pdb(self, original_pdb_content: str, processed_pdb_content: str, 
                           ph: float = 7.0) -> Optional[str]:
        """Cache processed PDB to GCS and track in Redis (permanent, no TTL)."""
        try:
            content_hash = self._get_md5_hash(original_pdb_content)
            cache_key = self.get_processed_pdb_cache_key(content_hash, ph)
            gcs_path = f"mmgbsa_cache/{content_hash[:16]}_ph{ph}_processed.pdb"
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False, encoding='utf-8') as tmp:
                tmp.write(processed_pdb_content)
                tmp_path = tmp.name
            
            self.gcs_storage.upload_file(tmp_path, gcs_path, content_type='text/plain')
            
            import os
            os.unlink(tmp_path)
            
            if self._redis_available():
                self.redis_client.set(cache_key, gcs_path)  # No TTL - permanent
                logger.info(f"Cached processed PDB to GCS: {gcs_path}")
            
            return gcs_path
        except Exception as e:
            logger.error(f"Error caching processed PDB: {e}")
            return None

    def invalidate_processed_pdb_cache(self, original_pdb_content: str, ph: float = 7.0) -> bool:
        """Invalidate cached processed PDB entry."""
        try:
            content_hash = self._get_md5_hash(original_pdb_content)
            cache_key = self.get_processed_pdb_cache_key(content_hash, ph)

            if self._redis_available():
                # Get GCS path before deleting
                gcs_path = self.redis_client.get(cache_key)

                # Delete Redis key
                self.redis_client.delete(cache_key)
                logger.info(f"Invalidated cache key: {cache_key}")

                # Optionally delete from GCS too
                if gcs_path:
                    try:
                        self.gcs_storage.delete_file(gcs_path)
                        logger.info(f"Deleted GCS file: {gcs_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete GCS file {gcs_path}: {e}")

                return True
            return False
        except Exception as e:
            logger.error(f"Error invalidating processed PDB cache: {e}")
            return False

    # File Content Caching
    def get_file_content(self, gcs_path: str) -> Optional[str]:
        """Get file content from cache or GCS."""
        if not self._redis_available():
            # If Redis is not available, try to get directly from GCS
            try:
                temp_path = self.gcs_storage.download_file(gcs_path, "/tmp/temp_download")
                with open(temp_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error getting file content for {gcs_path}: {e}")
                return None
                
        try:
            # Get file metadata first
            metadata = self.get_file_metadata(gcs_path)
            if not metadata:
                return None
            
            md5_hash = metadata.get('md5_hash')
            if not md5_hash:
                return None
            
            cache_key = self._get_cache_key("gcs_file", md5_hash)
            cached_content = self.redis_client.get(cache_key)
            
            if cached_content:
                logger.info(f"Cache HIT for file content: {gcs_path}")
                return cached_content
            
            # Cache miss - download from GCS
            logger.info(f"Cache MISS for file content: {gcs_path}")
            temp_path = self.gcs_storage.download_file(gcs_path, "/tmp/temp_download")
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache the content
            self.redis_client.setex(cache_key, self.FILE_CONTENT_TTL, content)
            logger.info(f"Cached file content: {gcs_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error getting file content for {gcs_path}: {e}")
            return None
    
    def cache_file_content(self, gcs_path: str, content: str) -> None:
        """Cache file content."""
        if not self._redis_available():
            return
            
        try:
            md5_hash = self._get_md5_hash(content)
            cache_key = self._get_cache_key("gcs_file", md5_hash)
            self.redis_client.setex(cache_key, self.FILE_CONTENT_TTL, content)
            logger.info(f"Cached file content: {gcs_path}")
        except Exception as e:
            logger.error(f"Error caching file content for {gcs_path}: {e}")
    
    # File Metadata Caching
    def get_file_metadata(self, gcs_path: str) -> Optional[Dict]:
        """Get file metadata from cache or GCS."""
        if not self._redis_available():
            # If Redis is not available, get directly from GCS
            try:
                return self.gcs_storage.get_file_metadata(gcs_path)
            except Exception as e:
                logger.error(f"Error getting file metadata for {gcs_path}: {e}")
                return None
                
        try:
            md5_hash = self._get_md5_hash(gcs_path)
            cache_key = self._get_cache_key("gcs_metadata", md5_hash)
            cached_metadata = self.redis_client.get(cache_key)
            
            if cached_metadata:
                logger.info(f"Cache HIT for file metadata: {gcs_path}")
                return json.loads(cached_metadata)
            
            # Cache miss - get from GCS
            logger.info(f"Cache MISS for file metadata: {gcs_path}")
            metadata = self.gcs_storage.get_file_metadata(gcs_path)
            
            if metadata:
                self.redis_client.setex(cache_key, self.METADATA_TTL, self._serialize_for_cache(metadata))
                logger.info(f"Cached file metadata: {gcs_path}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting file metadata for {gcs_path}: {e}")
            return None
    
    # Database Query Caching
    def cache_query(self, method: str, args: tuple, result: Any) -> None:
        """Cache database query results."""
        if not self._redis_available():
            return
            
        try:
            args_hash = self._get_md5_hash(str(args))
            cache_key = self._get_cache_key("file_query", f"{method}:{args_hash}")
            
            # Handle objects that have to_dict() method (like File objects)
            if isinstance(result, list) and result and hasattr(result[0], 'to_dict'):
                serializable_result = [obj.to_dict() for obj in result]
            elif hasattr(result, 'to_dict'):
                serializable_result = result.to_dict()
            else:
                serializable_result = result
                
            self.redis_client.setex(cache_key, self.QUERY_TTL, self._serialize_for_cache(serializable_result))
            logger.info(f"Cached query result: {method}")
        except Exception as e:
            logger.error(f"Error caching query result: {e}")
    
    def get_cached_query(self, method: str, args: tuple) -> Optional[Any]:
        """Get cached database query result."""
        if not self._redis_available():
            return None
            
        try:
            args_hash = self._get_md5_hash(str(args))
            cache_key = self._get_cache_key("file_query", f"{method}:{args_hash}")
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for query: {method}")
                deserialized = json.loads(cached_result)
                
                # Handle special cases where we need to reconstruct objects
                if method == "get_files_by_environment" and isinstance(deserialized, list):
                    # Reconstruct File objects from cached dictionaries
                    from webserver.model.file import File
                    return [File.from_row(file_dict) for file_dict in deserialized]
                elif method == "get_file" and isinstance(deserialized, dict):
                    # Reconstruct File object from cached dictionary
                    from webserver.model.file import File
                    return File.from_row(deserialized)
                else:
                    return deserialized
            
            logger.info(f"Cache MISS for query: {method}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached query: {e}")
            return None
    
    # Task Result Caching
    def get_task_result(self, filepath: str, task_id: str) -> Optional[Dict]:
        """Get cached task result."""
        if not self._redis_available():
            return None
            
        try:
            cache_key = self._get_cache_key("task_result", f"{filepath}:{task_id}")
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for task result: {task_id}")
                return json.loads(cached_result)
            
            logger.info(f"Cache MISS for task result: {task_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached task result: {e}")
            return None
    
    def cache_task_result(self, filepath: str, task_id: str, result: Dict) -> None:
        """Cache task result."""
        if not self._redis_available():
            return
            
        try:
            cache_key = self._get_cache_key("task_result", f"{filepath}:{task_id}")
            self.redis_client.setex(cache_key, self.TASK_RESULT_TTL, self._serialize_for_cache(result))
            logger.info(f"Cached task result: {task_id}")
        except Exception as e:
            logger.error(f"Error caching task result: {e}")
    
    # Smart Cache Invalidation
    def invalidate_file_caches(self, gcs_path: str) -> None:
        """Invalidate all caches related to a file."""
        try:
            # Invalidate file content cache
            metadata = self.get_file_metadata(gcs_path)
            if metadata and metadata.get('md5_hash'):
                content_cache_key = self._get_cache_key("gcs_file", metadata['md5_hash'])
                self.redis_client.delete(content_cache_key)
            
            # Invalidate metadata cache
            md5_hash = self._get_md5_hash(gcs_path)
            metadata_cache_key = self._get_cache_key("gcs_metadata", md5_hash)
            self.redis_client.delete(metadata_cache_key)
            
            # Invalidate related task results
            pattern = f"task_result:{gcs_path}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            
            logger.info(f"Invalidated caches for file: {gcs_path}")
            
        except Exception as e:
            logger.error(f"Error invalidating file caches: {e}")
    
    def invalidate_query_caches(self, method: str = None) -> None:
        """Invalidate database query caches."""
        try:
            if method:
                pattern = f"file_query:{method}:*"
            else:
                pattern = "file_query:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated query caches: {method or 'all'}")
                
        except Exception as e:
            logger.error(f"Error invalidating query caches: {e}")
    
    # Cache Statistics
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            stats = {
                'file_content_keys': len(self.redis_client.keys("gcs_file:*")),
                'metadata_keys': len(self.redis_client.keys("gcs_metadata:*")),
                'query_keys': len(self.redis_client.keys("file_query:*")),
                'task_result_keys': len(self.redis_client.keys("task_result:*")),
                'total_memory': self.redis_client.info()['used_memory_human']
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

# Global cache manager instance
cache_manager = CacheManager()

# Decorator for caching database queries
def cache_query_result(method: str):
    """Decorator to cache database query results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache first
            cached_result = cache_manager.get_cached_query(method, args)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.cache_query(method, args, result)
            return result
        return wrapper
    return decorator 