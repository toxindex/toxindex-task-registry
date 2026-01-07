#!/usr/bin/env python3
"""Test script to verify GCS and Redis caching setup."""

import os
import sys
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent.parent.parent.parent / ".env"
if env_file.exists():
    print(f"Loading environment from: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Strip quotes from value
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
else:
    print(f"Warning: .env file not found at {env_file}")

def test_redis():
    """Test Redis connectivity."""
    print("\n" + "="*50)
    print("Testing Redis Connection...")
    print("="*50)

    try:
        import redis

        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))

        print(f"  Connecting to {host}:{port}...")
        client = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=5)
        client.ping()
        print("  ✓ Redis connection successful!")

        # Test set/get
        client.set("test_key", "test_value")
        value = client.get("test_key")
        client.delete("test_key")

        if value == "test_value":
            print("  ✓ Redis read/write working!")
            return True
        else:
            print("  ✗ Redis read/write failed")
            return False

    except ImportError:
        print("  ✗ redis package not installed: pip install redis")
        return False
    except Exception as e:
        print(f"  ✗ Redis error: {e}")
        return False


def test_gcs():
    """Test GCS connectivity."""
    print("\n" + "="*50)
    print("Testing GCS Connection...")
    print("="*50)

    try:
        from google.cloud import storage

        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not bucket_name:
            print("  ✗ GCS_BUCKET_NAME not set in environment")
            return False

        print(f"  Bucket: {bucket_name}")
        print(f"  Credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'default/application-default')}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if bucket.exists():
            print("  ✓ GCS bucket exists and accessible!")
        else:
            print(f"  ✗ Bucket {bucket_name} does not exist")
            return False

        # Test upload/download
        import tempfile
        test_content = "test_content_for_cache_verification"
        test_path = "mmgbsa_cache/_test_verification.txt"

        # Upload
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        blob = bucket.blob(test_path)
        blob.upload_from_filename(tmp_path)
        os.unlink(tmp_path)
        print("  ✓ GCS upload working!")

        # Download
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = tmp.name

        blob.download_to_filename(tmp_path)
        with open(tmp_path, 'r') as f:
            downloaded = f.read()
        os.unlink(tmp_path)

        if downloaded == test_content:
            print("  ✓ GCS download working!")
        else:
            print("  ✗ GCS download content mismatch")
            return False

        # Cleanup
        blob.delete()
        print("  ✓ GCS delete working!")

        return True

    except ImportError:
        print("  ✗ google-cloud-storage not installed: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"  ✗ GCS error: {e}")
        return False


def test_cache_manager():
    """Test CacheManager integration."""
    print("\n" + "="*50)
    print("Testing CacheManager Integration...")
    print("="*50)

    try:
        # Add webserver path to sys.path
        webserver_path = Path(__file__).parent.parent.parent.parent / "sync" / "reference"
        sys.path.insert(0, str(webserver_path))

        # Mock the webserver.storage import
        import importlib.util
        spec = importlib.util.spec_from_file_location("storage", webserver_path / "storage.py")
        storage_module = importlib.util.module_from_spec(spec)
        sys.modules['webserver'] = type(sys)('webserver')
        sys.modules['webserver.storage'] = storage_module
        spec.loader.exec_module(storage_module)

        # Now import cache_manager
        spec = importlib.util.spec_from_file_location("cache_manager", webserver_path / "cache_manager.py")
        cache_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cache_module)

        cache_manager = cache_module.cache_manager

        if cache_manager.redis_client is None:
            print("  ⚠ CacheManager initialized but Redis unavailable (cache disabled)")
            return False

        print("  ✓ CacheManager initialized with Redis!")

        # Test processed PDB caching
        test_pdb_content = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
        test_processed = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H"

        # Cache
        gcs_path = cache_manager.cache_processed_pdb(test_pdb_content, test_processed, ph=7.0)
        if gcs_path:
            print(f"  ✓ Cached processed PDB to: {gcs_path}")
        else:
            print("  ✗ Failed to cache processed PDB")
            return False

        # Retrieve
        cached = cache_manager.get_cached_processed_pdb(test_pdb_content, ph=7.0)
        if cached == test_processed:
            print("  ✓ Retrieved cached PDB successfully!")
        else:
            print("  ✗ Retrieved content doesn't match")
            return False

        # Cleanup - delete from Redis and GCS
        content_hash = cache_manager._get_md5_hash(test_pdb_content)
        cache_key = cache_manager.get_processed_pdb_cache_key(content_hash, ph=7.0)
        cache_manager.redis_client.delete(cache_key)
        cache_manager.gcs_storage.delete_file(gcs_path)
        print("  ✓ Cleanup successful!")

        return True

    except Exception as e:
        import traceback
        print(f"  ✗ CacheManager error: {e}")
        traceback.print_exc()
        return False


def main():
    print("="*50)
    print("GCS Cache Setup Verification")
    print("="*50)

    results = {}

    results['redis'] = test_redis()
    results['gcs'] = test_gcs()

    if results['redis'] and results['gcs']:
        results['cache_manager'] = test_cache_manager()
    else:
        print("\n⚠ Skipping CacheManager test (requires both Redis and GCS)")
        results['cache_manager'] = False

    print("\n" + "="*50)
    print("Summary")
    print("="*50)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! GCS caching is ready to use.")
    else:
        print("Some tests failed. Please check the configuration above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
