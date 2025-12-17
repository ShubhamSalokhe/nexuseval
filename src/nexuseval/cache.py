"""
Caching layer for NexusEval to reduce redundant LLM API calls.

This module provides abstract cache interface and multiple backend implementations:
- In-memory cache (LRU-based, fast but not persistent)
- File-based cache (persistent, good for development)
- Redis cache (distributed, for production - optional dependency)
"""

import json
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from collections import OrderedDict
import asyncio

class CacheBackend(ABC):
    """
    Abstract base class for cache backends.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        """Store value in cache with optional TTL (seconds)."""
        pass
    
    @abstractmethod
    async def clear(self):
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """
    In-memory LRU cache implementation.
    Fast but not persistent across restarts.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum number of entries to cache
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl_map: Dict[str, float] = {}  # key -> expiration timestamp
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve value from cache."""
        async with self._lock:
            # Check if key exists and not expired
            if key in self.cache:
                # Check TTL
                if key in self.ttl_map and time.time() > self.ttl_map[key]:
                    # Expired
                    del self.cache[key]
                    del self.ttl_map[key]
                    self.misses += 1
                    return None
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        """Store value in cache."""
        async with self._lock:
            # Add or update
            self.cache[key] = value
            self.cache.move_to_end(key)
            
            # Set TTL if specified
            if ttl:
                self.ttl_map[key] = time.time() + ttl
            elif key in self.ttl_map:
                del self.ttl_map[key]
            
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.ttl_map:
                    del self.ttl_map[oldest_key]
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.ttl_map.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "backend": "in_memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class FileCache(CacheBackend):
    """
    File-based persistent cache.
    Good for development and single-machine deployments.
    """
    
    def __init__(self, cache_dir: str = ".nexuseval_cache", max_size: int = 1000):
        """
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
        
        # Metadata file for tracking
        self.metadata_file = self.cache_dir / "_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"access_times": {}}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve value from cache."""
        async with self._lock:
            cache_path = self._get_cache_path(key)
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check TTL
                    if "expires_at" in data and time.time() > data["expires_at"]:
                        cache_path.unlink()  # Delete expired
                        self.misses += 1
                        return None
                    
                    # Update access time
                    self.metadata["access_times"][key] = time.time()
                    self._save_metadata()
                    
                    self.hits += 1
                    return data.get("value")
                except Exception:
                    self.misses += 1
                    return None
            else:
                self.misses += 1
                return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        """Store value in cache."""
        async with self._lock:
            # Evict oldest if at capacity
            await self._evict_if_needed()
            
            cache_path = self._get_cache_path(key)
            
            data = {"value": value}
            if ttl:
                data["expires_at"] = time.time() + ttl
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            # Update metadata
            self.metadata["access_times"][key] = time.time()
            self._save_metadata()
    
    async def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_files = [f for f in cache_files if f.name != "_metadata.json"]
        
        if len(cache_files) >= self.max_size:
            # Remove oldest by access time
            access_times = self.metadata.get("access_times", {})
            cache_files_with_times = [
                (f, access_times.get(f.stem, 0))
                for f in cache_files
            ]
            cache_files_with_times.sort(key=lambda x: x[1])
            
            # Remove oldest
            oldest_file, _ = cache_files_with_times[0]
            oldest_file.unlink()
            if oldest_file.stem in access_times:
                del access_times[oldest_file.stem]
                self._save_metadata()
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.metadata = {"access_times": {}}
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_files = [f for f in cache_files if f.name != "_metadata.json"]
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "backend": "file",
            "cache_dir": str(self.cache_dir),
            "size": len(cache_files),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class NoOpCache(CacheBackend):
    """
    No-op cache that doesn't actually cache anything.
    Useful for disabling caching.
    """
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        pass
    
    async def clear(self):
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {"backend": "noop", "enabled": False}


class CacheManager:
    """
    High-level cache manager with key generation utilities.
    """
    
    def __init__(self, backend: CacheBackend):
        """
        Args:
            backend: Cache backend implementation
        """
        self.backend = backend
    
    @staticmethod
    def generate_key(prompt: str, model: str, **kwargs) -> str:
        """
        Generate cache key from prompt and model parameters.
        
        Args:
            prompt: LLM prompt
            model: Model name
            **kwargs: Additional parameters (temperature, etc.)
        
        Returns:
            Cache key (SHA256 hash)
        """
        # Create deterministic string from inputs
        key_data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Hash to get fixed-length key
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get value from cache or compute it.
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            ttl: Time to live in seconds
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.backend.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        value = await compute_fn()
        
        # Store in cache
        await self.backend.set(key, value, ttl)
        
        return value
    
    async def clear(self):
        """Clear all cache entries."""
        await self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.backend.get_stats()
