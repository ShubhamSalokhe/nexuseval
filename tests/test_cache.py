"""
Unit tests for caching functionality.
"""

import pytest
import asyncio
from pathlib import Path
from nexuseval.cache import (
    InMemoryCache,
    FileCache,
    NoOpCache,
    CacheManager
)


class TestInMemoryCache:
    """Test in-memory cache implementation."""
    
    @pytest.mark.asyncio
    async def test_basic_set_get(self):
        """Test basic set and get operations."""
        cache = InMemoryCache(max_size=10)
        
        await cache.set("key1", {"value": "data1"})
        result = await cache.get("key1")
        
        assert result == {"value": "data1"}
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = InMemoryCache()
        result = await cache.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InMemoryCache(max_size=2)
        
        await cache.set("key1", {"value": "data1"})
        await cache.set("key2", {"value": "data2"})
        await cache.set("key3", {"value": "data3"})  # Should evict key1
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") == {"value": "data2"}
        assert await cache.get("key3") == {"value": "data3"}
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = InMemoryCache()
        
        await cache.set("key1", {"value": "data"}, ttl=1)
        
        # Should exist immediately
        result = await cache.get("key1")
        assert result == {"value": "data"}
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing cache."""
        cache = InMemoryCache()
        
        await cache.set("key1", {"value": "data1"})
        await cache.set("key2", {"value": "data2"})
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    def test_get_stats(self):
        """Test cache statistics."""
        cache = InMemoryCache(max_size=100)
        stats = cache.get_stats()
        
        assert stats["backend"] == "in_memory"
        assert stats["max_size"] == 100
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestFileCache:
    """Test file-based cache implementation."""
    
    @pytest.mark.asyncio
    async def test_basic_set_get(self, tmp_path):
        """Test basic set and get operations."""
        cache = FileCache(cache_dir=str(tmp_path / "cache"))
        
        await cache.set("key1", {"value": "data1"})
        result = await cache.get("key1")
        
        assert result == {"value": "data1"}
    
    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path):
        """Test that cache persists across cache instances."""
        cache_dir = tmp_path / "cache"
        
        # Set value with first instance
        cache1 = FileCache(cache_dir=str(cache_dir))
        await cache1.set("persistent_key", {"value": "persistent_data"})
        
        # Get value with second instance
        cache2 = FileCache(cache_dir=str(cache_dir))
        result = await cache2.get("persistent_key")
        
        assert result == {"value": "persistent_data"}
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, tmp_path):
        """Test TTL expiration for file cache."""
        cache = FileCache(cache_dir=str(tmp_path / "cache"))
        
        await cache.set("key1", {"value": "data"}, ttl=1)
        
        # Should exist immediately
        result = await cache.get("key1")
        assert result == {"value": "data"}
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self, tmp_path):
        """Test clearing file cache."""
        cache = FileCache(cache_dir=str(tmp_path / "cache"))
        
        await cache.set("key1", {"value": "data1"})
        await cache.set("key2", {"value": "data2"})
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestNoOpCache:
    """Test no-op cache (disabled caching)."""
    
    @pytest.mark.asyncio
    async def test_always_returns_none(self):
        """Test that NoOpCache never caches anything."""
        cache = NoOpCache()
        
        await cache.set("key1", {"value": "data"})
        result = await cache.get("key1")
        
        assert result is None
    
    def test_get_stats(self):
        """Test NoOpCache stats."""
        cache = NoOpCache()
        stats = cache.get_stats()
        
        assert stats["backend"] == "noop"
        assert stats["enabled"] is False


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_generate_key(self):
        """Test deterministic key generation."""
        key1 = CacheManager.generate_key(
            prompt="test prompt",
            model="gpt-4",
            temperature=0.5
        )
        key2 = CacheManager.generate_key(
            prompt="test prompt",
            model="gpt-4",
            temperature=0.5
        )
        key3 = CacheManager.generate_key(
            prompt="different prompt",
            model="gpt-4",
            temperature=0.5
        )
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self):
        """Test get_or_compute with cache hit."""
        cache = InMemoryCache()
        manager = CacheManager(cache)
        
        # Pre-populate cache
        await cache.set("test_key", {"result": "cached"})
        
        # Should return cached value without calling compute_fn
        compute_called = False
        
        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return {"result": "computed"}
        
        result = await manager.get_or_compute("test_key", compute_fn)
        
        assert result == {"result": "cached"}
        assert not compute_called
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self):
        """Test get_or_compute with cache miss."""
        cache = InMemoryCache()
        manager = CacheManager(cache)
        
        compute_called = False
        
        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return {"result": "computed"}
        
        result = await manager.get_or_compute("new_key", compute_fn)
        
        assert result == {"result": "computed"}
        assert compute_called
        
        # Value should now be cached
        cached = await cache.get("new_key")
        assert cached == {"result": "computed"}
