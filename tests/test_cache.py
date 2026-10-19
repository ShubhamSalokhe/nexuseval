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


@pytest.mark.asyncio
async def test_file_cache_update_does_not_evict_another_key(tmp_path):
    cache = FileCache(str(tmp_path / "cache"), max_size=2)
    await cache.set("old", {"value": 1})
    await cache.set("updated", {"value": 2})
    await cache.set("updated", {"value": 3})
    assert await cache.get("old") == {"value": 1}
    assert await cache.get("updated") == {"value": 3}


@pytest.mark.asyncio
async def test_file_cache_recovers_from_corrupt_metadata(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "_metadata.json").write_text('{"access_times":')
    cache = FileCache(str(cache_dir))
    await cache.set("key", {"value": 1})
    assert await cache.get("key") == {"value": 1}
    assert not list(cache_dir.glob("*.tmp"))


@pytest.mark.asyncio
async def test_file_cache_rejects_path_keys(tmp_path):
    cache = FileCache(str(tmp_path / "cache"))
    for key in ("../outside", "/absolute", "_metadata", ""):
        with pytest.raises(ValueError, match="Cache key"):
            await cache.set(key, {"value": 1})
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "file"])
async def test_configured_zero_ttl_expires_immediately(backend, tmp_path):
    from nexuseval import CacheConfig
    manager = CacheManager.from_config(CacheConfig(
        backend=backend, ttl=0, cache_dir=str(tmp_path / "nested" / "cache")
    ))
    await manager.get_or_compute("key", lambda: asyncio.sleep(0, result={"value": 1}))
    assert await manager.backend.get("key") is None


@pytest.mark.parametrize("cache_type", [InMemoryCache, FileCache])
@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_cache_capacity_is_validated(cache_type, size, tmp_path):
    kwargs = {"cache_dir": str(tmp_path / "cache")} if cache_type is FileCache else {}
    with pytest.raises(ValueError, match="positive integer"):
        cache_type(max_size=size, **kwargs)


def test_cache_factory_honors_disable_and_rejects_unimplemented_redis():
    from nexuseval import CacheConfig
    assert isinstance(CacheManager.from_config(CacheConfig(enabled=False)).backend, NoOpCache)
    with pytest.raises(NotImplementedError, match="Redis"):
        CacheManager.from_config(CacheConfig(backend="redis"))


@pytest.mark.asyncio
async def test_reopening_file_cache_with_smaller_capacity(tmp_path):
    cache_dir = str(tmp_path / "cache")
    original = FileCache(cache_dir, max_size=3)
    for i in range(3):
        await original.set(str(i), {"value": i})
    smaller = FileCache(cache_dir, max_size=1)
    await smaller.set("new", {"value": 4})
    assert smaller.get_stats()["size"] == 1
    assert await smaller.get("new") == {"value": 4}


@pytest.mark.asyncio
async def test_concurrent_cache_misses_compute_once():
    manager = CacheManager(InMemoryCache())
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"value": calls}

    results = await asyncio.gather(*(manager.get_or_compute("same", compute) for _ in range(10)))
    assert calls == 1
    assert results == [{"value": 1}] * 10


def test_persistent_cache_reused_across_event_loops(tmp_path):
    manager = CacheManager(FileCache(str(tmp_path / "cache")))

    async def run():
        return await asyncio.gather(*(manager.get_or_compute(
            "same", lambda: asyncio.sleep(0.001, result={"value": 1})
        ) for _ in range(5)))

    assert asyncio.run(run()) == asyncio.run(run()) == [{"value": 1}] * 5


@pytest.mark.asyncio
async def test_file_cache_instances_share_eviction_and_metadata(tmp_path):
    caches = [FileCache(str(tmp_path), max_size=2) for _ in range(4)]
    await caches[0].set("old", {"value": 1})
    await caches[0].set("other", {"value": 2})
    assert await caches[1].get("old") == {"value": 1}
    await caches[0].set("new", {"value": 3})
    assert await caches[1].get("old") == {"value": 1}
    assert await caches[1].get("other") is None

    await asyncio.gather(*(
        caches[index % len(caches)].set(str(index), {"value": index})
        for index in range(40)
    ))
    reopened = FileCache(str(tmp_path), max_size=2)
    assert reopened.get_stats()["size"] == 2
    assert set(reopened.metadata["access_times"]) == {
        path.stem for path in tmp_path.glob("*.json") if path.name != "_metadata.json"
    }


def _write_shared_cache(cache_dir, worker_id):
    async def write():
        cache = FileCache(cache_dir, max_size=3)
        for index in range(15):
            await cache.set(f"{worker_id}-{index}", {"value": index})
    asyncio.run(write())


def test_file_cache_is_safe_across_processes(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_write_shared_cache, str(tmp_path), index) for index in range(3)]
        for future in futures:
            future.result(timeout=30)
    cache = FileCache(str(tmp_path), max_size=3)
    assert cache.get_stats()["size"] == 3
    assert set(cache.metadata["access_times"]) == {
        path.stem for path in tmp_path.glob("*.json") if path.name != "_metadata.json"
    }
