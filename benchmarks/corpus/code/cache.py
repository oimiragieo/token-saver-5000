"""Caching layer with Redis backend and in-process LRU fallback.

Provides a unified RedisCache interface. When Redis is unavailable, falls back
transparently to an LRU dict-based cache. Supports TTL, invalidation, and
namespace-based key isolation.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheStats:
    """Runtime statistics for a cache instance."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class _LRUFallback:
    """Simple LRU cache with per-entry TTL, used when Redis is unavailable."""

    def __init__(self, max_size: int = 1024) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if expires_at > 0 and time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_sec: int = 0) -> None:
        expires_at = time.time() + ttl_sec if ttl_sec > 0 else 0.0
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expires_at)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def keys_matching(self, pattern: str) -> list[str]:
        import fnmatch

        return [k for k in self._store if fnmatch.fnmatch(k, pattern)]

    def flush(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    def size(self) -> int:
        return len(self._store)


class RedisCache:
    """Cache client with Redis backend and transparent LRU fallback.

    If Redis is not available (or connection fails), all operations fall
    through to an in-process LRU cache without raising exceptions.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        namespace: str = "app",
        default_ttl_sec: int = 300,
        max_fallback_size: int = 4096,
    ) -> None:
        self._url = url
        self._namespace = namespace
        self._default_ttl = default_ttl_sec
        self._fallback = _LRUFallback(max_size=max_fallback_size)
        self._redis: Any = None  # would be redis.Redis in production
        self._using_fallback = True  # mock always uses fallback
        self._stats = CacheStats()

    def _make_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key.

        Args:
            key: Cache key (namespace is prepended automatically).

        Returns:
            Cached value or None if absent/expired.
        """
        full_key = self._make_key(key)
        value = self._fallback.get(full_key)
        if value is None:
            self._stats.misses += 1
        else:
            self._stats.hits += 1
        return value

    def set(self, key: str, value: Any, ttl_sec: int | None = None) -> None:
        """Store a value with optional TTL.

        Args:
            key: Cache key.
            value: Serialisable value to store.
            ttl_sec: Seconds until expiry; uses default_ttl_sec if None.
        """
        full_key = self._make_key(key)
        effective_ttl = ttl_sec if ttl_sec is not None else self._default_ttl
        self._fallback.set(full_key, value, effective_ttl)
        self._stats.sets += 1

    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if key existed, False otherwise.
        """
        full_key = self._make_key(key)
        deleted = self._fallback.delete(full_key)
        if deleted:
            self._stats.deletes += 1
        return deleted

    def invalidate(self, pattern: str) -> int:
        """Remove all keys matching a glob pattern within the namespace.

        Args:
            pattern: Glob pattern (e.g. "user:*").

        Returns:
            Number of keys removed.
        """
        full_pattern = self._make_key(pattern)
        keys = self._fallback.keys_matching(full_pattern)
        count = 0
        for k in keys:
            # Use internal key directly to avoid double-namespacing
            if self._fallback.delete(k):
                count += 1
                self._stats.deletes += 1
        return count

    def get_or_set(
        self,
        key: str,
        factory: Any,
        ttl_sec: int | None = None,
    ) -> Any:
        """Return cached value, or call factory() to populate cache.

        Args:
            key: Cache key.
            factory: Zero-argument callable that produces the value.
            ttl_sec: Optional TTL override.

        Returns:
            Cached or freshly-computed value.
        """
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl_sec)
        return value

    def flush_namespace(self) -> int:
        """Remove all keys in this cache's namespace.

        Returns:
            Number of keys removed.
        """
        return self.invalidate("*")

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics and health information."""
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": round(self._stats.hit_rate, 4),
            "sets": self._stats.sets,
            "deletes": self._stats.deletes,
            "using_fallback": self._using_fallback,
            "namespace": self._namespace,
            "size": self._fallback.size(),
        }

    def ping(self) -> bool:
        """Return True if the cache backend is reachable."""
        # Mock: fallback is always available
        return True
