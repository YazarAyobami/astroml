"""Graph computation cache for repeated graph outputs — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features, node
features) per data version and window to avoid recomputation across
experiments.  Supports both in-memory (default) and Redis backends.

Two families of API are exposed, and both are kept because callers and tests
already depend on each:

* ``get`` / ``set`` / ``invalidate`` take an explicit ``prefix`` and ``key``
  and are served by the configured in-memory store, or directly by Redis when
  ``config.backend`` is :attr:`GraphCacheBackend.REDIS`.
* ``get_adjacency`` / ``set_adjacency`` / ``get_edge_features`` /
  ``set_edge_features`` derive a stable key from the window parameters and are
  served by the in-process LRU sitting in front of the shared
  :class:`~astroml.cache.redis_cache.RedisCache` layer.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from astroml.cache.redis_cache import CacheKeyPrefix, RedisCache

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ADJACENCY_PREFIX = CacheKeyPrefix.GRAPH_WINDOW
_EDGE_FEATURE_PREFIX = CacheKeyPrefix.GRAPH_SNAPSHOT

# Default in-process LRU capacity (number of entries, not bytes).  Used when
# ``_LRUCache`` is constructed directly and as the fallback for
# ``GraphComputationCache`` when no config-driven capacity is available.
_DEFAULT_LRU_CAPACITY = 128

# Guards creation of the ``GraphComputationCache`` singleton.
_singleton_lock = threading.Lock()


class GraphCacheBackend(Enum):
    """Backend for graph computation cache."""

    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class GraphCacheConfig:
    """Configuration for graph computation cache."""

    backend: GraphCacheBackend = GraphCacheBackend.MEMORY
    max_size: int = 512
    default_ttl_seconds: int = 3600  # 1 hour
    redis_url: str = "redis://localhost:6379"
    # Per-prefix TTL overrides (seconds)
    adjacency_ttl: int = 3600
    edge_feature_ttl: int = 1800
    node_feature_ttl: int = 1800
    snapshot_ttl: int = 3600


@dataclass
class GraphCacheStats:
    """Graph cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


def _window_key(data_version: str, start_ts: int, end_ts: int, extra: str = "") -> str:
    """Stable cache key from window parameters."""
    payload = f"{data_version}:{start_ts}:{end_ts}:{extra}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return digest


class _LRUCache:
    """Minimal thread-unsafe in-process LRU backed by an OrderedDict."""

    def __init__(self, capacity: int = _DEFAULT_LRU_CAPACITY) -> None:
        self._cap = max(1, capacity)
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._cap:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class _MemoryGraphStore:
    """Thread-safe in-memory LRU cache for graph computations."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expires_at)
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        import time

        with self._lock:
            if key not in self._data:
                return None
            value, expires_at = self._data[key]
            if expires_at is not None and time.time() > expires_at:
                del self._data[key]
                self._access_order.remove(key)
                return None
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        import time

        with self._lock:
            if key in self._data:
                self._access_order.remove(key)
            elif len(self._data) >= self._max_size:
                # Evict LRU
                oldest = self._access_order.pop(0)
                del self._data[oldest]

            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._data[key] = (value, expires_at)
            self._access_order.append(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            if not prefix:
                count = len(self._data)
                self._data.clear()
                self._access_order.clear()
                return count
            keys_to_remove = [k for k in self._data if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._data[k]
                self._access_order.remove(k)
            return len(keys_to_remove)

    def size(self) -> int:
        with self._lock:
            return len(self._data)


class GraphComputationCache:
    """Two-level cache for graph computation results.

    Caches adjacency lists, edge features, node features and other
    intermediate graph outputs keyed by data version and window bounds, so
    experiments that share the same data slice reuse the cached result.

    Args:
        config: :class:`GraphCacheConfig` controlling the backend, the maximum
            number of entries held in memory, and the per-prefix TTLs.
        redis_ttl_adjacency: Redis TTL for adjacency entries in seconds
            (default 30 minutes).
        redis_ttl_edge_features: Redis TTL for edge feature entries in seconds
            (default 1 hour).
        lru_capacity: Number of entries to keep in the in-process LRU.
            Defaults to ``config.max_size``.

    Example::

        cache = GraphComputationCache()

        adj = cache.get_adjacency("v1.2", start_ts=1_000_000, end_ts=1_010_000)
        if adj is None:
            adj = build_adjacency(edges, start_ts, end_ts)
            cache.set_adjacency("v1.2", 1_000_000, 1_010_000, adj)

        @cache.cached_adjacency(version="v3", window="7d")
        def build_adjacency_for(window_edges):
            ...
    """

    _instance: GraphComputationCache | None = None

    def __new__(
        cls, config: GraphCacheConfig | None = None, **kwargs: Any
    ) -> GraphComputationCache:
        # ``GraphComputationCache()`` is a process-wide singleton, matching the
        # RedisCache layer it wraps.  Constructing with an explicit config or an
        # explicit tuning argument returns an independent instance, so callers
        # can hold several caches with different backends or capacities side by
        # side.
        if config is None and not kwargs:
            if cls._instance is None:
                with _singleton_lock:
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                        cls._instance._initialized = False
            return cls._instance
        instance = super().__new__(cls)
        instance._initialized = False
        return instance

    def __init__(
        self,
        config: GraphCacheConfig | None = None,
        redis_ttl_adjacency: int = 1_800,
        redis_ttl_edge_features: int = 3_600,
        lru_capacity: int | None = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self.config = config or GraphCacheConfig()
        self._stats = GraphCacheStats()
        self._store: _MemoryGraphStore | None = None
        self._redis_client = None
        self._lru: _LRUCache = _LRUCache(capacity=lru_capacity or self.config.max_size)
        self._redis: RedisCache = RedisCache()
        self._ttl_adj = redis_ttl_adjacency
        self._ttl_ef = redis_ttl_edge_features

        if self.config.backend == GraphCacheBackend.REDIS:
            try:
                import redis

                self._redis_client = redis.from_url(self.config.redis_url)
                self._redis_client.ping()
            except Exception as e:
                logger.warning("Redis unavailable for graph cache, falling back to memory: %s", e)
                self.config.backend = GraphCacheBackend.MEMORY

        if self.config.backend == GraphCacheBackend.MEMORY:
            self._store = _MemoryGraphStore(self.config.max_size)

        self._initialized = True

    @staticmethod
    def _hash_args(*args: Any, **kwargs: Any) -> str:
        """Generate a deterministic hash from function arguments."""
        parts: list[str] = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                parts.append(f"list:{len(arg)}")
            elif isinstance(arg, dict):
                parts.append(f"dict:{len(arg)}")
            else:
                parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}:{v}")
        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    # Generic prefix:key access
    # ------------------------------------------------------------------ #

    def get(self, prefix: str, key: str) -> Any | None:
        full_key = f"{prefix}:{key}"
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                data = self._redis_client.get(full_key)
                if data is not None:
                    self._stats.hits += 1
                    return _pickle.loads(data)
                self._stats.misses += 1
                return None
            except Exception as e:
                logger.warning("Redis graph cache GET error: %s", e)
                self._stats.misses += 1
                return None
        else:
            value = self._store.get(full_key)  # type: ignore[union-attr]
            if value is not None:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return value

    def set(self, prefix: str, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        full_key = f"{prefix}:{key}"
        ttl = ttl_seconds or self.config.default_ttl_seconds
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                self._redis_client.setex(full_key, ttl, _pickle.dumps(value))
                self._stats.sets += 1
            except Exception as e:
                logger.warning("Redis graph cache SET error: %s", e)
        else:
            self._store.set(full_key, value, ttl)  # type: ignore[union-attr]
            self._stats.sets += 1

    def invalidate(self, prefix: str, key: str | None = None) -> int:
        if key:
            full_key = f"{prefix}:{key}"
            if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
                try:
                    return 1 if self._redis_client.delete(full_key) else 0
                except Exception:
                    return 0
            else:
                return 1 if self._store.delete(full_key) else 0  # type: ignore[union-attr]
        else:
            pattern = f"{prefix}:*"
            if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
                try:
                    keys = self._redis_client.keys(pattern)
                    if keys:
                        return self._redis_client.delete(*keys)
                    return 0
                except Exception:
                    return 0
            else:
                return self._store.clear(prefix)  # type: ignore[union-attr]

    def clear(self) -> int:
        """Clear all entries from the graph computation cache and reset statistics."""
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                keys = self._redis_client.keys("graph:*")
                count = len(keys)
                if keys:
                    self._redis_client.delete(*keys)
                self.reset_stats()
                return count
            except Exception:
                self.reset_stats()
                return 0
        else:
            count = self._store.clear("") if self._store else 0
            self.reset_stats()
            return count

    def get_stats(self) -> GraphCacheStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = GraphCacheStats()

    # ------------------------------------------------------------------ #
    # Adjacency list caching (window-parameterised)
    # ------------------------------------------------------------------ #

    def get_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> Any | None:
        """Return a cached adjacency structure or ``None`` on miss."""
        key = self._adj_key(data_version, start_ts, end_ts)
        hit = self._lru.get(key)
        if hit is not None:
            logger.debug("GraphComputationCache: adjacency LRU hit for %s", key[:12])
            return hit
        value = self._redis.get(key)
        if value is not None:
            logger.debug("GraphComputationCache: adjacency Redis hit for %s", key[:12])
            self._lru.set(key, value)
        return value

    def set_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        adjacency: Any,
    ) -> None:
        """Store an adjacency structure in both cache levels."""
        key = self._adj_key(data_version, start_ts, end_ts)
        self._lru.set(key, adjacency)
        self._redis.set(key, adjacency, ttl_seconds=self._ttl_adj)

    def invalidate_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> None:
        """Evict an adjacency entry from both cache levels."""
        key = self._adj_key(data_version, start_ts, end_ts)
        self._lru.invalidate(key)
        self._redis.delete(key)

    # ------------------------------------------------------------------ #
    # Edge feature caching (window-parameterised)
    # ------------------------------------------------------------------ #

    def get_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> Any | None:
        """Return cached edge features or ``None`` on miss."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        hit = self._lru.get(key)
        if hit is not None:
            logger.debug("GraphComputationCache: edge_features LRU hit for %s", key[:12])
            return hit
        value = self._redis.get(key)
        if value is not None:
            logger.debug("GraphComputationCache: edge_features Redis hit for %s", key[:12])
            self._lru.set(key, value)
        return value

    def set_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        features: Any,
        feature_set: str = "default",
    ) -> None:
        """Store edge features in both cache levels."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        self._lru.set(key, features)
        self._redis.set(key, features, ttl_seconds=self._ttl_ef)

    def invalidate_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> None:
        """Evict edge features from both cache levels."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        self._lru.invalidate(key)
        self._redis.delete(key)

    def invalidate_version(self, data_version: str) -> None:
        """Evict all LRU entries (Redis entries expire naturally via TTL)."""
        self._lru.clear()
        logger.info("GraphComputationCache: LRU cleared on invalidate_version(%s)", data_version)

    @property
    def lru_size(self) -> int:
        """Number of entries currently in the in-process LRU."""
        return len(self._lru)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _adj_key(self, version: str, start: int, end: int) -> str:
        digest = _window_key(version, start, end)
        return f"{_ADJACENCY_PREFIX.value}:adj:{digest}"

    def _ef_key(self, version: str, start: int, end: int, feature_set: str) -> str:
        digest = _window_key(version, start, end, feature_set)
        return f"{_EDGE_FEATURE_PREFIX.value}:ef:{digest}"

    # -- Convenience decorators -----------------------------------------------

    def cached_adjacency(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache adjacency list computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"adj:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:adjacency", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:adjacency",
                    cache_key,
                    result,
                    ttl_seconds or self.config.adjacency_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def cached_edge_features(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache edge feature computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"ef:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:edge_features", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:edge_features",
                    cache_key,
                    result,
                    ttl_seconds or self.config.edge_feature_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def cached_node_features(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache node feature computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"nf:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:node_features", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:node_features",
                    cache_key,
                    result,
                    ttl_seconds or self.config.node_feature_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------


def get_graph_cache(config: GraphCacheConfig | None = None) -> GraphComputationCache:
    """Get or create the singleton graph computation cache."""
    return GraphComputationCache(config)


def invalidate_graph_cache(prefix: str = "", key: str | None = None) -> int:
    """Invalidate graph cache entries.

    Args:
        prefix: Cache prefix (e.g. ``'graph:adjacency'``). Empty string clears all.
        key: Specific key within prefix. ``None`` clears all for the prefix.

    Returns:
        Number of entries invalidated.
    """
    cache = get_graph_cache()
    if prefix:
        return cache.invalidate(prefix, key)
    count = 0
    for p in ("graph:adjacency", "graph:edge_features", "graph:node_features"):
        count += cache.invalidate(p)
    return count


def cached_graph_computation(
    data_version_arg: str = "data_version",
    start_ts_arg: str = "start_ts",
    end_ts_arg: str = "end_ts",
    cache: GraphComputationCache | None = None,
    ttl_seconds: int = 1_800,
):
    """Decorator that caches graph computation outputs per data version and window.

    The decorated function must accept ``data_version``, ``start_ts``, and
    ``end_ts`` keyword arguments (or positional args whose names match
    ``data_version_arg``, ``start_ts_arg``, ``end_ts_arg``).

    Example::

        @cached_graph_computation()
        def build_adjacency(data_version: str, start_ts: int, end_ts: int):
            ...  # expensive graph construction
    """
    _cache = cache or get_graph_cache()

    def decorator(func):  # type: ignore[no-untyped-def]
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            version = kwargs.get(data_version_arg, "unknown")
            start = kwargs.get(start_ts_arg, 0)
            end = kwargs.get(end_ts_arg, 0)

            key = _window_key(str(version), int(start), int(end), func.__name__)
            full_key = f"graph:computation:{key}"

            cached = _cache._lru.get(full_key)
            if cached is not None:
                return cached

            cached = _cache._redis.get(full_key)
            if cached is not None:
                _cache._lru.set(full_key, cached)
                return cached

            result = func(*args, **kwargs)
            _cache._lru.set(full_key, result)
            _cache._redis.set(full_key, result, ttl_seconds=ttl_seconds)
            return result

        return wrapper

    return decorator
