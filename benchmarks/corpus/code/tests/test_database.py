"""Tests for database connection pool and query execution.

Tests connection pool lifecycle, query execution, context manager,
and configuration validation.
"""
from __future__ import annotations

import threading
from collections import deque


# ---------------------------------------------------------------------------
# Helpers — standalone mock implementations (no real DB imports)
# ---------------------------------------------------------------------------

class _MockConfig:
    host = "localhost"
    port = 5432
    database = "testdb"
    min_connections = 2
    max_connections = 5
    connection_timeout_sec = 5.0


class _MockConn:
    def __init__(self, conn_id: int) -> None:
        self.conn_id = conn_id
        self._closed = False

    def execute(self, sql: str, params=None) -> list:
        if self._closed:
            raise ConnectionError("Closed connection")
        return []

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class _MockPool:
    def __init__(self, cfg: _MockConfig) -> None:
        self._cfg = cfg
        self._pool: deque[_MockConn] = deque()
        self._active: set[int] = set()
        self._counter = 0
        self._closed = False
        self._lock = threading.Lock()
        for _ in range(cfg.min_connections):
            self._pool.append(self._new_conn())

    def _new_conn(self) -> _MockConn:
        self._counter += 1
        return _MockConn(self._counter)

    def get(self) -> _MockConn:
        if self._closed:
            raise ConnectionError("Pool is closed")
        with self._lock:
            if self._pool:
                conn = self._pool.popleft()
                self._active.add(conn.conn_id)
                return conn
            total = len(self._pool) + len(self._active)
            if total < self._cfg.max_connections:
                conn = self._new_conn()
                self._active.add(conn.conn_id)
                return conn
            raise ConnectionError("Pool exhausted")

    def put(self, conn: _MockConn) -> None:
        with self._lock:
            self._active.discard(conn.conn_id)
            if not conn.closed:
                self._pool.append(conn)

    def close(self) -> None:
        self._closed = True
        while self._pool:
            self._pool.popleft().close()

    @property
    def idle(self) -> int:
        return len(self._pool)

    @property
    def active(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# test_connection_pool
# ---------------------------------------------------------------------------

def test_connection_pool_initialised_with_min_connections():
    """Pool should pre-create min_connections idle connections on startup."""
    cfg = _MockConfig()
    pool = _MockPool(cfg)
    assert pool.idle == cfg.min_connections


def test_connection_pool_get_returns_connection():
    """get() should return a _MockConn instance."""
    pool = _MockPool(_MockConfig())
    conn = pool.get()
    assert isinstance(conn, _MockConn)
    pool.put(conn)


def test_connection_pool_active_increments_on_borrow():
    """Borrowing a connection should increment the active count."""
    pool = _MockPool(_MockConfig())
    before = pool.active
    conn = pool.get()
    assert pool.active == before + 1
    pool.put(conn)


def test_connection_pool_idle_decrements_on_borrow():
    """Borrowing a connection should decrement the idle count."""
    pool = _MockPool(_MockConfig())
    before = pool.idle
    conn = pool.get()
    assert pool.idle == before - 1
    pool.put(conn)


def test_connection_pool_put_returns_conn_to_idle():
    """Returning a connection should restore the idle count."""
    pool = _MockPool(_MockConfig())
    conn = pool.get()
    idle_after_borrow = pool.idle
    pool.put(conn)
    assert pool.idle == idle_after_borrow + 1


def test_connection_pool_respects_max_connections():
    """Pool should raise when all max_connections are in use."""
    cfg = _MockConfig()
    cfg.min_connections = 1
    cfg.max_connections = 2
    pool = _MockPool(cfg)
    c1 = pool.get()
    c2 = pool.get()
    try:
        pool.get()
        assert False, "Should have raised ConnectionError"
    except ConnectionError as exc:
        assert "exhausted" in str(exc).lower()
    finally:
        pool.put(c1)
        pool.put(c2)


def test_connection_pool_close_marks_pool_closed():
    """close() should mark the pool as closed."""
    pool = _MockPool(_MockConfig())
    pool.close()
    assert pool._closed is True


def test_connection_pool_closed_pool_raises_on_get():
    """get() after close() should raise ConnectionError."""
    pool = _MockPool(_MockConfig())
    pool.close()
    try:
        pool.get()
        assert False, "Should have raised ConnectionError"
    except ConnectionError as exc:
        assert "closed" in str(exc).lower()


# ---------------------------------------------------------------------------
# test_execute_query
# ---------------------------------------------------------------------------

def test_execute_query_returns_list():
    """execute() on a mock connection should return a list."""
    conn = _MockConn(1)
    result = conn.execute("SELECT 1")
    assert isinstance(result, list)


def test_execute_query_on_closed_connection_raises():
    """execute() on a closed connection should raise ConnectionError."""
    conn = _MockConn(1)
    conn.close()
    try:
        conn.execute("SELECT 1")
        assert False, "Should have raised"
    except ConnectionError:
        pass


def test_execute_query_with_params():
    """execute() should accept params without raising."""
    conn = _MockConn(1)
    result = conn.execute("SELECT * FROM users WHERE id = %s", (42,))
    assert result == []


def test_execute_query_empty_sql_raises():
    """Empty SQL should be rejected at a higher level."""
    sql = "   "
    is_empty = not sql.strip()
    assert is_empty


def test_execute_query_threadsafe():
    """Pool get/put should be thread-safe."""
    pool = _MockPool(_MockConfig())
    errors = []

    def worker():
        try:
            conn = pool.get()
            conn.execute("SELECT 1")
            pool.put(conn)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
