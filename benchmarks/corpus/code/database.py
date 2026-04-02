"""Database connection pooling and query execution layer.

Provides a thread-safe connection pool, query helpers, and lifecycle management.
All implementations are mock/stub — no real DB connection is made.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    host: str = "localhost"
    port: int = 5432
    database: str = "appdb"
    user: str = "app_user"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout_sec: float = 30.0
    idle_timeout_sec: float = 600.0
    statement_timeout_ms: int = 30_000
    ssl_mode: str = "prefer"


class _MockConnection:
    """Simulated database connection for benchmarking."""

    def __init__(self, conn_id: int, config: DatabaseConfig) -> None:
        self.conn_id = conn_id
        self.config = config
        self._closed = False
        self._in_transaction = False
        self.created_at = time.time()

    def execute(self, sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """Execute a SQL statement and return rows."""
        if self._closed:
            raise ConnectionError(f"Connection {self.conn_id} is closed")
        # Mock: return empty result set
        return []

    def commit(self) -> None:
        """Commit the current transaction."""
        self._in_transaction = False

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self._in_transaction = False

    def close(self) -> None:
        """Close this connection."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class ConnectionPool:
    """Thread-safe database connection pool with min/max sizing.

    Manages a pool of mock database connections, providing borrow/return
    semantics with a configurable maximum size.
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()
        self._pool: deque[_MockConnection] = deque()
        self._active: set[int] = set()
        self._lock = threading.Lock()
        self._conn_counter = 0
        self._closed = False
        # Pre-create min connections
        for _ in range(self._config.min_connections):
            self._pool.append(self._create_connection())

    def _create_connection(self) -> _MockConnection:
        self._conn_counter += 1
        return _MockConnection(self._conn_counter, self._config)

    def get_connection(self) -> _MockConnection:
        """Borrow a connection from the pool.

        Returns:
            Available _MockConnection.

        Raises:
            ConnectionError: If pool is closed or exhausted and at max capacity.
        """
        if self._closed:
            raise ConnectionError("Connection pool is closed")

        with self._lock:
            if self._pool:
                conn = self._pool.popleft()
                self._active.add(conn.conn_id)
                return conn
            total = len(self._active) + len(self._pool)
            if total < self._config.max_connections:
                conn = self._create_connection()
                self._active.add(conn.conn_id)
                return conn
            raise ConnectionError("Connection pool exhausted")

    def return_connection(self, conn: _MockConnection) -> None:
        """Return a borrowed connection back to the pool.

        Args:
            conn: Previously borrowed connection.
        """
        with self._lock:
            self._active.discard(conn.conn_id)
            if not conn.closed:
                self._pool.append(conn)

    def close_pool(self) -> None:
        """Close all connections and mark the pool as closed."""
        with self._lock:
            self._closed = True
            while self._pool:
                conn = self._pool.popleft()
                conn.close()

    @property
    def pool_size(self) -> int:
        """Number of idle connections in the pool."""
        return len(self._pool)

    @property
    def active_connections(self) -> int:
        """Number of connections currently in use."""
        return len(self._active)

    def get_stats(self) -> dict[str, Any]:
        """Return pool statistics."""
        return {
            "pool_size": self.pool_size,
            "active_connections": self.active_connections,
            "total_created": self._conn_counter,
            "is_closed": self._closed,
            "max_connections": self._config.max_connections,
        }


@contextmanager
def get_connection(pool: ConnectionPool) -> Generator[_MockConnection, None, None]:
    """Context manager that borrows a connection and returns it on exit.

    Args:
        pool: ConnectionPool to borrow from.

    Yields:
        An active _MockConnection.
    """
    conn = pool.get_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.return_connection(conn)


def execute_query(
    pool: ConnectionPool,
    sql: str,
    params: tuple | None = None,
    timeout_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Execute a parameterised SQL query and return all rows.

    Args:
        pool: Active ConnectionPool.
        sql: SQL statement to execute.
        params: Optional tuple of query parameters.
        timeout_ms: Per-query timeout override in milliseconds.

    Returns:
        List of row dictionaries.

    Raises:
        ValueError: If sql is empty.
        ConnectionError: If pool is closed or exhausted.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL statement must not be empty")

    with get_connection(pool) as conn:
        return conn.execute(sql, params)


def execute_many(
    pool: ConnectionPool,
    sql: str,
    param_list: list[tuple],
) -> int:
    """Execute a parameterised statement for each row in param_list.

    Args:
        pool: Active ConnectionPool.
        sql: SQL template with placeholders.
        param_list: List of parameter tuples, one per execution.

    Returns:
        Number of rows processed.
    """
    count = 0
    with get_connection(pool) as conn:
        for params in param_list:
            conn.execute(sql, params)
            count += 1
    return count


def close_pool(pool: ConnectionPool) -> None:
    """Convenience wrapper to close a pool and log the action."""
    stats = pool.get_stats()
    pool.close_pool()
    # In production this would emit a structured log entry
    _ = stats  # silences linters in this mock
