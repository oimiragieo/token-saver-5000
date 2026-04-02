"""Application configuration loading and validation.

Reads settings from environment variables with defaults, validates them,
and exposes a typed ConfigSchema dataclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatabaseSettings:
    host: str = "localhost"
    port: int = 5432
    name: str = "appdb"
    user: str = "postgres"
    password: str = ""
    pool_min: int = 2
    pool_max: int = 10


@dataclass
class AuthSettings:
    secret_key: str = "change-me"
    token_expire_minutes: int = 30
    refresh_expire_days: int = 7
    algorithm: str = "HS256"


@dataclass
class CacheSettings:
    backend: str = "redis"
    url: str = "redis://localhost:6379/0"
    default_ttl_sec: int = 300
    max_memory_mb: int = 256


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    log_level: str = "INFO"


@dataclass
class ConfigSchema:
    """Top-level application configuration."""

    env: str = "development"
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    feature_flags: dict[str, bool] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_debug(self) -> bool:
        return self.server.debug or self.env == "development"


def _parse_int(value: str, default: int, name: str) -> int:
    """Parse an env var string to int, raising ConfigError on failure."""
    try:
        return int(value)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid integer for {name}: {value!r}") from exc


def _parse_bool(value: str) -> bool:
    """Parse common truthy/falsy string representations."""
    return value.lower() in ("1", "true", "yes", "on")


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""


def load_config(env: dict[str, str] | None = None) -> ConfigSchema:
    """Load configuration from environment variables.

    Args:
        env: Optional dict to use instead of os.environ (useful in tests).

    Returns:
        Populated ConfigSchema instance.

    Raises:
        ConfigError: If required values are missing or malformed.
    """
    e = env if env is not None else dict(os.environ)

    db = DatabaseSettings(
        host=e.get("DB_HOST", "localhost"),
        port=_parse_int(e.get("DB_PORT", "5432"), 5432, "DB_PORT"),
        name=e.get("DB_NAME", "appdb"),
        user=e.get("DB_USER", "postgres"),
        password=e.get("DB_PASSWORD", ""),
        pool_min=_parse_int(e.get("DB_POOL_MIN", "2"), 2, "DB_POOL_MIN"),
        pool_max=_parse_int(e.get("DB_POOL_MAX", "10"), 10, "DB_POOL_MAX"),
    )

    auth = AuthSettings(
        secret_key=e.get("AUTH_SECRET_KEY", "change-me"),
        token_expire_minutes=_parse_int(
            e.get("AUTH_TOKEN_EXPIRE_MINUTES", "30"), 30, "AUTH_TOKEN_EXPIRE_MINUTES"
        ),
        refresh_expire_days=_parse_int(
            e.get("AUTH_REFRESH_EXPIRE_DAYS", "7"), 7, "AUTH_REFRESH_EXPIRE_DAYS"
        ),
        algorithm=e.get("AUTH_ALGORITHM", "HS256"),
    )

    cache = CacheSettings(
        backend=e.get("CACHE_BACKEND", "redis"),
        url=e.get("CACHE_URL", "redis://localhost:6379/0"),
        default_ttl_sec=_parse_int(
            e.get("CACHE_DEFAULT_TTL_SEC", "300"), 300, "CACHE_DEFAULT_TTL_SEC"
        ),
        max_memory_mb=_parse_int(e.get("CACHE_MAX_MEMORY_MB", "256"), 256, "CACHE_MAX_MEMORY_MB"),
    )

    cors_raw = e.get("SERVER_CORS_ORIGINS", "http://localhost:3000")
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

    server = ServerSettings(
        host=e.get("SERVER_HOST", "0.0.0.0"),
        port=_parse_int(e.get("SERVER_PORT", "8000"), 8000, "SERVER_PORT"),
        debug=_parse_bool(e.get("SERVER_DEBUG", "false")),
        workers=_parse_int(e.get("SERVER_WORKERS", "4"), 4, "SERVER_WORKERS"),
        cors_origins=cors_origins,
        log_level=e.get("LOG_LEVEL", "INFO").upper(),
    )

    flags_raw = e.get("FEATURE_FLAGS", "")
    feature_flags: dict[str, bool] = {}
    for pair in flags_raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            feature_flags[k.strip()] = _parse_bool(v.strip())

    cfg = ConfigSchema(
        env=e.get("APP_ENV", "development"),
        db=db,
        auth=auth,
        cache=cache,
        server=server,
        feature_flags=feature_flags,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ConfigSchema) -> None:
    """Validate a ConfigSchema instance, raising ConfigError on problems.

    Args:
        cfg: ConfigSchema to validate.

    Raises:
        ConfigError: On any invalid combination or missing required value.
    """
    if not cfg.auth.secret_key or cfg.auth.secret_key == "change-me":
        if cfg.is_production:
            raise ConfigError("AUTH_SECRET_KEY must be set in production")

    if cfg.db.pool_min > cfg.db.pool_max:
        raise ConfigError(
            f"DB_POOL_MIN ({cfg.db.pool_min}) must be <= DB_POOL_MAX ({cfg.db.pool_max})"
        )

    if cfg.auth.token_expire_minutes < 1:
        raise ConfigError("AUTH_TOKEN_EXPIRE_MINUTES must be >= 1")

    valid_algos = {"HS256", "HS384", "HS512", "RS256"}
    if cfg.auth.algorithm not in valid_algos:
        raise ConfigError(
            f"AUTH_ALGORITHM {cfg.auth.algorithm!r} is not supported. "
            f"Choose from: {', '.join(sorted(valid_algos))}"
        )

    if cfg.server.workers < 1:
        raise ConfigError("SERVER_WORKERS must be >= 1")


def get_config_summary(cfg: ConfigSchema) -> dict[str, Any]:
    """Return a loggable summary of the config (no secrets)."""
    return {
        "env": cfg.env,
        "db_host": cfg.db.host,
        "db_port": cfg.db.port,
        "db_name": cfg.db.name,
        "cache_backend": cfg.cache.backend,
        "server_host": cfg.server.host,
        "server_port": cfg.server.port,
        "server_workers": cfg.server.workers,
        "log_level": cfg.server.log_level,
        "feature_flags": cfg.feature_flags,
    }
