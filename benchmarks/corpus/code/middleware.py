"""HTTP request middleware stack.

Provides rate limiting, CORS handling, authentication, and request logging
as composable middleware functions. Designed for ASGI/WSGI-style pipelines.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Request:
    """Simplified request representation for middleware processing."""

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    client_ip: str = "127.0.0.1"
    query_params: dict[str, str] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user: dict[str, Any] | None = None


@dataclass
class Response:
    """Simplified response representation."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str = ""


Handler = Callable[[Request], Response]
Middleware = Callable[[Request, Handler], Response]


def rate_limiter(
    requests_per_minute: int = 60,
    burst: int = 10,
    key_func: Callable[[Request], str] | None = None,
) -> Middleware:
    """Create a sliding-window rate-limiter middleware.

    Args:
        requests_per_minute: Sustained request rate limit.
        burst: Maximum burst above the sustained rate.
        key_func: Function that extracts a rate-limit key from the request.
                  Defaults to client IP.

    Returns:
        Middleware function that enforces the rate limit.
    """
    window_sec = 60.0
    # In-memory store: key -> list of request timestamps
    buckets: dict[str, list[float]] = defaultdict(list)

    def _default_key(req: Request) -> str:
        return req.client_ip

    get_key = key_func or _default_key

    def middleware(request: Request, next_handler: Handler) -> Response:
        key = get_key(request)
        now = time.time()
        window_start = now - window_sec

        # Evict stale entries
        buckets[key] = [t for t in buckets[key] if t > window_start]

        if len(buckets[key]) >= requests_per_minute + burst:
            return Response(
                status=429,
                error="Too Many Requests",
                headers={"Retry-After": "60"},
            )

        buckets[key].append(now)
        return next_handler(request)

    return middleware


def cors_handler(
    allowed_origins: list[str] | None = None,
    allowed_methods: list[str] | None = None,
    allow_credentials: bool = False,
    max_age_sec: int = 86400,
) -> Middleware:
    """Create CORS middleware that injects Access-Control headers.

    Args:
        allowed_origins: List of allowed origin strings; '*' allows all.
        allowed_methods: Allowed HTTP methods (default: GET, POST, PUT, DELETE, OPTIONS).
        allow_credentials: Whether to allow credentials.
        max_age_sec: Preflight cache duration in seconds.

    Returns:
        Middleware function that adds CORS headers.
    """
    origins = allowed_origins or ["*"]
    methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    methods_str = ", ".join(methods)

    def middleware(request: Request, next_handler: Handler) -> Response:
        origin = request.headers.get("Origin", "")
        if request.method == "OPTIONS":
            # Preflight
            return Response(
                status=204,
                headers={
                    "Access-Control-Allow-Origin": origin if origin in origins else origins[0],
                    "Access-Control-Allow-Methods": methods_str,
                    "Access-Control-Max-Age": str(max_age_sec),
                    **({"Access-Control-Allow-Credentials": "true"} if allow_credentials else {}),
                },
            )

        response = next_handler(request)
        if "*" in origins or origin in origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        return response

    return middleware


def auth_middleware(
    token_verifier: Callable[[str], dict[str, Any]],
    exempt_paths: list[str] | None = None,
) -> Middleware:
    """Create authentication middleware that validates Bearer tokens.

    Args:
        token_verifier: Callable that accepts a raw token string and returns the
                        decoded payload, or raises ValueError/PermissionError.
        exempt_paths: Paths that skip authentication (e.g. /health, /login).

    Returns:
        Middleware function that populates request.user on success.
    """
    skip = set(exempt_paths or [])

    def middleware(request: Request, next_handler: Handler) -> Response:
        if request.path in skip:
            return next_handler(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(status=401, error="Missing or malformed Authorization header")

        token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = token_verifier(token)
        except (ValueError, PermissionError) as exc:
            return Response(status=401, error=str(exc))

        request.user = payload
        return next_handler(request)

    return middleware


def request_logger(
    logger: Any | None = None,
) -> Middleware:
    """Create middleware that logs each request with timing.

    Args:
        logger: Optional logger with a log_request() method. If None, prints to stdout.

    Returns:
        Middleware function that times and logs requests.
    """

    def middleware(request: Request, next_handler: Handler) -> Response:
        start = time.perf_counter()
        response = next_handler(request)
        duration_ms = (time.perf_counter() - start) * 1000

        if logger is not None and hasattr(logger, "log_request"):
            logger.log_request(
                method=request.method,
                path=request.path,
                status=response.status,
                duration_ms=duration_ms,
                request_id=request.request_id,
            )
        else:
            print(
                f"{request.method} {request.path} -> {response.status} "
                f"({duration_ms:.1f}ms) req={request.request_id[:8]}"
            )

        return response

    return middleware


def compose_middleware(middlewares: list[Middleware], handler: Handler) -> Handler:
    """Compose a list of middlewares into a single handler.

    Middlewares are applied in order: middlewares[0] is outermost.

    Args:
        middlewares: List of middleware callables.
        handler: Final request handler.

    Returns:
        Composed handler that passes requests through all middlewares.
    """
    composed = handler
    for mw in reversed(middlewares):
        # Capture mw and current composed in closure
        def make_wrapped(middleware: Middleware, next_h: Handler) -> Handler:
            def wrapped(req: Request) -> Response:
                return middleware(req, next_h)

            return wrapped

        composed = make_wrapped(mw, composed)
    return composed
