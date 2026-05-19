"""Module-level OpenAI rate limiter for LangChain ChatOpenAI invocations.

Instantiated once at import time so all graph nodes share the same token bucket.
Disable via ``OPENAI_RATE_LIMIT_ENABLED=false`` in settings.
"""

from __future__ import annotations

from langchain_core.rate_limiters import InMemoryRateLimiter

from config.settings import settings


OPENAI_RATE_LIMITER: InMemoryRateLimiter | None = None
if settings.openai_rate_limit_enabled:
    OPENAI_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=settings.openai_rate_limit_requests_per_second,
        check_every_n_seconds=settings.openai_rate_limit_check_every_n_seconds,
        max_bucket_size=settings.openai_rate_limit_max_bucket_size,
    )
