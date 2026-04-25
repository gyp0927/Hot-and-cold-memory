"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, Info

# Query metrics
QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total queries",
    ["tier", "status"],
)

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Query latency",
    ["tier"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Chunk metrics
CHUNKS_TOTAL = Gauge(
    "rag_chunks_total",
    "Total chunks by tier",
    ["tier"],
)

# Migration metrics
MIGRATION_TOTAL = Counter(
    "rag_migration_total",
    "Total migrations",
    ["direction", "status"],
)

MIGRATION_DURATION = Histogram(
    "rag_migration_duration_seconds",
    "Migration latency",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# LLM metrics
LLM_REQUESTS_TOTAL = Counter(
    "rag_llm_requests_total",
    "Total LLM API calls",
    ["operation"],
)

LLM_REQUEST_DURATION = Histogram(
    "rag_llm_request_duration_seconds",
    "LLM API latency",
    ["operation"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# System info
APP_INFO = Info("rag_app", "Application information")
