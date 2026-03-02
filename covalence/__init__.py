"""Covalence Python client library.

Quick start::

    from covalence import CovalenceClient

    with CovalenceClient() as client:
        source = client.ingest_source("Hello, world!", title="Greeting")
        results = client.search("greeting message")

Async variant::

    import asyncio
    from covalence import AsyncCovalenceClient

    async def main():
        async with AsyncCovalenceClient() as client:
            source = await client.ingest_source("Hello, world!")
            results = await client.search("greeting")

    asyncio.run(main())
"""

from .async_client import AsyncCovalenceClient
from .client import CovalenceClient
from .exceptions import (
    CovalenceBadRequestError,
    CovalenceConnectionError,
    CovalenceError,
    CovalenceHTTPError,
    CovalenceNotFoundError,
    CovalenceServerError,
)
from .models import (
    Article,
    ArticleCompileRequest,
    ArticleCreateRequest,
    ArticleMergeRequest,
    ArticleUpdateRequest,
    AdminStats,
    CompileJob,
    Contention,
    ContentionResolution,
    ContentionStatus,
    Edge,
    EdgeDirection,
    EdgeLabel,
    EpistemicType,
    Memory,
    MemoryStatus,
    NeighborhoodEntry,
    NodeStatus,
    ProvenanceEntry,
    QueueEntry,
    QueueStatus,
    QueueTaskType,
    SearchIntent,
    SearchResult,
    SearchWeights,
    Session,
    SessionStatus,
    Source,
    SourceType,
    SplitResult,
    TraceResult,
    TreeIndexAllResponse,
)

__all__ = [
    # Clients
    "CovalenceClient",
    "AsyncCovalenceClient",
    # Exceptions
    "CovalenceError",
    "CovalenceBadRequestError",
    "CovalenceNotFoundError",
    "CovalenceServerError",
    "CovalenceHTTPError",
    "CovalenceConnectionError",
    # Models
    "Article",
    "ArticleCompileRequest",
    "ArticleCreateRequest",
    "ArticleMergeRequest",
    "ArticleUpdateRequest",
    "AdminStats",
    "CompileJob",
    "Contention",
    "ContentionResolution",
    "ContentionStatus",
    "Edge",
    "EdgeDirection",
    "EdgeLabel",
    "EpistemicType",
    "Memory",
    "MemoryStatus",
    "NeighborhoodEntry",
    "NodeStatus",
    "ProvenanceEntry",
    "QueueEntry",
    "QueueStatus",
    "QueueTaskType",
    "SearchIntent",
    "SearchResult",
    "SearchWeights",
    "Session",
    "SessionStatus",
    "Source",
    "SourceType",
    "SplitResult",
    "TraceResult",
    "TreeIndexAllResponse",
]

__version__ = "0.1.0"
