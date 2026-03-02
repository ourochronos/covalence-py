"""Asynchronous Covalence REST API client."""

from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    CovalenceBadRequestError,
    CovalenceConnectionError,
    CovalenceHTTPError,
    CovalenceNotFoundError,
    CovalenceServerError,
)
from .models import (
    AdminStats,
    AdminStatsResponse,
    Article,
    ArticleCompileRequest,
    ArticleCreateRequest,
    ArticleListResponse,
    ArticleMergeRequest,
    ArticleResponse,
    ArticleUpdateRequest,
    CompileJob,
    CompileJobResponse,
    Contention,
    ContentionListResponse,
    ContentionResolution,
    ContentionResponse,
    ContentionResolveRequest,
    ContentionStatus,
    Edge,
    EdgeCreateRequest,
    EdgeDirection,
    EdgeLabel,
    EdgeListResponse,
    EdgeResponse,
    EmbedAllResponse,
    EpistemicType,
    MaintenanceRequest,
    MaintenanceResult,
    MaintenanceResponse,
    Memory,
    MemoryForgetRequest,
    MemoryListResponse,
    MemoryRecallRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryStatusResponse,
    MemoryStoreRequest,
    NeighborhoodResponse,
    NodeStatus,
    ProvenanceResponse,
    QueueEntry,
    QueueEntryResponse,
    QueueListResponse,
    QueueStatus,
    SearchIntent,
    SearchRequest,
    SearchResponse,
    SearchWeights,
    Session,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    SessionStatus,
    Source,
    SourceIngestRequest,
    SourceListResponse,
    SourceResponse,
    SourceType,
    SplitResponse,
    SplitResult,
    TraceRequest,
    TraceResponse,
    TreeIndexAllRequest,
    TreeIndexAllResponse,
)

DEFAULT_BASE_URL = "http://localhost:8430"
DEFAULT_TIMEOUT = 30.0


def _raise_for_status(response: httpx.Response) -> None:
    """Raise a typed exception based on the HTTP status code."""
    if response.is_success:
        return
    try:
        body = response.json()
        message = body.get("error", response.text)
    except Exception:
        message = response.text

    status = response.status_code
    kwargs: dict[str, Any] = {"status_code": status, "response_body": message}

    if status == 400:
        raise CovalenceBadRequestError(message, **kwargs)
    if status == 404:
        raise CovalenceNotFoundError(message, **kwargs)
    if status == 500:
        raise CovalenceServerError(message, **kwargs)
    raise CovalenceHTTPError(message, **kwargs)


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}


class AsyncCovalenceClient:
    """Asynchronous client for the Covalence knowledge engine API.

    Parameters
    ----------
    base_url:
        Base URL of the Covalence engine (default ``http://localhost:8430``).
    timeout:
        Default request timeout in seconds (default 30).
    httpx_client:
        Optional pre-configured :class:`httpx.AsyncClient` instance.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = httpx_client is None
        self._client = httpx_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncCovalenceClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client (if owned by this instance)."""
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        try:
            r = await self._client.get(path, params=_clean_params(params))
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        return r.json()

    async def _post(self, path: str, *, json: Any = None) -> Any:
        try:
            r = await self._client.post(path, json=json)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    async def _patch(self, path: str, *, json: Any = None) -> Any:
        try:
            r = await self._client.patch(path, json=json)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    async def _delete(self, path: str) -> None:
        try:
            r = await self._client.delete(path)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)

    # ==================================================================
    # Sources
    # ==================================================================

    async def ingest_source(
        self,
        content: str,
        *,
        source_type: SourceType | str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        reliability: float | None = None,
    ) -> Source:
        """Ingest a new source document (async)."""
        req = SourceIngestRequest(
            content=content,
            source_type=source_type,  # type: ignore[arg-type]
            title=title,
            metadata=metadata,
            session_id=session_id,
            reliability=reliability,
        )
        data = await self._post("/sources", json=req.model_dump(exclude_none=True))
        return SourceResponse.model_validate(data).data

    async def list_sources(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        source_type: SourceType | str | None = None,
        status: NodeStatus | str | None = None,
        q: str | None = None,
    ) -> SourceListResponse:
        """Return a paginated list of sources (async)."""
        params = {
            "limit": limit,
            "cursor": cursor,
            "source_type": str(source_type.value) if isinstance(source_type, SourceType) else source_type,
            "status": str(status.value) if isinstance(status, NodeStatus) else status,
            "q": q,
        }
        data = await self._get("/sources", params=params)
        return SourceListResponse.model_validate(data)

    async def get_source(self, source_id: str) -> Source:
        """Fetch a single source by UUID (async)."""
        data = await self._get(f"/sources/{source_id}")
        return SourceResponse.model_validate(data).data

    async def delete_source(self, source_id: str) -> None:
        """Permanently hard-delete a source (async). **Irreversible.**"""
        await self._delete(f"/sources/{source_id}")

    # ==================================================================
    # Articles
    # ==================================================================

    async def create_article(
        self,
        content: str,
        *,
        title: str | None = None,
        domain_path: list[str] | None = None,
        epistemic_type: EpistemicType | str | None = None,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Article:
        """Directly create a new article (async)."""
        req = ArticleCreateRequest(
            content=content,
            title=title,
            domain_path=domain_path,
            epistemic_type=epistemic_type,  # type: ignore[arg-type]
            source_ids=source_ids,
            metadata=metadata,
        )
        data = await self._post("/articles", json=req.model_dump(exclude_none=True))
        return ArticleResponse.model_validate(data).data

    async def compile_article(
        self,
        source_ids: list[str],
        *,
        title_hint: str | None = None,
    ) -> CompileJob:
        """Enqueue an async LLM compilation job (async)."""
        req = ArticleCompileRequest(source_ids=source_ids, title_hint=title_hint)
        data = await self._post("/articles/compile", json=req.model_dump(exclude_none=True))
        return CompileJobResponse.model_validate(data).data

    async def merge_articles(self, article_id_a: str, article_id_b: str) -> Article:
        """Merge two articles into a new combined article (async)."""
        req = ArticleMergeRequest(article_id_a=article_id_a, article_id_b=article_id_b)
        data = await self._post("/articles/merge", json=req.model_dump())
        return ArticleResponse.model_validate(data).data

    async def list_articles(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        status: NodeStatus | str | None = None,
    ) -> ArticleListResponse:
        """Return a paginated list of articles (async)."""
        params = {
            "limit": limit,
            "cursor": cursor,
            "status": str(status.value) if isinstance(status, NodeStatus) else status,
        }
        data = await self._get("/articles", params=params)
        return ArticleListResponse.model_validate(data)

    async def get_article(self, article_id: str) -> Article:
        """Fetch a single article by UUID (async)."""
        data = await self._get(f"/articles/{article_id}")
        return ArticleResponse.model_validate(data).data

    async def update_article(
        self,
        article_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        domain_path: list[str] | None = None,
        pinned: bool | None = None,
    ) -> Article:
        """Partially update an article (async)."""
        req = ArticleUpdateRequest(content=content, title=title, domain_path=domain_path, pinned=pinned)
        data = await self._patch(f"/articles/{article_id}", json=req.model_dump(exclude_none=True))
        return ArticleResponse.model_validate(data).data

    async def archive_article(self, article_id: str) -> None:
        """Soft-delete an article (async)."""
        await self._delete(f"/articles/{article_id}")

    async def split_article(self, article_id: str) -> SplitResult:
        """Split an article into two parts (async)."""
        data = await self._post(f"/articles/{article_id}/split")
        return SplitResponse.model_validate(data).data

    async def get_provenance(self, article_id: str, *, max_depth: int | None = None) -> ProvenanceResponse:
        """Walk the graph backward from an article (async)."""
        params = {"max_depth": max_depth}
        data = await self._get(f"/articles/{article_id}/provenance", params=params)
        return ProvenanceResponse.model_validate(data)

    async def trace_claim(self, article_id: str, claim_text: str) -> TraceResponse:
        """Rank sources by TF-IDF similarity to a claim (async)."""
        req = TraceRequest(claim_text=claim_text)
        data = await self._post(f"/articles/{article_id}/trace", json=req.model_dump())
        return TraceResponse.model_validate(data)

    # ==================================================================
    # Search
    # ==================================================================

    async def search(
        self,
        query: str,
        *,
        embedding: list[float] | None = None,
        intent: SearchIntent | str | None = None,
        session_id: str | None = None,
        node_types: list[str] | None = None,
        limit: int | None = None,
        weights: SearchWeights | dict[str, float] | None = None,
    ) -> SearchResponse:
        """Unified three-dimensional search (async)."""
        if isinstance(weights, dict):
            weights = SearchWeights(**weights)
        req = SearchRequest(
            query=query,
            embedding=embedding,
            intent=intent,  # type: ignore[arg-type]
            session_id=session_id,
            node_types=node_types,
            limit=limit,
            weights=weights,
        )
        data = await self._post("/search", json=req.model_dump(exclude_none=True))
        return SearchResponse.model_validate(data)

    # ==================================================================
    # Edges / Graph
    # ==================================================================

    async def create_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        label: EdgeLabel | str,
        *,
        confidence: float | None = None,
        method: str | None = None,
        notes: str | None = None,
    ) -> Edge:
        """Create a typed edge between any two nodes (async)."""
        req = EdgeCreateRequest(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,  # type: ignore[arg-type]
            confidence=confidence,
            method=method,
            notes=notes,
        )
        data = await self._post("/edges", json=req.model_dump(exclude_none=True))
        return EdgeResponse.model_validate(data).data

    async def delete_edge(self, edge_id: str) -> None:
        """Remove an edge from both PostgreSQL and Apache AGE (async)."""
        await self._delete(f"/edges/{edge_id}")

    async def list_node_edges(
        self,
        node_id: str,
        *,
        direction: EdgeDirection | str | None = None,
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> EdgeListResponse:
        """List edges connected to a node (async)."""
        params: dict[str, Any] = {"limit": limit}
        if direction is not None:
            params["direction"] = direction.value if isinstance(direction, EdgeDirection) else direction
        if labels:
            params["labels"] = ",".join(labels)
        data = await self._get(f"/nodes/{node_id}/edges", params=params)
        return EdgeListResponse.model_validate(data)

    async def get_neighborhood(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        direction: EdgeDirection | str | None = None,
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> NeighborhoodResponse:
        """Traverse the graph outward from a node (async)."""
        params: dict[str, Any] = {"depth": depth, "limit": limit}
        if direction is not None:
            params["direction"] = direction.value if isinstance(direction, EdgeDirection) else direction
        if labels:
            params["labels"] = ",".join(labels)
        data = await self._get(f"/nodes/{node_id}/neighborhood", params=params)
        return NeighborhoodResponse.model_validate(data)

    # ==================================================================
    # Contentions
    # ==================================================================

    async def list_contentions(
        self,
        *,
        node_id: str | None = None,
        status: ContentionStatus | str | None = None,
    ) -> ContentionListResponse:
        """Return all contentions (async)."""
        params = {
            "node_id": node_id,
            "status": status.value if isinstance(status, ContentionStatus) else status,
        }
        data = await self._get("/contentions", params=params)
        return ContentionListResponse.model_validate(data)

    async def get_contention(self, contention_id: str) -> Contention:
        """Fetch a single contention by UUID (async)."""
        data = await self._get(f"/contentions/{contention_id}")
        return ContentionResponse.model_validate(data).data

    async def resolve_contention(
        self,
        contention_id: str,
        resolution: ContentionResolution | str,
        rationale: str,
    ) -> Contention:
        """Mark a contention as resolved (async)."""
        req = ContentionResolveRequest(resolution=resolution, rationale=rationale)  # type: ignore[arg-type]
        data = await self._post(f"/contentions/{contention_id}/resolve", json=req.model_dump())
        return ContentionResponse.model_validate(data).data

    # ==================================================================
    # Memory
    # ==================================================================

    async def store_memory(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        importance: float | None = None,
        context: str | None = None,
        supersedes_id: str | None = None,
    ) -> Memory:
        """Store a new memory observation (async)."""
        req = MemoryStoreRequest(
            content=content,
            tags=tags,
            importance=importance,
            context=context,
            supersedes_id=supersedes_id,
        )
        data = await self._post("/memory", json=req.model_dump(exclude_none=True))
        return MemoryResponse.model_validate(data).data

    async def recall_memories(
        self,
        query: str,
        *,
        limit: int | None = None,
        tags: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> MemoryListResponse:
        """Search active memories (async)."""
        req = MemoryRecallRequest(
            query=query,
            limit=limit,
            tags=tags,
            min_confidence=min_confidence,
        )
        data = await self._post("/memory/search", json=req.model_dump(exclude_none=True))
        return MemoryListResponse.model_validate(data)

    async def get_memory_status(self) -> MemoryStatus:
        """Return aggregate memory counts (async)."""
        data = await self._get("/memory/status")
        return MemoryStatusResponse.model_validate(data).data

    async def forget_memory(self, memory_id: str, *, reason: str | None = None) -> None:
        """Soft-delete a memory (async)."""
        req = MemoryForgetRequest(reason=reason)
        await self._patch(f"/memory/{memory_id}/forget", json=req.model_dump(exclude_none=True))

    # ==================================================================
    # Sessions
    # ==================================================================

    async def create_session(
        self,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session (async)."""
        req = SessionCreateRequest(label=label, metadata=metadata)
        data = await self._post("/sessions", json=req.model_dump(exclude_none=True))
        return SessionResponse.model_validate(data).data

    async def list_sessions(
        self,
        *,
        status: SessionStatus | str | None = None,
        limit: int | None = None,
    ) -> SessionListResponse:
        """Return sessions ordered by last_active_at (async)."""
        params = {
            "status": status.value if isinstance(status, SessionStatus) else status,
            "limit": limit,
        }
        data = await self._get("/sessions", params=params)
        return SessionListResponse.model_validate(data)

    async def get_session(self, session_id: str) -> Session:
        """Fetch a single session by UUID (async)."""
        data = await self._get(f"/sessions/{session_id}")
        return SessionResponse.model_validate(data).data

    async def close_session(self, session_id: str) -> None:
        """Set session status to ``closed`` (async)."""
        await self._post(f"/sessions/{session_id}/close")

    # ==================================================================
    # Admin
    # ==================================================================

    async def get_admin_stats(self) -> AdminStats:
        """Return a system health snapshot (async)."""
        data = await self._get("/admin/stats")
        return AdminStatsResponse.model_validate(data).data

    async def run_maintenance(
        self,
        *,
        recompute_scores: bool | None = None,
        process_queue: bool | None = None,
        evict_if_over_capacity: bool | None = None,
        evict_count: int | None = None,
    ) -> MaintenanceResult:
        """Trigger one or more maintenance operations (async)."""
        req = MaintenanceRequest(
            recompute_scores=recompute_scores,
            process_queue=process_queue,
            evict_if_over_capacity=evict_if_over_capacity,
            evict_count=evict_count,
        )
        data = await self._post("/admin/maintenance", json=req.model_dump(exclude_none=True))
        return MaintenanceResponse.model_validate(data).data

    async def list_queue(
        self,
        *,
        status: QueueStatus | str | None = None,
        limit: int | None = None,
    ) -> QueueListResponse:
        """List slow-path background task queue entries (async)."""
        params = {
            "status": status.value if isinstance(status, QueueStatus) else status,
            "limit": limit,
        }
        data = await self._get("/admin/queue", params=params)
        return QueueListResponse.model_validate(data)

    async def get_queue_entry(self, entry_id: str) -> QueueEntry:
        """Fetch a single queue entry by UUID (async)."""
        data = await self._get(f"/admin/queue/{entry_id}")
        return QueueEntryResponse.model_validate(data).data

    async def retry_queue_entry(self, entry_id: str) -> QueueEntry:
        """Reset a ``failed`` queue entry back to ``pending`` (async)."""
        data = await self._post(f"/admin/queue/{entry_id}/retry")
        return QueueEntryResponse.model_validate(data).data

    async def delete_queue_entry(self, entry_id: str) -> None:
        """Delete a queue entry (async)."""
        await self._delete(f"/admin/queue/{entry_id}")

    async def embed_all(self) -> EmbedAllResponse:
        """Enqueue embed tasks for all unembedded nodes (async)."""
        data = await self._post("/admin/embed-all")
        return EmbedAllResponse.model_validate(data)

    async def tree_index_all(
        self,
        *,
        overlap: float | None = None,
        force: bool | None = None,
        min_chars: int | None = None,
    ) -> TreeIndexAllResponse:
        """Enqueue tree_index tasks for all eligible nodes (async)."""
        req = TreeIndexAllRequest(overlap=overlap, force=force, min_chars=min_chars)
        data = await self._post("/admin/tree-index-all", json=req.model_dump(exclude_none=True))
        return TreeIndexAllResponse.model_validate(data)
