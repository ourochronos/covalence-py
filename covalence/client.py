"""Synchronous Covalence REST API client."""

from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    CovalenceBadRequestError,
    CovalenceConnectionError,
    CovalenceError,
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


class CovalenceClient:
    """Synchronous client for the Covalence knowledge engine API.

    Parameters
    ----------
    base_url:
        Base URL of the Covalence engine (default ``http://localhost:8430``).
    timeout:
        Default request timeout in seconds (default 30).
    api_key:
        Optional API key.  When provided, every request includes
        ``Authorization: Bearer <api_key>``.  Matches the server-side
        ``COVALENCE_API_KEY`` env var.
    httpx_client:
        Optional pre-configured :class:`httpx.Client` instance. When provided,
        ``base_url``, ``timeout``, and ``api_key`` are ignored and the caller
        is responsible for closing the client.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str | None = None,
        httpx_client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = httpx_client is None
        base_headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            base_headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=base_headers,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CovalenceClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client (if owned by this instance)."""
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        try:
            r = self._client.get(path, params=_clean_params(params))
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        return r.json()

    def _post(self, path: str, *, json: Any = None) -> Any:
        try:
            r = self._client.post(path, json=json)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def _patch(self, path: str, *, json: Any = None) -> Any:
        try:
            r = self._client.patch(path, json=json)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def _delete(self, path: str) -> None:
        try:
            r = self._client.delete(path)
        except httpx.TransportError as exc:
            raise CovalenceConnectionError(str(exc)) from exc
        _raise_for_status(r)

    # ==================================================================
    # Sources
    # ==================================================================

    def ingest_source(
        self,
        content: str,
        *,
        source_type: SourceType | str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        reliability: float | None = None,
    ) -> Source:
        """Ingest a new source document.

        Re-ingesting identical content is **idempotent** — the existing record
        is returned.

        Parameters
        ----------
        content:
            Raw text content of the source.
        source_type:
            Type hint affecting the default reliability score.
        title:
            Human-readable title.
        metadata:
            Arbitrary JSON metadata.
        session_id:
            If provided, a ``CAPTURED_IN`` edge is created.
        reliability:
            Override the default reliability score (0.0–1.0).
        """
        req = SourceIngestRequest(
            content=content,
            source_type=source_type,  # type: ignore[arg-type]
            title=title,
            metadata=metadata,
            session_id=session_id,
            reliability=reliability,
        )
        data = self._post("/sources", json=req.model_dump(exclude_none=True))
        return SourceResponse.model_validate(data).data

    def list_sources(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        source_type: SourceType | str | None = None,
        status: NodeStatus | str | None = None,
        q: str | None = None,
    ) -> SourceListResponse:
        """Return a paginated list of sources, newest first.

        Parameters
        ----------
        limit:
            Max results (1–100, default 20).
        cursor:
            Cursor-based pagination: return sources with ``id > cursor``.
        source_type:
            Filter by source type.
        status:
            Filter by status (``active``, ``archived``, ``tombstone``).
        q:
            Full-text search query.
        """
        params = {
            "limit": limit,
            "cursor": cursor,
            "source_type": str(source_type.value) if isinstance(source_type, SourceType) else source_type,
            "status": str(status.value) if isinstance(status, NodeStatus) else status,
            "q": q,
        }
        data = self._get("/sources", params=params)
        return SourceListResponse.model_validate(data)

    def get_source(self, source_id: str) -> Source:
        """Fetch a single source by UUID."""
        data = self._get(f"/sources/{source_id}")
        return SourceResponse.model_validate(data).data

    def delete_source(self, source_id: str) -> None:
        """Permanently hard-delete a source and its embedding.

        This is **irreversible**.
        """
        self._delete(f"/sources/{source_id}")

    # ==================================================================
    # Articles
    # ==================================================================

    def create_article(
        self,
        content: str,
        *,
        title: str | None = None,
        domain_path: list[str] | None = None,
        epistemic_type: EpistemicType | str | None = None,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Article:
        """Directly create a new article (agent-authored).

        Parameters
        ----------
        content:
            Article body text.
        title:
            Human-readable title.
        domain_path:
            Hierarchical domain tags.
        epistemic_type:
            One of ``semantic``, ``episodic``, ``procedural``, ``declarative``.
        source_ids:
            UUIDs of source nodes this article originates from.
        metadata:
            Arbitrary JSON metadata.
        """
        req = ArticleCreateRequest(
            content=content,
            title=title,
            domain_path=domain_path,
            epistemic_type=epistemic_type,  # type: ignore[arg-type]
            source_ids=source_ids,
            metadata=metadata,
        )
        data = self._post("/articles", json=req.model_dump(exclude_none=True))
        return ArticleResponse.model_validate(data).data

    def compile_article(
        self,
        source_ids: list[str],
        *,
        title_hint: str | None = None,
    ) -> CompileJob:
        """Enqueue an async LLM compilation job.

        Returns a job object with a ``job_id``; poll
        :meth:`get_queue_entry` until ``status == "completed"``.

        Parameters
        ----------
        source_ids:
            Non-empty list of source UUIDs to compile.
        title_hint:
            Optional title suggestion for the LLM.
        """
        req = ArticleCompileRequest(source_ids=source_ids, title_hint=title_hint)
        data = self._post("/articles/compile", json=req.model_dump(exclude_none=True))
        return CompileJobResponse.model_validate(data).data

    def merge_articles(self, article_id_a: str, article_id_b: str) -> Article:
        """Merge two articles into a new combined article.

        Both originals are **archived** after the merge.

        Parameters
        ----------
        article_id_a:
            First article UUID.
        article_id_b:
            Second article UUID.
        """
        req = ArticleMergeRequest(article_id_a=article_id_a, article_id_b=article_id_b)
        data = self._post("/articles/merge", json=req.model_dump())
        return ArticleResponse.model_validate(data).data

    def list_articles(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        status: NodeStatus | str | None = None,
    ) -> ArticleListResponse:
        """Return a paginated list of articles, newest first.

        Parameters
        ----------
        limit:
            Max results (1–100, default 20).
        cursor:
            Pagination cursor.
        status:
            Filter by status (default ``active``).
        """
        params = {
            "limit": limit,
            "cursor": cursor,
            "status": str(status.value) if isinstance(status, NodeStatus) else status,
        }
        data = self._get("/articles", params=params)
        return ArticleListResponse.model_validate(data)

    def get_article(self, article_id: str) -> Article:
        """Fetch a single article by UUID.

        Includes a live ``contention_count`` of active CONTRADICTS/CONTENDS edges.
        """
        data = self._get(f"/articles/{article_id}")
        return ArticleResponse.model_validate(data).data

    def update_article(
        self,
        article_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        domain_path: list[str] | None = None,
        pinned: bool | None = None,
    ) -> Article:
        """Partially update an article.

        Increments ``version`` and updates ``modified_at``. If ``content`` is
        changed, a new ``embed`` task is enqueued.
        """
        req = ArticleUpdateRequest(content=content, title=title, domain_path=domain_path, pinned=pinned)
        data = self._patch(f"/articles/{article_id}", json=req.model_dump(exclude_none=True))
        return ArticleResponse.model_validate(data).data

    def archive_article(self, article_id: str) -> None:
        """Soft-delete an article (status → ``archived``)."""
        self._delete(f"/articles/{article_id}")

    def split_article(self, article_id: str) -> SplitResult:
        """Split an article into two roughly equal parts at a paragraph boundary.

        The original is **archived**. Two new articles are created with
        ``SPLIT_INTO`` edges from the original.
        """
        data = self._post(f"/articles/{article_id}/split")
        return SplitResponse.model_validate(data).data

    def get_provenance(self, article_id: str, *, max_depth: int | None = None) -> ProvenanceResponse:
        """Walk the graph backward from an article following provenance edge types.

        Parameters
        ----------
        article_id:
            Article UUID.
        max_depth:
            Maximum traversal depth (default 5).
        """
        params = {"max_depth": max_depth}
        data = self._get(f"/articles/{article_id}/provenance", params=params)
        return ProvenanceResponse.model_validate(data)

    def trace_claim(self, article_id: str, claim_text: str) -> TraceResponse:
        """Rank an article's linked sources by TF-IDF similarity to a claim.

        Parameters
        ----------
        article_id:
            Article UUID.
        claim_text:
            The specific claim or sentence to trace.
        """
        req = TraceRequest(claim_text=claim_text)
        data = self._post(f"/articles/{article_id}/trace", json=req.model_dump())
        return TraceResponse.model_validate(data)

    # ==================================================================
    # Search
    # ==================================================================

    def search(
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
        """Unified three-dimensional search (vector + lexical + graph).

        Parameters
        ----------
        query:
            Natural language query string.
        embedding:
            Pre-computed query embedding. If omitted, auto-embedded from ``query``.
        intent:
            Routing hint: ``factual``, ``temporal``, ``causal``, or ``entity``.
        session_id:
            Session context for graph-aware ranking.
        node_types:
            Filter to specific node types, e.g. ``["article"]``.
        limit:
            Max results (default 10).
        weights:
            Custom dimension weights.
        """
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
        data = self._post("/search", json=req.model_dump(exclude_none=True))
        return SearchResponse.model_validate(data)

    # ==================================================================
    # Edges / Graph
    # ==================================================================

    def create_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        label: EdgeLabel | str,
        *,
        confidence: float | None = None,
        method: str | None = None,
        notes: str | None = None,
    ) -> Edge:
        """Create a typed edge between any two nodes.

        Parameters
        ----------
        from_node_id:
            Source (origin) node UUID.
        to_node_id:
            Target (destination) node UUID.
        label:
            Edge type label.
        confidence:
            Edge confidence (0.0–1.0, default 1.0).
        method:
            Creation method (default ``agent_explicit``).
        notes:
            Free-text annotation stored in edge metadata.
        """
        req = EdgeCreateRequest(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,  # type: ignore[arg-type]
            confidence=confidence,
            method=method,
            notes=notes,
        )
        data = self._post("/edges", json=req.model_dump(exclude_none=True))
        return EdgeResponse.model_validate(data).data

    def delete_edge(self, edge_id: str) -> None:
        """Remove an edge from both PostgreSQL and Apache AGE."""
        self._delete(f"/edges/{edge_id}")

    def list_node_edges(
        self,
        node_id: str,
        *,
        direction: EdgeDirection | str | None = None,
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> EdgeListResponse:
        """List edges connected to a node.

        Parameters
        ----------
        node_id:
            Node UUID (source, article, or session).
        direction:
            ``outbound``, ``inbound``, or both (default).
        labels:
            Edge type filter list.
        limit:
            Max results (default 50).
        """
        params: dict[str, Any] = {"limit": limit}
        if direction is not None:
            params["direction"] = direction.value if isinstance(direction, EdgeDirection) else direction
        if labels:
            params["labels"] = ",".join(labels)
        data = self._get(f"/nodes/{node_id}/edges", params=params)
        return EdgeListResponse.model_validate(data)

    def get_neighborhood(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        direction: EdgeDirection | str | None = None,
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> NeighborhoodResponse:
        """Traverse the graph outward from a node up to ``depth`` hops.

        Parameters
        ----------
        node_id:
            Starting node UUID.
        depth:
            Max traversal depth (1–5, default 2).
        direction:
            ``outbound``, ``inbound``, or both (default).
        labels:
            Edge type filter.
        limit:
            Max total neighbors returned (1–200, default 50).
        """
        params: dict[str, Any] = {"depth": depth, "limit": limit}
        if direction is not None:
            params["direction"] = direction.value if isinstance(direction, EdgeDirection) else direction
        if labels:
            params["labels"] = ",".join(labels)
        data = self._get(f"/nodes/{node_id}/neighborhood", params=params)
        return NeighborhoodResponse.model_validate(data)

    # ==================================================================
    # Contentions
    # ==================================================================

    def list_contentions(
        self,
        *,
        node_id: str | None = None,
        status: ContentionStatus | str | None = None,
    ) -> ContentionListResponse:
        """Return all contentions, optionally filtered by article node or status.

        Parameters
        ----------
        node_id:
            Filter to contentions involving this article node.
        status:
            Filter by status: ``detected``, ``resolved``, ``dismissed``.
        """
        params = {
            "node_id": node_id,
            "status": status.value if isinstance(status, ContentionStatus) else status,
        }
        data = self._get("/contentions", params=params)
        return ContentionListResponse.model_validate(data)

    def get_contention(self, contention_id: str) -> Contention:
        """Fetch a single contention by UUID."""
        data = self._get(f"/contentions/{contention_id}")
        return ContentionResponse.model_validate(data).data

    def resolve_contention(
        self,
        contention_id: str,
        resolution: ContentionResolution | str,
        rationale: str,
    ) -> Contention:
        """Mark a contention as resolved.

        Parameters
        ----------
        contention_id:
            Contention UUID.
        resolution:
            One of ``supersede_a``, ``supersede_b``, ``accept_both``, ``dismiss``.
        rationale:
            Free-text explanation of the decision.
        """
        req = ContentionResolveRequest(resolution=resolution, rationale=rationale)  # type: ignore[arg-type]
        data = self._post(f"/contentions/{contention_id}/resolve", json=req.model_dump())
        return ContentionResponse.model_validate(data).data

    # ==================================================================
    # Memory
    # ==================================================================

    def store_memory(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        importance: float | None = None,
        context: str | None = None,
        supersedes_id: str | None = None,
    ) -> Memory:
        """Store a new memory observation.

        Parameters
        ----------
        content:
            Memory content text.
        tags:
            Categorization tags.
        importance:
            0.0–1.0; drives confidence score (default 0.5).
        context:
            Provenance hint, e.g. ``"session:main"``.
        supersedes_id:
            UUID of a prior memory this replaces.
        """
        req = MemoryStoreRequest(
            content=content,
            tags=tags,
            importance=importance,
            context=context,
            supersedes_id=supersedes_id,
        )
        data = self._post("/memory", json=req.model_dump(exclude_none=True))
        return MemoryResponse.model_validate(data).data

    def recall_memories(
        self,
        query: str,
        *,
        limit: int | None = None,
        tags: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> MemoryListResponse:
        """Search active memories using full-text search.

        Parameters
        ----------
        query:
            Natural language recall query.
        limit:
            Max results (default 5).
        tags:
            Return only memories containing ALL listed tags.
        min_confidence:
            Minimum confidence threshold.
        """
        req = MemoryRecallRequest(
            query=query,
            limit=limit,
            tags=tags,
            min_confidence=min_confidence,
        )
        data = self._post("/memory/search", json=req.model_dump(exclude_none=True))
        return MemoryListResponse.model_validate(data)

    def get_memory_status(self) -> MemoryStatus:
        """Return aggregate counts for the memory subsystem."""
        data = self._get("/memory/status")
        return MemoryStatusResponse.model_validate(data).data

    def forget_memory(self, memory_id: str, *, reason: str | None = None) -> None:
        """Soft-delete a memory.

        The underlying node is **not** deleted; it is excluded from future
        recall results.

        Parameters
        ----------
        memory_id:
            Memory (source node) UUID.
        reason:
            Optional reason stored as ``metadata.forget_reason``.
        """
        req = MemoryForgetRequest(reason=reason)
        self._patch(f"/memory/{memory_id}/forget", json=req.model_dump(exclude_none=True))

    # ==================================================================
    # Sessions
    # ==================================================================

    def create_session(
        self,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session.

        Parameters
        ----------
        label:
            Human-readable session label.
        metadata:
            Arbitrary JSON metadata.
        """
        req = SessionCreateRequest(label=label, metadata=metadata)
        data = self._post("/sessions", json=req.model_dump(exclude_none=True))
        return SessionResponse.model_validate(data).data

    def list_sessions(
        self,
        *,
        status: SessionStatus | str | None = None,
        limit: int | None = None,
    ) -> SessionListResponse:
        """Return sessions ordered by ``last_active_at`` descending.

        Parameters
        ----------
        status:
            Filter by status: ``open`` or ``closed``.
        limit:
            Max results (default 50).
        """
        params = {
            "status": status.value if isinstance(status, SessionStatus) else status,
            "limit": limit,
        }
        data = self._get("/sessions", params=params)
        return SessionListResponse.model_validate(data)

    def get_session(self, session_id: str) -> Session:
        """Fetch a single session by UUID."""
        data = self._get(f"/sessions/{session_id}")
        return SessionResponse.model_validate(data).data

    def close_session(self, session_id: str) -> None:
        """Set session status to ``closed``."""
        self._post(f"/sessions/{session_id}/close")

    # ==================================================================
    # Admin
    # ==================================================================

    def get_admin_stats(self) -> AdminStats:
        """Return a system health snapshot."""
        data = self._get("/admin/stats")
        return AdminStatsResponse.model_validate(data).data

    def run_maintenance(
        self,
        *,
        recompute_scores: bool | None = None,
        process_queue: bool | None = None,
        evict_if_over_capacity: bool | None = None,
        evict_count: int | None = None,
    ) -> MaintenanceResult:
        """Trigger one or more maintenance operations.

        Parameters
        ----------
        recompute_scores:
            Recompute ``usage_score`` for all active nodes.
        process_queue:
            Time out stale ``processing`` queue jobs.
        evict_if_over_capacity:
            Archive lowest-scoring articles when count > 1000.
        evict_count:
            Max articles to evict per run (default 10).
        """
        req = MaintenanceRequest(
            recompute_scores=recompute_scores,
            process_queue=process_queue,
            evict_if_over_capacity=evict_if_over_capacity,
            evict_count=evict_count,
        )
        data = self._post("/admin/maintenance", json=req.model_dump(exclude_none=True))
        return MaintenanceResponse.model_validate(data).data

    def list_queue(
        self,
        *,
        status: QueueStatus | str | None = None,
        limit: int | None = None,
    ) -> QueueListResponse:
        """List slow-path background task queue entries.

        Results are ordered by ``priority DESC, created_at ASC``.

        Parameters
        ----------
        status:
            Filter by: ``pending``, ``processing``, ``failed``, ``completed``.
        limit:
            Max results (default 50).
        """
        params = {
            "status": status.value if isinstance(status, QueueStatus) else status,
            "limit": limit,
        }
        data = self._get("/admin/queue", params=params)
        return QueueListResponse.model_validate(data)

    def get_queue_entry(self, entry_id: str) -> QueueEntry:
        """Fetch a single queue entry by UUID."""
        data = self._get(f"/admin/queue/{entry_id}")
        return QueueEntryResponse.model_validate(data).data

    def retry_queue_entry(self, entry_id: str) -> QueueEntry:
        """Reset a ``failed`` queue entry back to ``pending``."""
        data = self._post(f"/admin/queue/{entry_id}/retry")
        return QueueEntryResponse.model_validate(data).data

    def delete_queue_entry(self, entry_id: str) -> None:
        """Delete a queue entry."""
        self._delete(f"/admin/queue/{entry_id}")

    def embed_all(self) -> EmbedAllResponse:
        """Enqueue ``embed`` tasks for every active node without an embedding."""
        data = self._post("/admin/embed-all")
        return EmbedAllResponse.model_validate(data)

    def tree_index_all(
        self,
        *,
        overlap: float | None = None,
        force: bool | None = None,
        min_chars: int | None = None,
    ) -> TreeIndexAllResponse:
        """Enqueue ``tree_index`` tasks for all eligible active nodes.

        Parameters
        ----------
        overlap:
            Chunk overlap ratio (default 0.20).
        force:
            Re-index already-indexed nodes if ``True`` (default ``False``).
        min_chars:
            Minimum content length in characters to qualify (default 700).
        """
        req = TreeIndexAllRequest(overlap=overlap, force=force, min_chars=min_chars)
        data = self._post("/admin/tree-index-all", json=req.model_dump(exclude_none=True))
        return TreeIndexAllResponse.model_validate(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove ``None`` values from a query parameter dict."""
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}
