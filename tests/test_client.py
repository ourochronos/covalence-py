"""Unit tests for the sync CovalenceClient using httpx mock transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from covalence.client import CovalenceClient
from covalence.exceptions import (
    CovalenceBadRequestError,
    CovalenceNotFoundError,
    CovalenceServerError,
)
from covalence.models import (
    ArticleListResponse,
    ContentionListResponse,
    ContentionResolution,
    ContentionStatus,
    EdgeDirection,
    EdgeLabel,
    EmbedAllResponse,
    MemoryListResponse,
    NodeStatus,
    QueueListResponse,
    QueueStatus,
    SearchIntent,
    SessionListResponse,
    SessionStatus,
    SourceListResponse,
    SourceType,
    TreeIndexAllResponse,
)
from tests.fixtures import (
    ADMIN_STATS_OBJ,
    ARTICLE_ID,
    ARTICLE_OBJ,
    CONTENTION_ID,
    CONTENTION_OBJ,
    EDGE_ID,
    EDGE_OBJ,
    JOB_ID,
    MEMORY_ID,
    MEMORY_OBJ,
    QUEUE_ENTRY_OBJ,
    QUEUE_ID,
    SESSION_ID,
    SESSION_OBJ,
    SOURCE_ID,
    SOURCE_OBJ,
)


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------


def _make_transport(routes: dict[str, Any]) -> httpx.MockTransport:
    """Build an httpx.MockTransport from a {(method, path): response_data} map."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "not found in mock"})
        entry = routes[key]
        if callable(entry):
            return entry(request)
        status, body = entry
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def _client(routes: dict[str, Any]) -> CovalenceClient:
    transport = _make_transport(routes)
    http = httpx.Client(transport=transport, base_url="http://test")
    return CovalenceClient(httpx_client=http)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestSources:
    def test_ingest_source(self):
        routes = {("POST", "/sources"): (201, {"data": SOURCE_OBJ})}
        with _client(routes) as c:
            s = c.ingest_source("Rust ownership...", title="Rust memory model overview")
        assert s.id == SOURCE_ID
        assert s.title == "Rust memory model overview"

    def test_ingest_source_with_all_params(self):
        routes = {("POST", "/sources"): (201, {"data": SOURCE_OBJ})}
        with _client(routes) as c:
            s = c.ingest_source(
                "content",
                source_type=SourceType.document,
                metadata={"url": "http://example.com"},
                session_id=SESSION_ID,
                reliability=0.9,
            )
        assert s.id == SOURCE_ID

    def test_list_sources(self):
        routes = {("GET", "/sources"): (200, {"data": [SOURCE_OBJ], "meta": {"count": 1}})}
        with _client(routes) as c:
            resp = c.list_sources(limit=10)
        assert isinstance(resp, SourceListResponse)
        assert len(resp.data) == 1
        assert resp.meta.count == 1

    def test_get_source(self):
        routes = {("GET", f"/sources/{SOURCE_ID}"): (200, {"data": SOURCE_OBJ})}
        with _client(routes) as c:
            s = c.get_source(SOURCE_ID)
        assert s.id == SOURCE_ID

    def test_delete_source(self):
        routes = {("DELETE", f"/sources/{SOURCE_ID}"): (204, None)}
        with _client(routes) as c:
            result = c.delete_source(SOURCE_ID)
        assert result is None

    def test_get_source_not_found(self):
        routes = {("GET", f"/sources/{SOURCE_ID}"): (404, {"error": "not found"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceNotFoundError):
                c.get_source(SOURCE_ID)


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


class TestArticles:
    def test_create_article(self):
        routes = {("POST", "/articles"): (201, {"data": ARTICLE_OBJ})}
        with _client(routes) as c:
            a = c.create_article("Rust guarantees...", title="Rust Memory Safety")
        assert a.id == ARTICLE_ID
        assert a.domain_path == ["rust", "memory", "safety"]

    def test_create_article_with_source_ids(self):
        routes = {("POST", "/articles"): (201, {"data": ARTICLE_OBJ})}
        with _client(routes) as c:
            a = c.create_article("body", source_ids=[SOURCE_ID])
        assert a.id == ARTICLE_ID

    def test_compile_article(self):
        routes = {("POST", "/articles/compile"): (202, {"data": {"job_id": JOB_ID, "status": "pending"}})}
        with _client(routes) as c:
            job = c.compile_article([SOURCE_ID], title_hint="Rust patterns")
        assert job.job_id == JOB_ID
        assert job.status == "pending"

    def test_merge_articles(self):
        routes = {("POST", "/articles/merge"): (201, {"data": ARTICLE_OBJ})}
        with _client(routes) as c:
            a = c.merge_articles(ARTICLE_ID, "other-id")
        assert a.id == ARTICLE_ID

    def test_list_articles(self):
        routes = {("GET", "/articles"): (200, {"data": [ARTICLE_OBJ], "meta": {"count": 1}})}
        with _client(routes) as c:
            resp = c.list_articles()
        assert isinstance(resp, ArticleListResponse)
        assert len(resp.data) == 1

    def test_list_articles_with_status(self):
        routes = {("GET", "/articles"): (200, {"data": [], "meta": {"count": 0}})}
        with _client(routes) as c:
            resp = c.list_articles(status=NodeStatus.archived)
        assert resp.meta.count == 0

    def test_get_article(self):
        routes = {("GET", f"/articles/{ARTICLE_ID}"): (200, {"data": ARTICLE_OBJ})}
        with _client(routes) as c:
            a = c.get_article(ARTICLE_ID)
        assert a.version == 3
        assert a.contention_count == 1

    def test_update_article(self):
        updated = {**ARTICLE_OBJ, "title": "Rust Memory Safety (Updated)", "pinned": True}
        routes = {("PATCH", f"/articles/{ARTICLE_ID}"): (200, {"data": updated})}
        with _client(routes) as c:
            a = c.update_article(ARTICLE_ID, title="Rust Memory Safety (Updated)", pinned=True)
        assert a.pinned is True

    def test_archive_article(self):
        routes = {("DELETE", f"/articles/{ARTICLE_ID}"): (204, None)}
        with _client(routes) as c:
            result = c.archive_article(ARTICLE_ID)
        assert result is None

    def test_split_article(self):
        split_data = {
            "original_id": ARTICLE_ID,
            "part_a": {**ARTICLE_OBJ, "id": "part-a"},
            "part_b": {**ARTICLE_OBJ, "id": "part-b"},
        }
        routes = {("POST", f"/articles/{ARTICLE_ID}/split"): (201, {"data": split_data})}
        with _client(routes) as c:
            result = c.split_article(ARTICLE_ID)
        assert result.original_id == ARTICLE_ID
        assert result.part_a.id == "part-a"
        assert result.part_b.id == "part-b"

    def test_get_provenance(self):
        prov_data = [
            {
                "source_node": SOURCE_OBJ,
                "edge_type": "ORIGINATES",
                "confidence": 1.0,
                "depth": 1,
            }
        ]
        routes = {("GET", f"/articles/{ARTICLE_ID}/provenance"): (200, {"data": prov_data})}
        with _client(routes) as c:
            resp = c.get_provenance(ARTICLE_ID)
        assert len(resp.data) == 1
        assert resp.data[0].edge_type == "ORIGINATES"

    def test_trace_claim(self):
        trace_data = [
            {
                "source_id": SOURCE_ID,
                "title": "Rust memory model overview",
                "score": 0.847,
                "snippet": "...ownership system...",
            }
        ]
        routes = {
            ("POST", f"/articles/{ARTICLE_ID}/trace"): (
                200,
                {"data": trace_data, "meta": {"count": 1}},
            )
        }
        with _client(routes) as c:
            resp = c.trace_claim(ARTICLE_ID, "ownership eliminates dangling pointer bugs")
        assert resp.data[0].score == 0.847

    def test_article_not_found(self):
        routes = {("GET", f"/articles/{ARTICLE_ID}"): (404, {"error": "not found"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceNotFoundError):
                c.get_article(ARTICLE_ID)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_basic(self):
        result = {
            "node_id": ARTICLE_ID,
            "score": 0.913,
            "node_type": "article",
            "title": "Rust Memory Safety",
            "content_preview": "Rust guarantees...",
        }
        routes = {
            ("POST", "/search"): (
                200,
                {
                    "data": [result],
                    "meta": {"total_results": 1, "elapsed_ms": 42},
                },
            )
        }
        with _client(routes) as c:
            resp = c.search("how does Rust prevent memory leaks")
        assert len(resp.data) == 1
        assert resp.data[0].score == 0.913
        assert resp.meta is not None
        assert resp.meta.total_results == 1

    def test_search_with_all_params(self):
        routes = {("POST", "/search"): (200, {"data": [], "meta": {"total_results": 0}})}
        with _client(routes) as c:
            resp = c.search(
                "Rust memory",
                intent=SearchIntent.factual,
                session_id=SESSION_ID,
                node_types=["article"],
                limit=5,
                weights={"vector": 0.7, "lexical": 0.2, "graph": 0.1},
            )
        assert resp.data == []


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestEdges:
    def test_create_edge(self):
        routes = {("POST", "/edges"): (201, {"data": EDGE_OBJ})}
        with _client(routes) as c:
            e = c.create_edge(
                SOURCE_ID,
                ARTICLE_ID,
                EdgeLabel.CONFIRMS,
                confidence=0.9,
                notes="Corroborates ownership rules",
            )
        assert e.edge_type == "CONFIRMS"
        assert e.confidence == 0.9

    def test_delete_edge(self):
        routes = {("DELETE", f"/edges/{EDGE_ID}"): (204, None)}
        with _client(routes) as c:
            c.delete_edge(EDGE_ID)

    def test_list_node_edges(self):
        routes = {
            ("GET", f"/nodes/{SOURCE_ID}/edges"): (
                200,
                {"data": [EDGE_OBJ], "meta": {"count": 1}},
            )
        }
        with _client(routes) as c:
            resp = c.list_node_edges(SOURCE_ID, direction=EdgeDirection.outbound)
        assert len(resp.data) == 1

    def test_list_node_edges_with_labels(self):
        routes = {
            ("GET", f"/nodes/{SOURCE_ID}/edges"): (
                200,
                {"data": [], "meta": {"count": 0}},
            )
        }
        with _client(routes) as c:
            resp = c.list_node_edges(SOURCE_ID, labels=["CONFIRMS", "ORIGINATES"])
        assert resp.meta.count == 0

    def test_get_neighborhood(self):
        neighbor = {
            "node": ARTICLE_OBJ,
            "edge": EDGE_OBJ,
            "depth": 1,
        }
        routes = {
            ("GET", f"/nodes/{SOURCE_ID}/neighborhood"): (
                200,
                {"data": [neighbor], "meta": {"count": 1}},
            )
        }
        with _client(routes) as c:
            resp = c.get_neighborhood(SOURCE_ID, depth=2)
        assert len(resp.data) == 1
        assert resp.data[0].depth == 1


# ---------------------------------------------------------------------------
# Contentions
# ---------------------------------------------------------------------------


class TestContentions:
    def test_list_contentions(self):
        routes = {("GET", "/contentions"): (200, {"data": [CONTENTION_OBJ]})}
        with _client(routes) as c:
            resp = c.list_contentions()
        assert isinstance(resp, ContentionListResponse)
        assert len(resp.data) == 1

    def test_list_contentions_filtered(self):
        routes = {("GET", "/contentions"): (200, {"data": [CONTENTION_OBJ]})}
        with _client(routes) as c:
            resp = c.list_contentions(node_id=ARTICLE_ID, status=ContentionStatus.detected)
        assert resp.data[0].status.value == "detected"

    def test_get_contention(self):
        routes = {("GET", f"/contentions/{CONTENTION_ID}"): (200, {"data": CONTENTION_OBJ})}
        with _client(routes) as c:
            c_obj = c.get_contention(CONTENTION_ID)
        assert c_obj.id == CONTENTION_ID
        assert c_obj.severity == "high"

    def test_resolve_contention(self):
        resolved = {
            **CONTENTION_OBJ,
            "status": "resolved",
            "resolution": "supersede_a: The source is an uncited blog post...",
            "resolved_at": "2026-03-01T11:30:00Z",
        }
        routes = {("POST", f"/contentions/{CONTENTION_ID}/resolve"): (200, {"data": resolved})}
        with _client(routes) as c:
            c_obj = c.resolve_contention(
                CONTENTION_ID,
                ContentionResolution.supersede_a,
                "The source is an uncited blog post...",
            )
        assert c_obj.status.value == "resolved"

    def test_resolve_contention_bad_resolution_pydantic(self):
        """Pydantic rejects invalid enum values client-side before the request fires."""
        from pydantic import ValidationError

        routes: dict = {}
        with _client(routes) as c:
            with pytest.raises(ValidationError):
                c.resolve_contention(CONTENTION_ID, "bad_value", "reason")  # type: ignore[arg-type]

    def test_resolve_contention_server_400(self):
        """Server may still return 400 for other business-rule violations."""
        routes = {("POST", f"/contentions/{CONTENTION_ID}/resolve"): (400, {"error": "already resolved"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceBadRequestError):
                # Use a valid enum value so Pydantic passes, but server still rejects
                c.resolve_contention(CONTENTION_ID, ContentionResolution.dismiss, "reason")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_store_memory(self):
        routes = {("POST", "/memory"): (201, {"data": MEMORY_OBJ})}
        with _client(routes) as c:
            m = c.store_memory(
                "The user prefers dark mode",
                tags=["preferences", "ui"],
                importance=0.8,
                context="conversation:user",
            )
        assert m.id == MEMORY_ID
        assert m.forgotten is False

    def test_store_memory_with_supersedes(self):
        routes = {("POST", "/memory"): (201, {"data": MEMORY_OBJ})}
        with _client(routes) as c:
            m = c.store_memory("new preference", supersedes_id=MEMORY_ID)
        assert m.id == MEMORY_ID

    def test_recall_memories(self):
        routes = {("POST", "/memory/search"): (200, {"data": [MEMORY_OBJ]})}
        with _client(routes) as c:
            resp = c.recall_memories("UI preferences", tags=["ui"], min_confidence=0.5)
        assert isinstance(resp, MemoryListResponse)
        assert resp.data[0].tags == ["preferences", "ui"]

    def test_get_memory_status(self):
        status_data = {
            "total_memories": 42,
            "active_memories": 38,
            "forgotten_memories": 4,
        }
        routes = {("GET", "/memory/status"): (200, {"data": status_data})}
        with _client(routes) as c:
            s = c.get_memory_status()
        assert s.total_memories == 42
        assert s.active_memories == 38

    def test_forget_memory(self):
        routes = {("PATCH", f"/memory/{MEMORY_ID}/forget"): (204, None)}
        with _client(routes) as c:
            c.forget_memory(MEMORY_ID, reason="outdated")

    def test_forget_memory_not_found(self):
        routes = {("PATCH", f"/memory/{MEMORY_ID}/forget"): (404, {"error": "not found"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceNotFoundError):
                c.forget_memory(MEMORY_ID)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_create_session(self):
        routes = {("POST", "/sessions"): (201, {"data": SESSION_OBJ})}
        with _client(routes) as c:
            s = c.create_session(label="research-session-2026-03-01")
        assert s.id == SESSION_ID
        assert s.status.value == "open"

    def test_list_sessions(self):
        routes = {("GET", "/sessions"): (200, {"data": [SESSION_OBJ]})}
        with _client(routes) as c:
            resp = c.list_sessions()
        assert isinstance(resp, SessionListResponse)
        assert len(resp.data) == 1

    def test_list_sessions_filtered(self):
        routes = {("GET", "/sessions"): (200, {"data": [SESSION_OBJ]})}
        with _client(routes) as c:
            resp = c.list_sessions(status=SessionStatus.open, limit=10)
        assert resp.data[0].id == SESSION_ID

    def test_get_session(self):
        routes = {("GET", f"/sessions/{SESSION_ID}"): (200, {"data": SESSION_OBJ})}
        with _client(routes) as c:
            s = c.get_session(SESSION_ID)
        assert s.label == "research-session-2026-03-01"

    def test_close_session(self):
        routes = {("POST", f"/sessions/{SESSION_ID}/close"): (204, None)}
        with _client(routes) as c:
            c.close_session(SESSION_ID)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TestAdmin:
    def test_get_admin_stats(self):
        routes = {("GET", "/admin/stats"): (200, {"data": ADMIN_STATS_OBJ})}
        with _client(routes) as c:
            stats = c.get_admin_stats()
        assert stats.nodes.total == 1420
        assert stats.edges.in_sync is True
        assert stats.queue.failed == 1
        assert stats.embeddings.nodes_without == 50

    def test_run_maintenance(self):
        routes = {
            ("POST", "/admin/maintenance"): (
                200,
                {"data": {"actions_taken": ["recomputed usage scores", "evicted 12 low-usage articles"]}},
            )
        }
        with _client(routes) as c:
            result = c.run_maintenance(recompute_scores=True, evict_if_over_capacity=True, evict_count=20)
        assert "recomputed usage scores" in result.actions_taken

    def test_list_queue(self):
        routes = {("GET", "/admin/queue"): (200, {"data": [QUEUE_ENTRY_OBJ]})}
        with _client(routes) as c:
            resp = c.list_queue(status=QueueStatus.pending)
        assert isinstance(resp, QueueListResponse)
        assert resp.data[0].task_type == "embed"

    def test_get_queue_entry(self):
        routes = {("GET", f"/admin/queue/{QUEUE_ID}"): (200, {"data": QUEUE_ENTRY_OBJ})}
        with _client(routes) as c:
            entry = c.get_queue_entry(QUEUE_ID)
        assert entry.id == QUEUE_ID
        assert entry.priority == 3

    def test_retry_queue_entry(self):
        routes = {("POST", f"/admin/queue/{QUEUE_ID}/retry"): (200, {"data": QUEUE_ENTRY_OBJ})}
        with _client(routes) as c:
            entry = c.retry_queue_entry(QUEUE_ID)
        assert entry.status.value == "pending"

    def test_retry_non_failed_entry_raises(self):
        routes = {("POST", f"/admin/queue/{QUEUE_ID}/retry"): (400, {"error": "entry is not failed"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceBadRequestError):
                c.retry_queue_entry(QUEUE_ID)

    def test_delete_queue_entry(self):
        routes = {("DELETE", f"/admin/queue/{QUEUE_ID}"): (204, None)}
        with _client(routes) as c:
            c.delete_queue_entry(QUEUE_ID)

    def test_embed_all(self):
        routes = {("POST", "/admin/embed-all"): (200, {"queued": 47})}
        with _client(routes) as c:
            resp = c.embed_all()
        assert isinstance(resp, EmbedAllResponse)
        assert resp.queued == 47

    def test_tree_index_all(self):
        routes = {
            ("POST", "/admin/tree-index-all"): (
                200,
                {"queued": 23, "overlap": 0.15, "force": False, "min_chars": 500},
            )
        }
        with _client(routes) as c:
            resp = c.tree_index_all(overlap=0.15, min_chars=500)
        assert isinstance(resp, TreeIndexAllResponse)
        assert resp.queued == 23
        assert resp.overlap == 0.15


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_400_raises_bad_request(self):
        routes = {("POST", "/sources"): (400, {"error": "invalid body"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceBadRequestError) as exc_info:
                c.ingest_source("x")
        assert exc_info.value.status_code == 400

    def test_404_raises_not_found(self):
        routes = {("GET", f"/articles/{ARTICLE_ID}"): (404, {"error": "not found"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceNotFoundError) as exc_info:
                c.get_article(ARTICLE_ID)
        assert exc_info.value.status_code == 404

    def test_500_raises_server_error(self):
        routes = {("POST", "/articles"): (500, {"error": "database error"})}
        with _client(routes) as c:
            with pytest.raises(CovalenceServerError) as exc_info:
                c.create_article("body")
        assert exc_info.value.status_code == 500

    def test_context_manager(self):
        """Client closes cleanly as context manager."""
        routes = {("GET", f"/sources/{SOURCE_ID}"): (200, {"data": SOURCE_OBJ})}
        with _client(routes) as c:
            s = c.get_source(SOURCE_ID)
        assert s.id == SOURCE_ID
