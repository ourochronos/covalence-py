"""Unit tests for the async AsyncCovalenceClient using httpx mock transport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from covalence.async_client import AsyncCovalenceClient
from covalence.exceptions import CovalenceNotFoundError, CovalenceServerError
from covalence.models import (
    ContentionResolution,
    EdgeLabel,
    QueueStatus,
    SessionStatus,
    SourceType,
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


def _make_async_client(routes: dict[str, Any]) -> AsyncCovalenceClient:
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "not found in mock"})
        status, body = routes[key]
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return AsyncCovalenceClient(httpx_client=http)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestAsyncSources:
    @pytest.mark.asyncio
    async def test_ingest_source(self):
        client = _make_async_client({("POST", "/sources"): (201, {"data": SOURCE_OBJ})})
        async with client:
            s = await client.ingest_source("content", title="Test")
        assert s.id == SOURCE_ID

    @pytest.mark.asyncio
    async def test_list_sources(self):
        client = _make_async_client(
            {("GET", "/sources"): (200, {"data": [SOURCE_OBJ], "meta": {"count": 1}})}
        )
        async with client:
            resp = await client.list_sources(limit=5)
        assert resp.meta.count == 1

    @pytest.mark.asyncio
    async def test_get_source(self):
        client = _make_async_client({("GET", f"/sources/{SOURCE_ID}"): (200, {"data": SOURCE_OBJ})})
        async with client:
            s = await client.get_source(SOURCE_ID)
        assert s.id == SOURCE_ID

    @pytest.mark.asyncio
    async def test_delete_source(self):
        client = _make_async_client({("DELETE", f"/sources/{SOURCE_ID}"): (204, None)})
        async with client:
            await client.delete_source(SOURCE_ID)

    @pytest.mark.asyncio
    async def test_get_source_not_found(self):
        client = _make_async_client({("GET", f"/sources/{SOURCE_ID}"): (404, {"error": "not found"})})
        async with client:
            with pytest.raises(CovalenceNotFoundError):
                await client.get_source(SOURCE_ID)


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


class TestAsyncArticles:
    @pytest.mark.asyncio
    async def test_create_article(self):
        client = _make_async_client({("POST", "/articles"): (201, {"data": ARTICLE_OBJ})})
        async with client:
            a = await client.create_article("content", title="Title")
        assert a.id == ARTICLE_ID

    @pytest.mark.asyncio
    async def test_compile_article(self):
        client = _make_async_client(
            {("POST", "/articles/compile"): (202, {"data": {"job_id": JOB_ID, "status": "pending"}})}
        )
        async with client:
            job = await client.compile_article([SOURCE_ID])
        assert job.job_id == JOB_ID

    @pytest.mark.asyncio
    async def test_merge_articles(self):
        client = _make_async_client({("POST", "/articles/merge"): (201, {"data": ARTICLE_OBJ})})
        async with client:
            a = await client.merge_articles(ARTICLE_ID, "other-id")
        assert a.id == ARTICLE_ID

    @pytest.mark.asyncio
    async def test_list_articles(self):
        client = _make_async_client(
            {("GET", "/articles"): (200, {"data": [ARTICLE_OBJ], "meta": {"count": 1}})}
        )
        async with client:
            resp = await client.list_articles()
        assert len(resp.data) == 1

    @pytest.mark.asyncio
    async def test_get_article(self):
        client = _make_async_client({("GET", f"/articles/{ARTICLE_ID}"): (200, {"data": ARTICLE_OBJ})})
        async with client:
            a = await client.get_article(ARTICLE_ID)
        assert a.version == 3

    @pytest.mark.asyncio
    async def test_update_article(self):
        updated = {**ARTICLE_OBJ, "pinned": True}
        client = _make_async_client({("PATCH", f"/articles/{ARTICLE_ID}"): (200, {"data": updated})})
        async with client:
            a = await client.update_article(ARTICLE_ID, pinned=True)
        assert a.pinned is True

    @pytest.mark.asyncio
    async def test_archive_article(self):
        client = _make_async_client({("DELETE", f"/articles/{ARTICLE_ID}"): (204, None)})
        async with client:
            await client.archive_article(ARTICLE_ID)

    @pytest.mark.asyncio
    async def test_split_article(self):
        split_data = {
            "original_id": ARTICLE_ID,
            "part_a": {**ARTICLE_OBJ, "id": "pa"},
            "part_b": {**ARTICLE_OBJ, "id": "pb"},
        }
        client = _make_async_client({("POST", f"/articles/{ARTICLE_ID}/split"): (201, {"data": split_data})})
        async with client:
            result = await client.split_article(ARTICLE_ID)
        assert result.part_a.id == "pa"

    @pytest.mark.asyncio
    async def test_get_provenance(self):
        prov_data = [{"source_node": SOURCE_OBJ, "edge_type": "ORIGINATES", "confidence": 1.0, "depth": 1}]
        client = _make_async_client(
            {("GET", f"/articles/{ARTICLE_ID}/provenance"): (200, {"data": prov_data})}
        )
        async with client:
            resp = await client.get_provenance(ARTICLE_ID)
        assert resp.data[0].edge_type == "ORIGINATES"

    @pytest.mark.asyncio
    async def test_trace_claim(self):
        trace_data = [{"source_id": SOURCE_ID, "title": "T", "score": 0.9, "snippet": "..."}]
        client = _make_async_client(
            {("POST", f"/articles/{ARTICLE_ID}/trace"): (200, {"data": trace_data, "meta": {"count": 1}})}
        )
        async with client:
            resp = await client.trace_claim(ARTICLE_ID, "claim text")
        assert resp.data[0].score == 0.9


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestAsyncSearch:
    @pytest.mark.asyncio
    async def test_search(self):
        result = {"node_id": ARTICLE_ID, "score": 0.9, "node_type": "article", "title": "T"}
        client = _make_async_client(
            {("POST", "/search"): (200, {"data": [result], "meta": {"total_results": 1}})}
        )
        async with client:
            resp = await client.search("Rust memory")
        assert resp.data[0].node_id == ARTICLE_ID


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestAsyncEdges:
    @pytest.mark.asyncio
    async def test_create_edge(self):
        client = _make_async_client({("POST", "/edges"): (201, {"data": EDGE_OBJ})})
        async with client:
            e = await client.create_edge(SOURCE_ID, ARTICLE_ID, EdgeLabel.CONFIRMS)
        assert e.edge_type == "CONFIRMS"

    @pytest.mark.asyncio
    async def test_delete_edge(self):
        client = _make_async_client({("DELETE", f"/edges/{EDGE_ID}"): (204, None)})
        async with client:
            await client.delete_edge(EDGE_ID)

    @pytest.mark.asyncio
    async def test_list_node_edges(self):
        client = _make_async_client(
            {("GET", f"/nodes/{SOURCE_ID}/edges"): (200, {"data": [EDGE_OBJ], "meta": {"count": 1}})}
        )
        async with client:
            resp = await client.list_node_edges(SOURCE_ID)
        assert resp.meta.count == 1

    @pytest.mark.asyncio
    async def test_get_neighborhood(self):
        neighbor = {"node": ARTICLE_OBJ, "edge": EDGE_OBJ, "depth": 1}
        client = _make_async_client(
            {("GET", f"/nodes/{SOURCE_ID}/neighborhood"): (200, {"data": [neighbor], "meta": {"count": 1}})}
        )
        async with client:
            resp = await client.get_neighborhood(SOURCE_ID)
        assert resp.data[0].depth == 1


# ---------------------------------------------------------------------------
# Contentions
# ---------------------------------------------------------------------------


class TestAsyncContentions:
    @pytest.mark.asyncio
    async def test_list_contentions(self):
        client = _make_async_client({("GET", "/contentions"): (200, {"data": [CONTENTION_OBJ]})})
        async with client:
            resp = await client.list_contentions()
        assert len(resp.data) == 1

    @pytest.mark.asyncio
    async def test_get_contention(self):
        client = _make_async_client(
            {("GET", f"/contentions/{CONTENTION_ID}"): (200, {"data": CONTENTION_OBJ})}
        )
        async with client:
            c = await client.get_contention(CONTENTION_ID)
        assert c.id == CONTENTION_ID

    @pytest.mark.asyncio
    async def test_resolve_contention(self):
        resolved = {**CONTENTION_OBJ, "status": "resolved", "resolution": "supersede_a: reason"}
        client = _make_async_client(
            {("POST", f"/contentions/{CONTENTION_ID}/resolve"): (200, {"data": resolved})}
        )
        async with client:
            c = await client.resolve_contention(CONTENTION_ID, ContentionResolution.supersede_a, "reason")
        assert c.status.value == "resolved"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestAsyncMemory:
    @pytest.mark.asyncio
    async def test_store_memory(self):
        client = _make_async_client({("POST", "/memory"): (201, {"data": MEMORY_OBJ})})
        async with client:
            m = await client.store_memory("dark mode preference", tags=["ui"], importance=0.8)
        assert m.id == MEMORY_ID

    @pytest.mark.asyncio
    async def test_recall_memories(self):
        client = _make_async_client({("POST", "/memory/search"): (200, {"data": [MEMORY_OBJ]})})
        async with client:
            resp = await client.recall_memories("UI preferences")
        assert resp.data[0].importance == 0.8

    @pytest.mark.asyncio
    async def test_get_memory_status(self):
        status_data = {"total_memories": 10, "active_memories": 9, "forgotten_memories": 1}
        client = _make_async_client({("GET", "/memory/status"): (200, {"data": status_data})})
        async with client:
            s = await client.get_memory_status()
        assert s.total_memories == 10

    @pytest.mark.asyncio
    async def test_forget_memory(self):
        client = _make_async_client({("PATCH", f"/memory/{MEMORY_ID}/forget"): (204, None)})
        async with client:
            await client.forget_memory(MEMORY_ID)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestAsyncSessions:
    @pytest.mark.asyncio
    async def test_create_session(self):
        client = _make_async_client({("POST", "/sessions"): (201, {"data": SESSION_OBJ})})
        async with client:
            s = await client.create_session(label="test-session")
        assert s.id == SESSION_ID

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        client = _make_async_client({("GET", "/sessions"): (200, {"data": [SESSION_OBJ]})})
        async with client:
            resp = await client.list_sessions(status=SessionStatus.open)
        assert len(resp.data) == 1

    @pytest.mark.asyncio
    async def test_get_session(self):
        client = _make_async_client({("GET", f"/sessions/{SESSION_ID}"): (200, {"data": SESSION_OBJ})})
        async with client:
            s = await client.get_session(SESSION_ID)
        assert s.label == "research-session-2026-03-01"

    @pytest.mark.asyncio
    async def test_close_session(self):
        client = _make_async_client({("POST", f"/sessions/{SESSION_ID}/close"): (204, None)})
        async with client:
            await client.close_session(SESSION_ID)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TestAsyncAdmin:
    @pytest.mark.asyncio
    async def test_get_admin_stats(self):
        client = _make_async_client({("GET", "/admin/stats"): (200, {"data": ADMIN_STATS_OBJ})})
        async with client:
            stats = await client.get_admin_stats()
        assert stats.nodes.total == 1420

    @pytest.mark.asyncio
    async def test_run_maintenance(self):
        client = _make_async_client(
            {
                ("POST", "/admin/maintenance"): (
                    200,
                    {"data": {"actions_taken": ["recomputed usage scores"]}},
                )
            }
        )
        async with client:
            result = await client.run_maintenance(recompute_scores=True)
        assert "recomputed usage scores" in result.actions_taken

    @pytest.mark.asyncio
    async def test_list_queue(self):
        client = _make_async_client({("GET", "/admin/queue"): (200, {"data": [QUEUE_ENTRY_OBJ]})})
        async with client:
            resp = await client.list_queue(status=QueueStatus.pending)
        assert resp.data[0].task_type == "embed"

    @pytest.mark.asyncio
    async def test_get_queue_entry(self):
        client = _make_async_client({("GET", f"/admin/queue/{QUEUE_ID}"): (200, {"data": QUEUE_ENTRY_OBJ})})
        async with client:
            entry = await client.get_queue_entry(QUEUE_ID)
        assert entry.id == QUEUE_ID

    @pytest.mark.asyncio
    async def test_retry_queue_entry(self):
        client = _make_async_client(
            {("POST", f"/admin/queue/{QUEUE_ID}/retry"): (200, {"data": QUEUE_ENTRY_OBJ})}
        )
        async with client:
            entry = await client.retry_queue_entry(QUEUE_ID)
        assert entry.status.value == "pending"

    @pytest.mark.asyncio
    async def test_delete_queue_entry(self):
        client = _make_async_client({("DELETE", f"/admin/queue/{QUEUE_ID}"): (204, None)})
        async with client:
            await client.delete_queue_entry(QUEUE_ID)

    @pytest.mark.asyncio
    async def test_embed_all(self):
        client = _make_async_client({("POST", "/admin/embed-all"): (200, {"queued": 15})})
        async with client:
            resp = await client.embed_all()
        assert resp.queued == 15

    @pytest.mark.asyncio
    async def test_tree_index_all(self):
        client = _make_async_client(
            {
                ("POST", "/admin/tree-index-all"): (
                    200,
                    {"queued": 7, "overlap": 0.2, "force": False, "min_chars": 700},
                )
            }
        )
        async with client:
            resp = await client.tree_index_all()
        assert resp.queued == 7


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAsyncErrors:
    @pytest.mark.asyncio
    async def test_server_error(self):
        client = _make_async_client({("POST", "/articles"): (500, {"error": "internal error"})})
        async with client:
            with pytest.raises(CovalenceServerError):
                await client.create_article("body")
