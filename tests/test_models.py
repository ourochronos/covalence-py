"""Model validation tests."""

from __future__ import annotations

import pytest

from covalence.models import (
    Article,
    ArticleCreateRequest,
    Contention,
    ContentionResolveRequest,
    ContentionStatus,
    Edge,
    EdgeCreateRequest,
    EdgeLabel,
    EpistemicType,
    Memory,
    MemoryStoreRequest,
    NodeStatus,
    QueueEntry,
    QueueStatus,
    SearchRequest,
    SearchWeights,
    Session,
    SessionStatus,
    Source,
    SourceIngestRequest,
    SourceType,
    SplitResult,
    TraceRequest,
    TreeIndexAllRequest,
)
from tests.fixtures import (
    ARTICLE_ID,
    ARTICLE_OBJ,
    CONTENTION_OBJ,
    EDGE_OBJ,
    MEMORY_OBJ,
    QUEUE_ENTRY_OBJ,
    SESSION_OBJ,
    SOURCE_ID,
    SOURCE_OBJ,
)


class TestSourceModel:
    def test_parse_full(self):
        s = Source.model_validate(SOURCE_OBJ)
        assert s.id == SOURCE_ID
        assert s.source_type == "document"
        assert s.status == NodeStatus.active
        assert s.confidence == 0.85
        assert s.version == 1

    def test_defaults(self):
        s = Source(id="abc", content="hello")
        assert s.node_type == "source"
        assert s.status == NodeStatus.active
        assert s.metadata == {}

    def test_ingest_request_minimal(self):
        r = SourceIngestRequest(content="test")
        dumped = r.model_dump(exclude_none=True)
        assert dumped == {"content": "test"}

    def test_ingest_request_full(self):
        r = SourceIngestRequest(
            content="test",
            source_type=SourceType.document,
            title="T",
            metadata={"k": "v"},
            session_id="session-1",
            reliability=0.9,
        )
        d = r.model_dump(exclude_none=True)
        assert d["source_type"] == "document"
        assert d["reliability"] == 0.9

    def test_ingest_request_reliability_bounds(self):
        with pytest.raises(Exception):
            SourceIngestRequest(content="x", reliability=1.5)
        with pytest.raises(Exception):
            SourceIngestRequest(content="x", reliability=-0.1)


class TestArticleModel:
    def test_parse_full(self):
        a = Article.model_validate(ARTICLE_OBJ)
        assert a.id == ARTICLE_ID
        assert a.epistemic_type == EpistemicType.semantic
        assert a.domain_path == ["rust", "memory", "safety"]
        assert a.pinned is False
        assert a.usage_score == 4.2

    def test_defaults(self):
        a = Article(id="x", content="body")
        assert a.node_type == "article"
        assert a.status == NodeStatus.active
        assert a.epistemic_type == EpistemicType.semantic
        assert a.domain_path == []
        assert a.contention_count == 0

    def test_create_request_minimal(self):
        r = ArticleCreateRequest(content="body")
        d = r.model_dump(exclude_none=True)
        assert d == {"content": "body"}

    def test_create_request_with_source_ids(self):
        r = ArticleCreateRequest(content="body", source_ids=["s1", "s2"])
        assert r.source_ids == ["s1", "s2"]

    def test_epistemic_type_enum(self):
        for val in ("semantic", "episodic", "procedural", "declarative"):
            a = Article(id="x", content="b", epistemic_type=val)
            assert a.epistemic_type == EpistemicType(val)


class TestEdgeModel:
    def test_parse_full(self):
        e = Edge.model_validate(EDGE_OBJ)
        assert e.edge_type == "CONFIRMS"
        assert e.confidence == 0.9
        assert e.age_id == 12345

    def test_create_request_label_enum(self):
        r = EdgeCreateRequest(
            from_node_id="a",
            to_node_id="b",
            label=EdgeLabel.CONFIRMS,
        )
        assert r.label == EdgeLabel.CONFIRMS

    def test_edge_label_values(self):
        assert EdgeLabel.ORIGINATES.value == "ORIGINATES"
        assert EdgeLabel.SUPERSEDES.value == "SUPERSEDES"
        assert EdgeLabel.MERGED_FROM.value == "MERGED_FROM"


class TestContentionModel:
    def test_parse_full(self):
        c = Contention.model_validate(CONTENTION_OBJ)
        assert c.status == ContentionStatus.detected
        assert c.severity == "high"
        assert c.resolved_at is None

    def test_resolve_request(self):
        from covalence.models import ContentionResolution

        r = ContentionResolveRequest(resolution=ContentionResolution.supersede_a, rationale="reason")
        d = r.model_dump()
        assert d["resolution"] == "supersede_a"


class TestMemoryModel:
    def test_parse_full(self):
        m = Memory.model_validate(MEMORY_OBJ)
        assert m.tags == ["preferences", "ui"]
        assert m.importance == 0.8
        assert m.forgotten is False

    def test_store_request_minimal(self):
        r = MemoryStoreRequest(content="remember this")
        assert r.content == "remember this"
        assert r.tags is None

    def test_store_request_importance_bounds(self):
        with pytest.raises(Exception):
            MemoryStoreRequest(content="x", importance=2.0)


class TestSessionModel:
    def test_parse(self):
        s = Session.model_validate(SESSION_OBJ)
        assert s.status == SessionStatus.open
        assert s.label == "research-session-2026-03-01"

    def test_defaults(self):
        s = Session(id="s1")
        assert s.status == SessionStatus.open
        assert s.metadata == {}


class TestQueueEntryModel:
    def test_parse(self):
        q = QueueEntry.model_validate(QUEUE_ENTRY_OBJ)
        assert q.status == QueueStatus.pending
        assert q.task_type == "embed"
        assert q.priority == 3


class TestSearchModel:
    def test_minimal(self):
        r = SearchRequest(query="test")
        d = r.model_dump(exclude_none=True)
        assert d == {"query": "test"}

    def test_with_weights(self):
        r = SearchRequest(
            query="test",
            weights=SearchWeights(vector=0.7, lexical=0.2, graph=0.1),
        )
        assert r.weights is not None
        assert r.weights.vector == 0.7


class TestSplitResult:
    def test_parse(self):
        data = {
            "original_id": ARTICLE_ID,
            "part_a": {**ARTICLE_OBJ, "id": "part-a-id", "title": "Rust Memory Safety (Part 1)"},
            "part_b": {**ARTICLE_OBJ, "id": "part-b-id", "title": "Rust Memory Safety (Part 2)"},
        }
        sr = SplitResult.model_validate(data)
        assert sr.original_id == ARTICLE_ID
        assert sr.part_a.id == "part-a-id"
        assert sr.part_b.id == "part-b-id"


class TestTraceRequest:
    def test_fields(self):
        r = TraceRequest(claim_text="ownership eliminates dangling pointer bugs")
        assert r.claim_text == "ownership eliminates dangling pointer bugs"


class TestTreeIndexAllRequest:
    def test_exclude_none(self):
        r = TreeIndexAllRequest()
        d = r.model_dump(exclude_none=True)
        assert d == {}

    def test_with_values(self):
        r = TreeIndexAllRequest(overlap=0.15, force=True, min_chars=500)
        d = r.model_dump(exclude_none=True)
        assert d["overlap"] == 0.15
        assert d["force"] is True
        assert d["min_chars"] == 500
