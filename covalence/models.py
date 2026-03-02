"""Pydantic v2 models for the Covalence REST API.

All models map directly to the shapes documented in the API reference.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    document = "document"
    code = "code"
    tool_output = "tool_output"
    user_input = "user_input"
    web = "web"
    conversation = "conversation"
    observation = "observation"


class NodeStatus(str, Enum):
    active = "active"
    archived = "archived"
    tombstone = "tombstone"


class EpistemicType(str, Enum):
    semantic = "semantic"
    episodic = "episodic"
    procedural = "procedural"
    declarative = "declarative"


class SearchIntent(str, Enum):
    factual = "factual"
    temporal = "temporal"
    causal = "causal"
    entity = "entity"


class ContentionStatus(str, Enum):
    detected = "detected"
    resolved = "resolved"
    dismissed = "dismissed"


class ContentionResolution(str, Enum):
    supersede_a = "supersede_a"
    supersede_b = "supersede_b"
    accept_both = "accept_both"
    dismiss = "dismiss"


class SessionStatus(str, Enum):
    open = "open"
    closed = "closed"


class QueueStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    failed = "failed"
    completed = "completed"


class QueueTaskType(str, Enum):
    embed = "embed"
    contention_check = "contention_check"
    compile = "compile"
    tree_index = "tree_index"
    tree_embed = "tree_embed"


class EdgeLabel(str, Enum):
    # Provenance
    ORIGINATES = "ORIGINATES"
    CONFIRMS = "CONFIRMS"
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    CONTENDS = "CONTENDS"
    EXTENDS = "EXTENDS"
    DERIVES_FROM = "DERIVES_FROM"
    MERGED_FROM = "MERGED_FROM"
    SPLIT_INTO = "SPLIT_INTO"
    SPLIT_FROM = "SPLIT_FROM"
    # Temporal
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    CONCURRENT_WITH = "CONCURRENT_WITH"
    # Causal
    CAUSES = "CAUSES"
    MOTIVATED_BY = "MOTIVATED_BY"
    IMPLEMENTS = "IMPLEMENTS"
    # Semantic
    RELATES_TO = "RELATES_TO"
    GENERALIZES = "GENERALIZES"
    # Session
    CAPTURED_IN = "CAPTURED_IN"
    # Entity
    INVOLVES = "INVOLVES"
    # Legacy aliases
    COMPILED_FROM = "COMPILED_FROM"
    ELABORATES = "ELABORATES"


class EdgeDirection(str, Enum):
    outbound = "outbound"
    inbound = "inbound"


# ---------------------------------------------------------------------------
# Shared / Base models
# ---------------------------------------------------------------------------


class Meta(BaseModel):
    count: int


# ---------------------------------------------------------------------------
# Source models
# ---------------------------------------------------------------------------


class Source(BaseModel):
    id: str
    node_type: Literal["source"] = "source"
    title: str | None = None
    content: str | None = None
    source_type: str | None = None
    status: NodeStatus = NodeStatus.active
    confidence: float = 0.5
    reliability: float = 0.5
    fingerprint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: datetime | None = None
    modified_at: datetime | None = None


class SourceIngestRequest(BaseModel):
    content: str
    source_type: SourceType | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None
    session_id: str | None = None
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceListResponse(BaseModel):
    data: list[Source]
    meta: Meta


class SourceResponse(BaseModel):
    data: Source


# ---------------------------------------------------------------------------
# Article models
# ---------------------------------------------------------------------------


class Article(BaseModel):
    id: str
    node_type: Literal["article"] = "article"
    title: str | None = None
    content: str | None = None
    status: NodeStatus = NodeStatus.active
    confidence: float = 0.5
    epistemic_type: EpistemicType = EpistemicType.semantic
    domain_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    pinned: bool = False
    usage_score: float = 0.0
    contention_count: int = 0
    created_at: datetime | None = None
    modified_at: datetime | None = None


class ArticleCreateRequest(BaseModel):
    content: str
    title: str | None = None
    domain_path: list[str] | None = None
    epistemic_type: EpistemicType | None = None
    source_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ArticleUpdateRequest(BaseModel):
    content: str | None = None
    title: str | None = None
    domain_path: list[str] | None = None
    pinned: bool | None = None


class ArticleCompileRequest(BaseModel):
    source_ids: list[str]
    title_hint: str | None = None


class ArticleMergeRequest(BaseModel):
    article_id_a: str
    article_id_b: str


class CompileJob(BaseModel):
    job_id: str
    status: str


class CompileJobResponse(BaseModel):
    data: CompileJob


class ArticleResponse(BaseModel):
    data: Article


class ArticleListResponse(BaseModel):
    data: list[Article]
    meta: Meta


class SplitResult(BaseModel):
    original_id: str
    part_a: Article
    part_b: Article


class SplitResponse(BaseModel):
    data: SplitResult


# ---------------------------------------------------------------------------
# Provenance / trace models
# ---------------------------------------------------------------------------


class ProvenanceEntry(BaseModel):
    source_node: dict[str, Any]
    edge_type: str
    confidence: float
    depth: int


class ProvenanceResponse(BaseModel):
    data: list[ProvenanceEntry]


class TraceRequest(BaseModel):
    claim_text: str


class TraceResult(BaseModel):
    source_id: str
    title: str | None = None
    score: float
    snippet: str | None = None


class TraceResponse(BaseModel):
    data: list[TraceResult]
    meta: Meta | None = None


# ---------------------------------------------------------------------------
# Search models
# ---------------------------------------------------------------------------


class SearchWeights(BaseModel):
    vector: float | None = None
    lexical: float | None = None
    graph: float | None = None


class SearchRequest(BaseModel):
    query: str
    embedding: list[float] | None = None
    intent: SearchIntent | None = None
    session_id: str | None = None
    node_types: list[str] | None = None
    limit: int | None = None
    weights: SearchWeights | None = None


class SearchResult(BaseModel):
    node_id: str
    score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    graph_score: float | None = None
    confidence: float | None = None
    node_type: str | None = None
    title: str | None = None
    content_preview: str | None = None


class SearchMeta(BaseModel):
    total_results: int | None = None
    lexical_backend: str | None = None
    dimensions_used: list[str] | None = None
    elapsed_ms: int | None = None


class SearchResponse(BaseModel):
    data: list[SearchResult]
    meta: SearchMeta | None = None


# ---------------------------------------------------------------------------
# Edge models
# ---------------------------------------------------------------------------


class Edge(BaseModel):
    id: str
    age_id: int | None = None
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    created_by: str | None = None


class EdgeCreateRequest(BaseModel):
    from_node_id: str
    to_node_id: str
    label: EdgeLabel
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: str | None = None
    notes: str | None = None


class EdgeResponse(BaseModel):
    data: Edge


class EdgeListResponse(BaseModel):
    data: list[Edge]
    meta: Meta


# ---------------------------------------------------------------------------
# Graph neighborhood models
# ---------------------------------------------------------------------------


class NeighborhoodEntry(BaseModel):
    node: dict[str, Any]
    edge: dict[str, Any]
    depth: int


class NeighborhoodResponse(BaseModel):
    data: list[NeighborhoodEntry]
    meta: Meta


# ---------------------------------------------------------------------------
# Contention models
# ---------------------------------------------------------------------------


class Contention(BaseModel):
    id: str
    node_id: str
    source_node_id: str
    description: str | None = None
    status: ContentionStatus = ContentionStatus.detected
    resolution: str | None = None
    severity: str | None = None
    detected_at: datetime | None = None
    resolved_at: datetime | None = None


class ContentionResolveRequest(BaseModel):
    resolution: ContentionResolution
    rationale: str


class ContentionResponse(BaseModel):
    data: Contention


class ContentionListResponse(BaseModel):
    data: list[Contention]


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    context: str | None = None
    confidence: float = 0.6
    created_at: datetime | None = None
    forgotten: bool = False


class MemoryStoreRequest(BaseModel):
    content: str
    tags: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    context: str | None = None
    supersedes_id: str | None = None


class MemoryRecallRequest(BaseModel):
    query: str
    limit: int | None = None
    tags: list[str] | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MemoryForgetRequest(BaseModel):
    reason: str | None = None


class MemoryStatus(BaseModel):
    total_memories: int
    active_memories: int
    forgotten_memories: int


class MemoryResponse(BaseModel):
    data: Memory


class MemoryListResponse(BaseModel):
    data: list[Memory]


class MemoryStatusResponse(BaseModel):
    data: MemoryStatus


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class Session(BaseModel):
    id: str
    label: str | None = None
    status: SessionStatus = SessionStatus.open
    created_at: datetime | None = None
    last_active_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCreateRequest(BaseModel):
    label: str | None = None
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    data: Session


class SessionListResponse(BaseModel):
    data: list[Session]


# ---------------------------------------------------------------------------
# Admin models
# ---------------------------------------------------------------------------


class AdminNodeStats(BaseModel):
    total: int
    sources: int
    articles: int
    sessions: int
    active: int
    archived: int
    pinned: int


class AdminEdgeStats(BaseModel):
    sql_count: int
    age_count: int
    in_sync: bool


class AdminQueueStats(BaseModel):
    pending: int
    processing: int
    failed: int
    completed_24h: int


class AdminEmbeddingStats(BaseModel):
    total: int
    nodes_without: int


class AdminStats(BaseModel):
    nodes: AdminNodeStats
    edges: AdminEdgeStats
    queue: AdminQueueStats
    embeddings: AdminEmbeddingStats


class AdminStatsResponse(BaseModel):
    data: AdminStats


class MaintenanceRequest(BaseModel):
    recompute_scores: bool | None = None
    process_queue: bool | None = None
    evict_if_over_capacity: bool | None = None
    evict_count: int | None = None


class MaintenanceResult(BaseModel):
    actions_taken: list[str]


class MaintenanceResponse(BaseModel):
    data: MaintenanceResult


class QueueEntry(BaseModel):
    id: str
    task_type: str
    node_id: str | None = None
    status: QueueStatus
    priority: int | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class QueueEntryResponse(BaseModel):
    data: QueueEntry


class QueueListResponse(BaseModel):
    data: list[QueueEntry]


class EmbedAllResponse(BaseModel):
    queued: int


class TreeIndexAllRequest(BaseModel):
    overlap: float | None = None
    force: bool | None = None
    min_chars: int | None = None


class TreeIndexAllResponse(BaseModel):
    queued: int
    overlap: float | None = None
    force: bool | None = None
    min_chars: int | None = None
