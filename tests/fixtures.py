"""Shared fixture data for tests."""

from __future__ import annotations

SOURCE_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
ARTICLE_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
EDGE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
CONTENTION_ID = "b6d2f3a1-0000-4000-8000-000000000001"
MEMORY_ID = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
SESSION_ID = "c2fe46a9-0000-4000-8000-000000000099"
JOB_ID = "550e8400-e29b-41d4-a716-446655440000"
QUEUE_ID = "d290f1ee-6c54-4b01-90e6-d701748f0851"

SOURCE_OBJ = {
    "id": SOURCE_ID,
    "node_type": "source",
    "title": "Rust memory model overview",
    "content": "Rust's ownership system guarantees memory safety without a garbage collector.",
    "source_type": "document",
    "status": "active",
    "confidence": 0.85,
    "reliability": 0.85,
    "fingerprint": "a3f5e9abc",
    "metadata": {},
    "version": 1,
    "created_at": "2026-03-01T10:00:00Z",
    "modified_at": "2026-03-01T10:00:00Z",
}

ARTICLE_OBJ = {
    "id": ARTICLE_ID,
    "node_type": "article",
    "title": "Rust Memory Safety",
    "content": "Rust guarantees memory safety through its ownership and borrowing system.",
    "status": "active",
    "confidence": 0.72,
    "epistemic_type": "semantic",
    "domain_path": ["rust", "memory", "safety"],
    "metadata": {},
    "version": 3,
    "pinned": False,
    "usage_score": 4.2,
    "contention_count": 1,
    "created_at": "2026-03-01T10:00:00Z",
    "modified_at": "2026-03-01T12:30:00Z",
}

EDGE_OBJ = {
    "id": EDGE_ID,
    "age_id": 12345,
    "source_node_id": SOURCE_ID,
    "target_node_id": ARTICLE_ID,
    "edge_type": "CONFIRMS",
    "weight": 1.0,
    "confidence": 0.9,
    "metadata": {},
    "created_at": "2026-03-01T10:00:00Z",
    "created_by": "agent_explicit",
}

CONTENTION_OBJ = {
    "id": CONTENTION_ID,
    "node_id": ARTICLE_ID,
    "source_node_id": SOURCE_ID,
    "description": "Source claims Rust uses GC; article states it does not",
    "status": "detected",
    "resolution": None,
    "severity": "high",
    "detected_at": "2026-03-01T11:00:00Z",
    "resolved_at": None,
}

MEMORY_OBJ = {
    "id": MEMORY_ID,
    "content": "The user prefers dark mode in all UI contexts.",
    "tags": ["preferences", "ui"],
    "importance": 0.8,
    "context": "conversation:user",
    "confidence": 0.72,
    "created_at": "2026-03-01T10:00:00Z",
    "forgotten": False,
}

SESSION_OBJ = {
    "id": SESSION_ID,
    "label": "research-session-2026-03-01",
    "status": "open",
    "created_at": "2026-03-01T10:00:00Z",
    "last_active_at": "2026-03-01T10:15:00Z",
    "metadata": {},
}

QUEUE_ENTRY_OBJ = {
    "id": QUEUE_ID,
    "task_type": "embed",
    "node_id": ARTICLE_ID,
    "status": "pending",
    "priority": 3,
    "created_at": "2026-03-01T10:00:00Z",
    "started_at": None,
    "completed_at": None,
}

ADMIN_STATS_OBJ = {
    "nodes": {
        "total": 1420,
        "sources": 980,
        "articles": 430,
        "sessions": 10,
        "active": 1350,
        "archived": 70,
        "pinned": 5,
    },
    "edges": {
        "sql_count": 3200,
        "age_count": 3200,
        "in_sync": True,
    },
    "queue": {
        "pending": 12,
        "processing": 2,
        "failed": 1,
        "completed_24h": 348,
    },
    "embeddings": {
        "total": 1300,
        "nodes_without": 50,
    },
}
