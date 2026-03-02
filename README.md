# covalence-client

Python client library for the [Covalence](https://github.com/ourochronos/covalence) knowledge engine API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Full API coverage** — all 38 endpoints across Sources, Articles, Search, Edges, Contentions, Memory, Sessions, and Admin
- **Sync & async** — `CovalenceClient` (httpx sync) and `AsyncCovalenceClient` (httpx async)
- **Typed** — Pydantic v2 models for every request/response shape; full `py.typed` PEP 561 marker
- **Error handling** — typed exception hierarchy (`CovalenceBadRequestError`, `CovalenceNotFoundError`, etc.)
- **Python 3.10+** compatible

---

## Installation

```bash
pip install covalence-client
```

Or from source:

```bash
git clone https://github.com/ourochronos/covalence-py
cd covalence-py
pip install -e ".[dev]"
```

---

## Quick Start

### Synchronous

```python
from covalence import CovalenceClient

with CovalenceClient("http://localhost:8430") as client:
    # Ingest a source
    source = client.ingest_source(
        "Rust's ownership system guarantees memory safety.",
        title="Rust Memory Overview",
        source_type="document",
        reliability=0.9,
    )

    # Create an article
    article = client.create_article(
        "Rust achieves memory safety without a GC via ownership.",
        title="Rust Memory Safety",
        domain_path=["rust", "memory"],
        source_ids=[source.id],
    )

    # Search
    results = client.search("Rust memory safety", limit=5)
    for r in results.data:
        print(f"[{r.score:.3f}] {r.title}")
```

### Asynchronous

```python
import asyncio
from covalence import AsyncCovalenceClient

async def main():
    async with AsyncCovalenceClient("http://localhost:8430") as client:
        source = await client.ingest_source("Hello, Covalence!")
        results = await client.search("greeting")
        print(results.data)

asyncio.run(main())
```

---

## API Overview

### Sources

| Method | Description |
|--------|-------------|
| `ingest_source(content, ...)` | Ingest a raw source document (idempotent by SHA-256) |
| `list_sources(limit, cursor, ...)` | Paginated list of sources |
| `get_source(source_id)` | Fetch a source by UUID |
| `delete_source(source_id)` | Hard-delete a source (irreversible) |

### Articles

| Method | Description |
|--------|-------------|
| `create_article(content, ...)` | Create an article directly |
| `compile_article(source_ids, ...)` | Async LLM compilation → returns job |
| `merge_articles(id_a, id_b)` | Merge two articles (originals archived) |
| `list_articles(limit, ...)` | Paginated list of articles |
| `get_article(article_id)` | Fetch an article with live contention count |
| `update_article(article_id, ...)` | Partial update (content, title, domain_path, pinned) |
| `archive_article(article_id)` | Soft-delete (status → archived) |
| `split_article(article_id)` | Split into two parts at paragraph boundary |
| `get_provenance(article_id, ...)` | Walk provenance graph backward |
| `trace_claim(article_id, claim_text)` | Rank sources by TF-IDF similarity to a claim |

### Search

| Method | Description |
|--------|-------------|
| `search(query, ...)` | Three-dimensional search (vector + lexical + graph) |

### Edges / Graph

| Method | Description |
|--------|-------------|
| `create_edge(from_node_id, to_node_id, label, ...)` | Create a typed edge |
| `delete_edge(edge_id)` | Remove an edge |
| `list_node_edges(node_id, ...)` | List edges for a node |
| `get_neighborhood(node_id, ...)` | Graph neighborhood traversal |

### Contentions

| Method | Description |
|--------|-------------|
| `list_contentions(node_id, status)` | List detected/resolved contentions |
| `get_contention(contention_id)` | Fetch a single contention |
| `resolve_contention(id, resolution, rationale)` | Resolve or dismiss a contention |

### Memory

| Method | Description |
|--------|-------------|
| `store_memory(content, ...)` | Store an agent observation memory |
| `recall_memories(query, ...)` | Full-text search over active memories |
| `get_memory_status()` | Aggregate memory counts |
| `forget_memory(memory_id, ...)` | Soft-delete a memory |

### Sessions

| Method | Description |
|--------|-------------|
| `create_session(label, ...)` | Create a new session context |
| `list_sessions(status, ...)` | List sessions |
| `get_session(session_id)` | Fetch a session |
| `close_session(session_id)` | Close a session |

### Admin

| Method | Description |
|--------|-------------|
| `get_admin_stats()` | System health snapshot |
| `run_maintenance(...)` | Trigger recompute_scores, eviction, etc. |
| `list_queue(status, ...)` | List background task queue entries |
| `get_queue_entry(entry_id)` | Fetch a queue entry |
| `retry_queue_entry(entry_id)` | Reset a failed entry to pending |
| `delete_queue_entry(entry_id)` | Delete a queue entry |
| `embed_all()` | Queue embeddings for all unembedded nodes |
| `tree_index_all(...)` | Queue tree-index tasks for eligible nodes |

---

## Error Handling

```python
from covalence import CovalenceClient
from covalence.exceptions import (
    CovalenceNotFoundError,
    CovalenceBadRequestError,
    CovalenceServerError,
    CovalenceConnectionError,
)

with CovalenceClient() as client:
    try:
        article = client.get_article("non-existent-id")
    except CovalenceNotFoundError:
        print("Article not found")
    except CovalenceBadRequestError as e:
        print(f"Bad request: {e.message}")
    except CovalenceServerError:
        print("Server error — try again later")
    except CovalenceConnectionError:
        print("Could not reach the Covalence engine")
```

All exceptions carry `.status_code` and `.response_body` attributes.

---

## Development

```bash
pip install -e ".[dev]"
ruff format .
python -m pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
