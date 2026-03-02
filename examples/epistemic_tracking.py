"""Epistemic tracking example: contentions, provenance, and trace.

This example demonstrates how to:
  - Detect and resolve contentions between sources and articles
  - Walk the provenance graph of an article
  - Trace a specific claim back to its source evidence

Run against a live Covalence engine:
    python examples/epistemic_tracking.py
"""

from __future__ import annotations

from covalence import CovalenceClient
from covalence.exceptions import CovalenceNotFoundError
from covalence.models import ContentionResolution, ContentionStatus, EdgeLabel, SourceType


def main() -> None:
    with CovalenceClient("http://localhost:8430") as client:
        # ── 1. Ingest two conflicting sources ────────────────────────────────
        print("Ingesting authoritative source…")
        good_source = client.ingest_source(
            "Rust does NOT use a garbage collector. Memory safety is enforced at "
            "compile time through the ownership and borrowing system.",
            source_type=SourceType.document,
            title="Official Rust Book: Ownership",
            reliability=0.95,
        )
        print(f"  → {good_source.id}")

        print("Ingesting conflicting (blog) source…")
        bad_source = client.ingest_source(
            "Rust uses a modern garbage collector to manage memory automatically.",
            source_type=SourceType.web,
            title="Uncited Blog Post",
            reliability=0.3,
        )
        print(f"  → {bad_source.id}")

        # ── 2. Create an article from the authoritative source ───────────────
        print("Creating knowledge article…")
        article = client.create_article(
            "Rust guarantees memory safety without a garbage collector. "
            "The ownership model and borrow checker enforce these rules at compile time.",
            title="Rust Memory Model",
            domain_path=["rust", "memory"],
            source_ids=[good_source.id],
        )
        print(f"  → Article: {article.id}")

        # ── 3. Manually create a CONTRADICTS edge ────────────────────────────
        print("Creating CONTRADICTS edge…")
        edge = client.create_edge(
            bad_source.id,
            article.id,
            EdgeLabel.CONTRADICTS,
            confidence=0.9,
            notes="Blog claims Rust uses GC; article states it does not.",
        )
        print(f"  → Edge: {edge.id}  ({edge.edge_type})")

        # ── 4. List open contentions ─────────────────────────────────────────
        print("Listing detected contentions…")
        contentions = client.list_contentions(
            node_id=article.id,
            status=ContentionStatus.detected,
        )
        print(f"  → {len(contentions.data)} contention(s) found")
        for c in contentions.data:
            print(f"     [{c.severity}] {c.description}")

        # ── 5. Resolve a contention ──────────────────────────────────────────
        if contentions.data:
            c0 = contentions.data[0]
            print(f"Resolving contention {c0.id}…")
            resolved = client.resolve_contention(
                c0.id,
                resolution=ContentionResolution.supersede_a,
                rationale=(
                    "The conflicting source is an uncited blog post with low reliability. "
                    "The article cites the official Rust documentation."
                ),
            )
            print(f"  → Status: {resolved.status.value}  resolution: {resolved.resolution}")

        # ── 6. Walk provenance chain ─────────────────────────────────────────
        print("Walking provenance graph…")
        prov = client.get_provenance(article.id, max_depth=3)
        for entry in prov.data:
            node_id = entry.source_node.get("id", "?")
            title = entry.source_node.get("title", "(untitled)")
            print(f"  depth={entry.depth}  {entry.edge_type}  ← {title!r} ({node_id})")

        # ── 7. Trace a specific claim ─────────────────────────────────────────
        print("Tracing claim to sources…")
        trace = client.trace_claim(article.id, "Rust has no garbage collector")
        for result in trace.data:
            print(f"  [{result.score:.3f}] {result.title!r}")
            if result.snippet:
                print(f"     …{result.snippet[:80]}…")

        # ── 8. Graph neighborhood ─────────────────────────────────────────────
        print(f"Neighborhood of article {article.id}…")
        neighborhood = client.get_neighborhood(article.id, depth=2, limit=10)
        print(f"  → {neighborhood.meta.count} neighbor(s)")
        for entry in neighborhood.data:
            n = entry.node
            print(f"     depth={entry.depth}  {n.get('node_type')}  {n.get('title', '?')!r}")


if __name__ == "__main__":
    main()
