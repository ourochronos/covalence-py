"""Basic usage example: ingest a source, create an article, compile, and search.

Run against a live Covalence engine:
    python examples/basic_usage.py
"""

from __future__ import annotations

from covalence import CovalenceClient
from covalence.models import SearchIntent, SourceType


def main() -> None:
    with CovalenceClient("http://localhost:8430") as client:
        # ── 1. Ingest a source ───────────────────────────────────────────────
        print("Ingesting source…")
        source = client.ingest_source(
            "Rust's ownership system guarantees memory safety without a garbage collector. "
            "The borrow checker enforces these rules at compile time, eliminating entire "
            "classes of bugs such as use-after-free and data races.",
            source_type=SourceType.document,
            title="Rust Memory Safety Overview",
            metadata={"url": "https://doc.rust-lang.org/book/", "author": "Steve Klabnik"},
            reliability=0.90,
        )
        print(f"  → Source: {source.id}  (confidence={source.confidence})")

        # ── 2. Create an article directly ────────────────────────────────────
        print("Creating article…")
        article = client.create_article(
            "Rust guarantees memory safety through its ownership and borrowing system. "
            "There is no garbage collector; instead the borrow checker runs at compile time.",
            title="Rust Memory Safety",
            domain_path=["rust", "memory", "safety"],
            source_ids=[source.id],
        )
        print(f"  → Article: {article.id}  (version={article.version})")

        # ── 3. Compile sources into an article (async job) ───────────────────
        print("Compiling sources into a new article (async)…")
        job = client.compile_article(
            [source.id],
            title_hint="Rust memory management patterns",
        )
        print(f"  → Compile job: {job.job_id}  status={job.status}")
        print("     Poll GET /admin/queue/{job_id} until status='completed'.")

        # ── 4. Unified search ────────────────────────────────────────────────
        print("Searching…")
        results = client.search(
            "how does Rust prevent memory leaks",
            intent=SearchIntent.factual,
            node_types=["article"],
            limit=5,
        )
        print(f"  → {len(results.data)} result(s)")
        for r in results.data:
            print(f"     [{r.score:.3f}] {r.title!r}  ({r.node_type})")

        # ── 5. List articles ─────────────────────────────────────────────────
        listing = client.list_articles(limit=10)
        print(f"Active articles in KB: {listing.meta.count}")

        # ── 6. Get admin stats ───────────────────────────────────────────────
        stats = client.get_admin_stats()
        print(
            f"System stats → nodes={stats.nodes.total}, "
            f"edges={stats.edges.sql_count}, "
            f"queue_pending={stats.queue.pending}"
        )


if __name__ == "__main__":
    main()
