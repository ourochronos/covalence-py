"""Agent memory example: store, recall, supersede, and forget memories.

This example demonstrates the Memory API — a purpose-built wrapper around
source nodes tagged with ``metadata.memory = true``.

Run against a live Covalence engine:
    python examples/agent_memory.py
"""

from __future__ import annotations

from covalence import CovalenceClient


def main() -> None:
    with CovalenceClient("http://localhost:8430") as client:
        # ── 1. Store a memory ────────────────────────────────────────────────
        print("Storing initial memory…")
        mem1 = client.store_memory(
            "The user prefers dark mode in all UI contexts.",
            tags=["preferences", "ui"],
            importance=0.8,
            context="conversation:user",
        )
        print(f"  → Memory stored: {mem1.id}  (confidence={mem1.confidence:.2f})")

        # ── 2. Store another memory and supersede the first ──────────────────
        print("Storing updated preference (supersedes previous)…")
        mem2 = client.store_memory(
            "The user now prefers light mode when working outdoors.",
            tags=["preferences", "ui"],
            importance=0.85,
            context="conversation:user",
            supersedes_id=mem1.id,
        )
        print(f"  → New memory: {mem2.id}  supersedes {mem1.id}")

        # ── 3. Recall memories ───────────────────────────────────────────────
        print("Recalling UI preference memories…")
        results = client.recall_memories(
            "user interface preferences",
            tags=["ui"],
            min_confidence=0.5,
            limit=5,
        )
        for m in results.data:
            print(f"  [{m.confidence:.2f}] {m.content!r}  forgotten={m.forgotten}")

        # ── 4. Memory system status ──────────────────────────────────────────
        status = client.get_memory_status()
        print(
            f"Memory status → total={status.total_memories}, "
            f"active={status.active_memories}, "
            f"forgotten={status.forgotten_memories}"
        )

        # ── 5. Forget an outdated memory ─────────────────────────────────────
        print(f"Forgetting memory {mem1.id}…")
        client.forget_memory(mem1.id, reason="Superseded by newer preference observation.")
        print("  → Done.")

        # ── 6. Recall again — forgotten memory should not appear ─────────────
        print("Recalling again after forget…")
        results2 = client.recall_memories("user interface preferences", tags=["ui"])
        for m in results2.data:
            print(f"  [{m.confidence:.2f}] {m.content!r}")
        print(f"  (Expected 1 result; got {len(results2.data)})")


if __name__ == "__main__":
    main()
