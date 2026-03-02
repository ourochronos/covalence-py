"""Covalence CLI — interact with your Covalence knowledge engine from the terminal."""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from .client import CovalenceClient
from .exceptions import CovalenceError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

console = Console()
err_console = Console(stderr=True)

DEFAULT_URL = "http://localhost:8430"


def _get_client(ctx: click.Context) -> CovalenceClient:
    return CovalenceClient(base_url=ctx.obj["url"])


def _as_json(ctx: click.Context) -> bool:
    return ctx.obj.get("as_json", False)


def _dump(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model (or plain dict) to a serialisable dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return dict(obj)


def _print_json(data: Any) -> None:
    console.print(JSON(json.dumps(data, default=str)))


def _handle_error(exc: Exception) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--url",
    envvar="COVALENCE_URL",
    default=DEFAULT_URL,
    show_default=True,
    help="Covalence server URL (env: COVALENCE_URL).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output raw JSON instead of rich tables.",
)
@click.pass_context
def cli(ctx: click.Context, url: str, as_json: bool) -> None:
    """Covalence CLI — manage your knowledge engine."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["as_json"] = as_json


# ===========================================================================
# status
# ===========================================================================


@cli.command("status")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def cmd_status(ctx: click.Context, as_json: bool) -> None:
    """Show admin stats (nodes, edges, queue, embeddings)."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        stats = client.get_admin_stats()
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(stats))
        return

    # --- pretty print ---
    console.print("[bold cyan]Covalence System Status[/bold cyan]\n")

    nodes = stats.nodes
    tbl = Table(title="Nodes", show_header=True, header_style="bold green")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Total", str(nodes.total))
    tbl.add_row("Sources", str(nodes.sources))
    tbl.add_row("Articles", str(nodes.articles))
    tbl.add_row("Sessions", str(nodes.sessions))
    tbl.add_row("Active", str(nodes.active))
    tbl.add_row("Archived", str(nodes.archived))
    tbl.add_row("Pinned", str(nodes.pinned))
    console.print(tbl)

    q = stats.queue
    qtbl = Table(title="Queue", show_header=True, header_style="bold yellow")
    qtbl.add_column("Status")
    qtbl.add_column("Count", justify="right")
    qtbl.add_row("Pending", str(q.pending))
    qtbl.add_row("Processing", str(q.processing))
    qtbl.add_row("Failed", str(q.failed))
    qtbl.add_row("Completed (24 h)", str(q.completed_24h))
    console.print(qtbl)

    emb = stats.embeddings
    etbl = Table(title="Embeddings", show_header=True, header_style="bold magenta")
    etbl.add_column("Metric")
    etbl.add_column("Value", justify="right")
    etbl.add_row("Total", str(emb.total))
    etbl.add_row("Nodes without embedding", str(emb.nodes_without))
    console.print(etbl)


# ===========================================================================
# search
# ===========================================================================


@cli.command("search")
@click.argument("query")
@click.option("--limit", default=10, show_default=True, help="Max results.")
@click.option(
    "--intent",
    type=click.Choice(["factual", "temporal", "causal", "entity"]),
    default=None,
    help="Search intent hint.",
)
@click.option("--domain", default=None, help="(Informational) domain hint — passed as part of the query.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def cmd_search(
    ctx: click.Context,
    query: str,
    limit: int,
    intent: str | None,
    domain: str | None,
    as_json: bool,
) -> None:
    """Search the knowledge base for QUERY."""
    as_json = as_json or _as_json(ctx)
    effective_query = f"[{domain}] {query}" if domain else query
    try:
        client = _get_client(ctx)
        resp = client.search(effective_query, intent=intent, limit=limit)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json([_dump(r) for r in resp.data])
        return

    if not resp.data:
        console.print("[yellow]No results found.[/yellow]")
        return

    tbl = Table(title=f"Search results for: {query!r}", header_style="bold cyan")
    tbl.add_column("Score", justify="right", style="green", no_wrap=True)
    tbl.add_column("Type", style="dim")
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Title / Preview")

    for r in resp.data:
        preview = r.title or (r.content_preview or "")[:80]
        tbl.add_row(f"{r.score:.4f}", r.node_type or "", r.node_id[:8] + "…", preview)

    console.print(tbl)
    if resp.meta:
        console.print(
            f"[dim]Elapsed: {resp.meta.elapsed_ms} ms  |  "
            f"Dimensions: {', '.join(resp.meta.dimensions_used or [])}[/dim]"
        )


# ===========================================================================
# ingest
# ===========================================================================


@cli.command("ingest")
@click.argument("file", type=click.Path(exists=False), default="-", required=False)
@click.option(
    "--type",
    "source_type",
    type=click.Choice(
        ["document", "code", "tool_output", "user_input", "web", "conversation", "observation"]
    ),
    default=None,
    help="Source type.",
)
@click.option("--title", default=None, help="Human-readable title.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def cmd_ingest(
    ctx: click.Context,
    file: str,
    source_type: str | None,
    title: str | None,
    as_json: bool,
) -> None:
    """Ingest FILE (or stdin) as a new source.

    Pass '-' or omit FILE to read from stdin.

    \b
    Examples:
      covalence ingest README.md --title "Project README" --type document
      cat notes.txt | covalence ingest --type conversation
    """
    as_json = as_json or _as_json(ctx)

    if file == "-" or file is None:
        if sys.stdin.isatty():
            err_console.print("[yellow]Reading from stdin — paste content, then Ctrl-D to finish.[/yellow]")
        content = sys.stdin.read()
    else:
        try:
            with open(file) as fh:
                content = fh.read()
        except OSError as exc:
            _handle_error(exc)

    if not content.strip():
        err_console.print("[bold red]Error:[/bold red] Empty content — nothing to ingest.")
        sys.exit(1)

    try:
        client = _get_client(ctx)
        source = client.ingest_source(content, source_type=source_type, title=title)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(source))
        return

    console.print(f"[bold green]✓ Ingested source[/bold green] [cyan]{source.id}[/cyan]")
    if source.title:
        console.print(f"  Title: {source.title}")
    console.print(f"  Type:  {source.source_type}")
    console.print(f"  Size:  {len(content)} chars")


# ===========================================================================
# compile
# ===========================================================================


@cli.command("compile")
@click.argument("source_ids", nargs=-1, required=True)
@click.option("--title", "title_hint", default=None, help="Title hint for the LLM.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def cmd_compile(
    ctx: click.Context,
    source_ids: tuple[str, ...],
    title_hint: str | None,
    as_json: bool,
) -> None:
    """Compile SOURCE_IDS into a new article (async LLM job).

    \b
    Example:
      covalence compile abc123 def456 --title "My summary"
    """
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        job = client.compile_article(list(source_ids), title_hint=title_hint)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(job))
        return

    console.print("[bold green]✓ Compilation job enqueued[/bold green]")
    console.print(f"  Job ID: [cyan]{job.job_id}[/cyan]")
    console.print(f"  Status: {job.status}")
    console.print("  [dim]Poll with:[/dim] covalence article get <article-id> once completed.")


# ===========================================================================
# maintenance
# ===========================================================================


@cli.command("maintenance")
@click.option("--recompute", is_flag=True, default=False, help="Recompute usage scores.")
@click.option("--process-queue", is_flag=True, default=False, help="Process pending queue entries.")
@click.option("--evict", is_flag=True, default=False, help="Evict over-capacity articles.")
@click.option("--evict-count", default=None, type=int, help="Max articles to evict.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def cmd_maintenance(
    ctx: click.Context,
    recompute: bool,
    process_queue: bool,
    evict: bool,
    evict_count: int | None,
    as_json: bool,
) -> None:
    """Run maintenance operations on the knowledge engine.

    \b
    Flags can be combined:
      covalence maintenance --recompute --process-queue --evict
    """
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        result = client.run_maintenance(
            recompute_scores=recompute or None,
            process_queue=process_queue or None,
            evict_if_over_capacity=evict or None,
            evict_count=evict_count,
        )
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(result))
        return

    if result.actions_taken:
        console.print("[bold green]Maintenance complete. Actions taken:[/bold green]")
        for action in result.actions_taken:
            console.print(f"  • {action}")
    else:
        console.print("[yellow]Maintenance ran, but no actions were taken.[/yellow]")


# ===========================================================================
# article  (sub-group)
# ===========================================================================


@cli.group("article")
def article_group() -> None:
    """Article management commands."""


@article_group.command("get")
@click.argument("article_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def article_get(ctx: click.Context, article_id: str, as_json: bool) -> None:
    """Fetch article by ARTICLE_ID."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        article = client.get_article(article_id)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(article))
        return

    console.print(f"[bold]Article[/bold] [cyan]{article.id}[/cyan]")
    console.print(f"  Title:   {article.title or '[dim](none)[/dim]'}")
    console.print(f"  Status:  {article.status.value}")
    console.print(f"  Version: {article.version}")
    console.print(
        f"  Domain:  {' > '.join(article.domain_path) if article.domain_path else '[dim](none)[/dim]'}"
    )
    console.print(f"  Score:   {article.usage_score:.4f}")
    console.print(f"  Pinned:  {article.pinned}")
    if article.content:
        preview = article.content[:500] + ("…" if len(article.content) > 500 else "")
        console.print(f"\n[dim]{preview}[/dim]")


@article_group.command("list")
@click.option("--limit", default=20, show_default=True, help="Max results.")
@click.option("--cursor", default=None, help="Pagination cursor.")
@click.option("--status", type=click.Choice(["active", "archived", "tombstone"]), default=None)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def article_list(
    ctx: click.Context,
    limit: int,
    cursor: str | None,
    status: str | None,
    as_json: bool,
) -> None:
    """List articles."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        resp = client.list_articles(limit=limit, cursor=cursor, status=status)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json([_dump(a) for a in resp.data])
        return

    if not resp.data:
        console.print("[yellow]No articles found.[/yellow]")
        return

    tbl = Table(title=f"Articles ({resp.meta.count} total)", header_style="bold cyan")
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Status")
    tbl.add_column("V", justify="right", style="dim")
    tbl.add_column("Score", justify="right", style="green")
    tbl.add_column("Pinned")
    tbl.add_column("Title / Preview")

    for a in resp.data:
        preview = a.title or (a.content or "")[:60]
        tbl.add_row(
            a.id[:8] + "…",
            a.status.value,
            str(a.version),
            f"{a.usage_score:.3f}",
            "✓" if a.pinned else "",
            preview,
        )

    console.print(tbl)


# ===========================================================================
# source  (sub-group)
# ===========================================================================


@cli.group("source")
def source_group() -> None:
    """Source management commands."""


@source_group.command("get")
@click.argument("source_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def source_get(ctx: click.Context, source_id: str, as_json: bool) -> None:
    """Fetch source by SOURCE_ID."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        source = client.get_source(source_id)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(source))
        return

    console.print(f"[bold]Source[/bold] [cyan]{source.id}[/cyan]")
    console.print(f"  Title:      {source.title or '[dim](none)[/dim]'}")
    console.print(f"  Type:       {source.source_type or '[dim](none)[/dim]'}")
    console.print(f"  Status:     {source.status.value}")
    console.print(f"  Confidence: {source.confidence:.4f}")
    console.print(f"  Reliability:{source.reliability:.4f}")
    if source.content:
        preview = source.content[:500] + ("…" if len(source.content) > 500 else "")
        console.print(f"\n[dim]{preview}[/dim]")


@source_group.command("list")
@click.option("--limit", default=20, show_default=True, help="Max results.")
@click.option("--cursor", default=None, help="Pagination cursor.")
@click.option(
    "--type",
    "source_type",
    type=click.Choice(
        ["document", "code", "tool_output", "user_input", "web", "conversation", "observation"]
    ),
    default=None,
    help="Filter by source type.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def source_list(
    ctx: click.Context,
    limit: int,
    cursor: str | None,
    source_type: str | None,
    as_json: bool,
) -> None:
    """List sources."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        resp = client.list_sources(limit=limit, cursor=cursor, source_type=source_type)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json([_dump(s) for s in resp.data])
        return

    if not resp.data:
        console.print("[yellow]No sources found.[/yellow]")
        return

    tbl = Table(title=f"Sources ({resp.meta.count} total)", header_style="bold cyan")
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Type")
    tbl.add_column("Status")
    tbl.add_column("Reliability", justify="right", style="green")
    tbl.add_column("Title / Preview")

    for s in resp.data:
        preview = s.title or (s.content or "")[:60]
        tbl.add_row(
            s.id[:8] + "…",
            s.source_type or "",
            s.status.value,
            f"{s.reliability:.3f}",
            preview,
        )

    console.print(tbl)


# ===========================================================================
# memory  (sub-group)
# ===========================================================================


@cli.group("memory")
def memory_group() -> None:
    """Memory management commands."""


@memory_group.command("store")
@click.argument("content")
@click.option("--tags", default=None, help="Comma-separated tags.")
@click.option("--importance", default=None, type=float, help="Importance score 0.0–1.0.")
@click.option("--context", default=None, help="Provenance context string.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def memory_store(
    ctx: click.Context,
    content: str,
    tags: str | None,
    importance: float | None,
    context: str | None,
    as_json: bool,
) -> None:
    """Store CONTENT as a new memory."""
    as_json = as_json or _as_json(ctx)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    try:
        client = _get_client(ctx)
        memory = client.store_memory(content, tags=tag_list, importance=importance, context=context)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(memory))
        return

    console.print(f"[bold green]✓ Memory stored[/bold green] [cyan]{memory.id}[/cyan]")
    console.print(f"  Tags:       {', '.join(memory.tags) if memory.tags else '(none)'}")
    console.print(f"  Importance: {memory.importance:.2f}")
    console.print(f"  Confidence: {memory.confidence:.4f}")


@memory_group.command("recall")
@click.argument("query")
@click.option("--limit", default=5, show_default=True, help="Max results.")
@click.option("--tags", default=None, help="Comma-separated tag filter.")
@click.option("--min-confidence", default=None, type=float, help="Minimum confidence threshold.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def memory_recall(
    ctx: click.Context,
    query: str,
    limit: int,
    tags: str | None,
    min_confidence: float | None,
    as_json: bool,
) -> None:
    """Recall memories matching QUERY."""
    as_json = as_json or _as_json(ctx)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    try:
        client = _get_client(ctx)
        resp = client.recall_memories(query, limit=limit, tags=tag_list, min_confidence=min_confidence)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json([_dump(m) for m in resp.data])
        return

    if not resp.data:
        console.print("[yellow]No memories found.[/yellow]")
        return

    tbl = Table(title=f"Memories matching: {query!r}", header_style="bold cyan")
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Confidence", justify="right", style="green")
    tbl.add_column("Tags")
    tbl.add_column("Content")

    for m in resp.data:
        tbl.add_row(
            m.id[:8] + "…",
            f"{m.confidence:.4f}",
            ", ".join(m.tags),
            m.content[:80],
        )

    console.print(tbl)


# ===========================================================================
# contention  (sub-group)
# ===========================================================================


@cli.group("contention")
def contention_group() -> None:
    """Contention management commands."""


@contention_group.command("list")
@click.option(
    "--status",
    type=click.Choice(["detected", "resolved", "dismissed"]),
    default=None,
    help="Filter by contention status.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def contention_list(ctx: click.Context, status: str | None, as_json: bool) -> None:
    """List active contentions."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        resp = client.list_contentions(status=status)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json([_dump(c) for c in resp.data])
        return

    if not resp.data:
        console.print("[green]No contentions found.[/green]")
        return

    tbl = Table(title="Contentions", header_style="bold red")
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("Status")
    tbl.add_column("Severity")
    tbl.add_column("Article Node", style="dim")
    tbl.add_column("Description")

    for c in resp.data:
        tbl.add_row(
            c.id[:8] + "…",
            c.status.value,
            c.severity or "",
            c.node_id[:8] + "…",
            (c.description or "")[:80],
        )

    console.print(tbl)


@contention_group.command("resolve")
@click.argument("contention_id")
@click.option(
    "--resolution",
    required=True,
    type=click.Choice(["supersede_a", "supersede_b", "accept_both", "dismiss"]),
    help="Resolution type.",
)
@click.option("--rationale", required=True, help="Explanation of the resolution decision.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def contention_resolve(
    ctx: click.Context,
    contention_id: str,
    resolution: str,
    rationale: str,
    as_json: bool,
) -> None:
    """Resolve contention CONTENTION_ID."""
    as_json = as_json or _as_json(ctx)
    try:
        client = _get_client(ctx)
        contention = client.resolve_contention(contention_id, resolution=resolution, rationale=rationale)
    except CovalenceError as exc:
        _handle_error(exc)

    if as_json:
        _print_json(_dump(contention))
        return

    console.print(f"[bold green]✓ Contention resolved[/bold green] [cyan]{contention.id}[/cyan]")
    console.print(f"  Resolution: {contention.resolution}")
    console.print(f"  Status:     {contention.status.value}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
