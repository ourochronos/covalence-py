"""Tests for the Covalence CLI (covalence.cli).

All tests use click.testing.CliRunner and mock CovalenceClient so no live
server is required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from covalence.cli import cli
from covalence.models import (
    AdminEdgeStats,
    AdminEmbeddingStats,
    AdminNodeStats,
    AdminQueueStats,
    AdminStats,
    Article,
    CompileJob,
    Contention,
    ContentionListResponse,
    ContentionStatus,
    MaintenanceResult,
    Memory,
    MemoryListResponse,
    Meta,
    NodeStatus,
    SearchMeta,
    SearchResponse,
    SearchResult,
    Source,
    SourceListResponse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _mock_client() -> MagicMock:
    return MagicMock()


def _sample_article(**kwargs: Any) -> Article:
    defaults = dict(
        id="article-uuid-1234",
        title="Test Article",
        content="This is article content.",
        status=NodeStatus.active,
        version=1,
        usage_score=0.42,
        pinned=False,
        domain_path=["science", "physics"],
        created_at=datetime(2025, 1, 1),
        modified_at=datetime(2025, 1, 2),
    )
    defaults.update(kwargs)
    return Article(**defaults)


def _sample_source(**kwargs: Any) -> Source:
    defaults = dict(
        id="source-uuid-5678",
        title="Test Source",
        content="Source content here.",
        source_type="document",
        status=NodeStatus.active,
        confidence=0.8,
        reliability=0.75,
        created_at=datetime(2025, 1, 1),
    )
    defaults.update(kwargs)
    return Source(**defaults)


def _sample_memory(**kwargs: Any) -> Memory:
    defaults = dict(
        id="memory-uuid-abcd",
        content="The sky is blue.",
        tags=["observation"],
        importance=0.7,
        confidence=0.65,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def _sample_contention(**kwargs: Any) -> Contention:
    defaults = dict(
        id="contention-uuid-ef01",
        node_id="article-uuid-1234",
        source_node_id="source-uuid-5678",
        description="Conflicting claim about X.",
        status=ContentionStatus.detected,
        severity="medium",
    )
    defaults.update(kwargs)
    return Contention(**defaults)


def _sample_admin_stats() -> AdminStats:
    return AdminStats(
        nodes=AdminNodeStats(
            total=100,
            sources=60,
            articles=35,
            sessions=5,
            active=90,
            archived=10,
            pinned=3,
        ),
        edges=AdminEdgeStats(sql_count=200, age_count=200, in_sync=True),
        queue=AdminQueueStats(pending=2, processing=1, failed=0, completed_24h=50),
        embeddings=AdminEmbeddingStats(total=95, nodes_without=5),
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_rich(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            client = MockClient.return_value
            client.get_admin_stats.return_value = _sample_admin_stats()

            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "100" in result.output  # total nodes
        assert "Nodes" in result.output
        assert "Queue" in result.output

    def test_status_json(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_admin_stats.return_value = _sample_admin_stats()

            result = runner.invoke(cli, ["status", "--json"])

        assert result.exit_code == 0, result.output
        # rich JSON output contains the data
        assert "100" in result.output  # total nodes from json dump

    def test_status_global_json_flag(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_admin_stats.return_value = _sample_admin_stats()

            result = runner.invoke(cli, ["--json", "status"])

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_resp(self) -> SearchResponse:
        return SearchResponse(
            data=[
                SearchResult(
                    node_id="abc123",
                    score=0.95,
                    node_type="article",
                    title="Relevant Article",
                    content_preview="Some preview text.",
                )
            ],
            meta=SearchMeta(total_results=1, elapsed_ms=12, dimensions_used=["vector"]),
        )

    def test_search_rich(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.search.return_value = self._make_resp()

            result = runner.invoke(cli, ["search", "knowledge graphs"])

        assert result.exit_code == 0, result.output
        assert "Relevant Article" in result.output
        client.search.assert_called_once_with("knowledge graphs", intent=None, limit=10)

    def test_search_with_intent(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.search.return_value = self._make_resp()

            result = runner.invoke(cli, ["search", "events", "--intent", "temporal"])

        assert result.exit_code == 0, result.output
        client.search.assert_called_once_with("events", intent="temporal", limit=10)

    def test_search_with_domain(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.search.return_value = self._make_resp()

            result = runner.invoke(cli, ["search", "photons", "--domain", "physics"])

        assert result.exit_code == 0, result.output
        # domain prefix is prepended to the query
        client.search.assert_called_once_with("[physics] photons", intent=None, limit=10)

    def test_search_json(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.search.return_value = self._make_resp()

            result = runner.invoke(cli, ["search", "test", "--json"])

        assert result.exit_code == 0, result.output
        assert "node_id" in result.output

    def test_search_no_results(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.search.return_value = SearchResponse(data=[], meta=None)

            result = runner.invoke(cli, ["search", "nothing"])

        assert result.exit_code == 0
        assert "No results" in result.output


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_stdin(self, runner: CliRunner) -> None:
        src = _sample_source()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.ingest_source.return_value = src

            result = runner.invoke(
                cli, ["ingest", "--type", "document", "--title", "My doc"], input="Hello world"
            )

        assert result.exit_code == 0, result.output
        assert "Ingested" in result.output
        client.ingest_source.assert_called_once_with("Hello world", source_type="document", title="My doc")

    def test_ingest_file(self, runner: CliRunner, tmp_path: Any) -> None:
        src = _sample_source()
        content = "File based content."
        f = tmp_path / "doc.txt"
        f.write_text(content)

        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.ingest_source.return_value = src

            result = runner.invoke(cli, ["ingest", str(f)])

        assert result.exit_code == 0, result.output
        assert "Ingested" in result.output
        client.ingest_source.assert_called_once_with(content, source_type=None, title=None)

    def test_ingest_empty_stdin(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient"):
            result = runner.invoke(cli, ["ingest"], input="   \n")

        assert result.exit_code != 0
        assert "Empty content" in result.output

    def test_ingest_json(self, runner: CliRunner) -> None:
        src = _sample_source()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.ingest_source.return_value = src

            result = runner.invoke(cli, ["ingest", "--json"], input="some text")

        assert result.exit_code == 0, result.output
        assert "source-uuid-5678" in result.output


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


class TestCompile:
    def test_compile(self, runner: CliRunner) -> None:
        job = CompileJob(job_id="job-abc", status="pending")
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.compile_article.return_value = job

            result = runner.invoke(cli, ["compile", "src-1", "src-2", "--title", "My article"])

        assert result.exit_code == 0, result.output
        assert "job-abc" in result.output
        client.compile_article.assert_called_once_with(["src-1", "src-2"], title_hint="My article")

    def test_compile_json(self, runner: CliRunner) -> None:
        job = CompileJob(job_id="job-xyz", status="pending")
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.compile_article.return_value = job

            result = runner.invoke(cli, ["compile", "src-1", "--json"])

        assert result.exit_code == 0, result.output
        assert "job_id" in result.output


# ---------------------------------------------------------------------------
# maintenance
# ---------------------------------------------------------------------------


class TestMaintenance:
    def test_maintenance(self, runner: CliRunner) -> None:
        maint = MaintenanceResult(actions_taken=["recomputed scores", "processed queue"])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.run_maintenance.return_value = maint

            result = runner.invoke(cli, ["maintenance", "--recompute", "--process-queue"])

        assert result.exit_code == 0, result.output
        assert "recomputed scores" in result.output
        client.run_maintenance.assert_called_once_with(
            recompute_scores=True,
            process_queue=True,
            evict_if_over_capacity=None,
            evict_count=None,
        )

    def test_maintenance_no_actions(self, runner: CliRunner) -> None:
        maint = MaintenanceResult(actions_taken=[])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.run_maintenance.return_value = maint

            result = runner.invoke(cli, ["maintenance"])

        assert result.exit_code == 0, result.output
        assert "no actions" in result.output.lower()

    def test_maintenance_json(self, runner: CliRunner) -> None:
        maint = MaintenanceResult(actions_taken=["evicted 3 articles"])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.run_maintenance.return_value = maint

            result = runner.invoke(cli, ["maintenance", "--evict", "--json"])

        assert result.exit_code == 0, result.output
        assert "actions_taken" in result.output


# ---------------------------------------------------------------------------
# article get / list
# ---------------------------------------------------------------------------


class TestArticle:
    def test_article_get(self, runner: CliRunner) -> None:
        art = _sample_article()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_article.return_value = art

            result = runner.invoke(cli, ["article", "get", "article-uuid-1234"])

        assert result.exit_code == 0, result.output
        assert "article-uuid-1234" in result.output
        assert "Test Article" in result.output

    def test_article_get_json(self, runner: CliRunner) -> None:
        art = _sample_article()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_article.return_value = art

            result = runner.invoke(cli, ["article", "get", "article-uuid-1234", "--json"])

        assert result.exit_code == 0, result.output
        assert "article-uuid-1234" in result.output
        assert "content" in result.output

    def test_article_list(self, runner: CliRunner) -> None:
        art = _sample_article()
        resp = MagicMock()
        resp.data = [art]
        resp.meta = Meta(count=1)
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_articles.return_value = resp

            result = runner.invoke(cli, ["article", "list"])

        assert result.exit_code == 0, result.output
        assert "Test Article" in result.output

    def test_article_list_empty(self, runner: CliRunner) -> None:
        resp = MagicMock()
        resp.data = []
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_articles.return_value = resp

            result = runner.invoke(cli, ["article", "list"])

        assert result.exit_code == 0, result.output
        assert "No articles" in result.output

    def test_article_list_json(self, runner: CliRunner) -> None:
        art = _sample_article()
        resp = MagicMock()
        resp.data = [art]
        resp.meta = Meta(count=1)
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_articles.return_value = resp

            result = runner.invoke(cli, ["article", "list", "--json"])

        assert result.exit_code == 0, result.output
        assert "article-uuid-1234" in result.output

    def test_article_list_with_options(self, runner: CliRunner) -> None:
        resp = MagicMock()
        resp.data = []
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_articles.return_value = resp

            result = runner.invoke(cli, ["article", "list", "--limit", "5", "--status", "archived"])

        assert result.exit_code == 0, result.output
        client.list_articles.assert_called_once_with(limit=5, cursor=None, status="archived")


# ---------------------------------------------------------------------------
# source get / list
# ---------------------------------------------------------------------------


class TestSource:
    def test_source_get(self, runner: CliRunner) -> None:
        src = _sample_source()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_source.return_value = src

            result = runner.invoke(cli, ["source", "get", "source-uuid-5678"])

        assert result.exit_code == 0, result.output
        assert "source-uuid-5678" in result.output
        assert "Test Source" in result.output

    def test_source_get_json(self, runner: CliRunner) -> None:
        src = _sample_source()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_source.return_value = src

            result = runner.invoke(cli, ["source", "get", "source-uuid-5678", "--json"])

        assert result.exit_code == 0, result.output
        assert "source-uuid-5678" in result.output

    def test_source_list(self, runner: CliRunner) -> None:
        src = _sample_source()
        resp = SourceListResponse(data=[src], meta=Meta(count=1))
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_sources.return_value = resp

            result = runner.invoke(cli, ["source", "list"])

        assert result.exit_code == 0, result.output
        assert "Test Source" in result.output

    def test_source_list_empty(self, runner: CliRunner) -> None:
        resp = SourceListResponse(data=[], meta=Meta(count=0))
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_sources.return_value = resp

            result = runner.invoke(cli, ["source", "list"])

        assert result.exit_code == 0, result.output
        assert "No sources" in result.output

    def test_source_list_with_type_filter(self, runner: CliRunner) -> None:
        resp = SourceListResponse(data=[], meta=Meta(count=0))
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_sources.return_value = resp

            result = runner.invoke(cli, ["source", "list", "--type", "document"])

        assert result.exit_code == 0, result.output
        client.list_sources.assert_called_once_with(limit=20, cursor=None, source_type="document")


# ---------------------------------------------------------------------------
# memory store / recall
# ---------------------------------------------------------------------------


class TestMemory:
    def test_memory_store(self, runner: CliRunner) -> None:
        mem = _sample_memory()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.store_memory.return_value = mem

            result = runner.invoke(
                cli,
                ["memory", "store", "The sky is blue.", "--tags", "science,nature", "--importance", "0.8"],
            )

        assert result.exit_code == 0, result.output
        assert "memory-uuid-abcd" in result.output
        client.store_memory.assert_called_once_with(
            "The sky is blue.",
            tags=["science", "nature"],
            importance=0.8,
            context=None,
        )

    def test_memory_store_json(self, runner: CliRunner) -> None:
        mem = _sample_memory()
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.store_memory.return_value = mem

            result = runner.invoke(cli, ["memory", "store", "Some content", "--json"])

        assert result.exit_code == 0, result.output
        assert "memory-uuid-abcd" in result.output

    def test_memory_recall(self, runner: CliRunner) -> None:
        mem = _sample_memory()
        resp = MemoryListResponse(data=[mem])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.recall_memories.return_value = resp

            result = runner.invoke(cli, ["memory", "recall", "sky color"])

        assert result.exit_code == 0, result.output
        assert "The sky is blue" in result.output
        client.recall_memories.assert_called_once_with("sky color", limit=5, tags=None, min_confidence=None)

    def test_memory_recall_with_tags(self, runner: CliRunner) -> None:
        resp = MemoryListResponse(data=[])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.recall_memories.return_value = resp

            result = runner.invoke(
                cli, ["memory", "recall", "test", "--tags", "science,nature", "--limit", "3"]
            )

        assert result.exit_code == 0, result.output
        client.recall_memories.assert_called_once_with(
            "test", limit=3, tags=["science", "nature"], min_confidence=None
        )

    def test_memory_recall_no_results(self, runner: CliRunner) -> None:
        resp = MemoryListResponse(data=[])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.recall_memories.return_value = resp

            result = runner.invoke(cli, ["memory", "recall", "nothing"])

        assert result.exit_code == 0, result.output
        assert "No memories" in result.output

    def test_memory_recall_json(self, runner: CliRunner) -> None:
        mem = _sample_memory()
        resp = MemoryListResponse(data=[mem])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.recall_memories.return_value = resp

            result = runner.invoke(cli, ["memory", "recall", "test", "--json"])

        assert result.exit_code == 0, result.output
        assert "memory-uuid-abcd" in result.output


# ---------------------------------------------------------------------------
# contention list / resolve
# ---------------------------------------------------------------------------


class TestContention:
    def test_contention_list(self, runner: CliRunner) -> None:
        cont = _sample_contention()
        resp = ContentionListResponse(data=[cont])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_contentions.return_value = resp

            result = runner.invoke(cli, ["contention", "list"])

        assert result.exit_code == 0, result.output
        assert "Conflicting claim about X" in result.output

    def test_contention_list_empty(self, runner: CliRunner) -> None:
        resp = ContentionListResponse(data=[])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_contentions.return_value = resp

            result = runner.invoke(cli, ["contention", "list"])

        assert result.exit_code == 0, result.output
        assert "No contentions" in result.output

    def test_contention_list_with_status(self, runner: CliRunner) -> None:
        resp = ContentionListResponse(data=[])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_contentions.return_value = resp

            result = runner.invoke(cli, ["contention", "list", "--status", "resolved"])

        assert result.exit_code == 0, result.output
        client.list_contentions.assert_called_once_with(status="resolved")

    def test_contention_list_json(self, runner: CliRunner) -> None:
        cont = _sample_contention()
        resp = ContentionListResponse(data=[cont])
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.list_contentions.return_value = resp

            result = runner.invoke(cli, ["contention", "list", "--json"])

        assert result.exit_code == 0, result.output
        assert "contention-uuid-ef01" in result.output

    def test_contention_resolve(self, runner: CliRunner) -> None:
        cont = _sample_contention(status=ContentionStatus.resolved, resolution="dismiss")
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.resolve_contention.return_value = cont

            result = runner.invoke(
                cli,
                [
                    "contention",
                    "resolve",
                    "contention-uuid-ef01",
                    "--resolution",
                    "dismiss",
                    "--rationale",
                    "Not relevant",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "resolved" in result.output
        client.resolve_contention.assert_called_once_with(
            "contention-uuid-ef01", resolution="dismiss", rationale="Not relevant"
        )

    def test_contention_resolve_json(self, runner: CliRunner) -> None:
        cont = _sample_contention(status=ContentionStatus.resolved, resolution="accept_both")
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.resolve_contention.return_value = cont

            result = runner.invoke(
                cli,
                [
                    "contention",
                    "resolve",
                    "contention-uuid-ef01",
                    "--resolution",
                    "accept_both",
                    "--rationale",
                    "Both valid",
                    "--json",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "contention-uuid-ef01" in result.output


# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------


class TestGlobalFlags:
    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Covalence CLI" in result.output

    def test_custom_url(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_admin_stats.return_value = _sample_admin_stats()

            result = runner.invoke(cli, ["--url", "http://myserver:9000", "status"])

        assert result.exit_code == 0, result.output
        MockClient.assert_called_once_with(base_url="http://myserver:9000")

    def test_envvar_url(self, runner: CliRunner) -> None:
        with patch("covalence.cli.CovalenceClient") as MockClient:
            client = MockClient.return_value
            client.get_admin_stats.return_value = _sample_admin_stats()

            result = runner.invoke(cli, ["status"], env={"COVALENCE_URL": "http://env-server:8888"})

        assert result.exit_code == 0, result.output
        MockClient.assert_called_once_with(base_url="http://env-server:8888")

    def test_subcommand_help(self, runner: CliRunner) -> None:
        for sub in ["search", "ingest", "compile", "maintenance"]:
            result = runner.invoke(cli, [sub, "--help"])
            assert result.exit_code == 0, f"{sub} --help failed: {result.output}"

    def test_subgroup_help(self, runner: CliRunner) -> None:
        for grp in ["article", "source", "memory", "contention"]:
            result = runner.invoke(cli, [grp, "--help"])
            assert result.exit_code == 0, f"{grp} --help failed: {result.output}"
