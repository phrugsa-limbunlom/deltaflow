import asyncio
import json
import urllib.error

import pytest
from conftest import FakeResponse
from mcp.server.mcpserver.exceptions import ToolError

from paper_mcp import github, server
from paper_mcp.blueprint import load_blueprint, missing_sections, required_sections
from paper_mcp.papers import Paper

SECTIONS = [
    "Summary",
    "Paper",
    "Key idea",
    "Proposed API in deltaflow",
    "Math / algorithm",
    "Implementation tasks",
    "Tests",
    "Docs",
]


def filled_body() -> str:
    return "\n\n".join(f"## {section}\n\nContent." for section in SECTIONS)


def test_blueprint_sections():
    assert required_sections(load_blueprint()) == SECTIONS


def test_missing_sections_is_case_insensitive():
    body = filled_body().replace("## Key idea", "## KEY IDEA").replace("## Docs\n", "")
    assert missing_sections(body, load_blueprint()) == ["Docs"]


def test_tools_are_registered():
    tools = asyncio.run(server.server.list_tools())
    assert {tool.name for tool in tools} == {"get_blueprint", "read_paper", "create_issue"}


def test_read_paper_truncates_and_includes_blueprint(monkeypatch):
    paper = Paper(title="T", url="https://arxiv.org/abs/1", text="x" * 10)
    monkeypatch.setattr(server, "load_paper", lambda source: paper)

    result = server.read_paper("1234.5678", max_chars=4)

    assert result["text"] == "xxxx"
    assert result["truncated"] is True
    assert result["blueprint"] == load_blueprint()


def test_read_paper_reports_bad_source(tmp_path):
    with pytest.raises(ToolError, match="Not a PDF"):
        server.read_paper(str(tmp_path / "missing.pdf"))


def test_create_issue_rejects_incomplete_body(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    with pytest.raises(ToolError, match="Math / algorithm"):
        server.create_issue("Implement X", filled_body().replace("## Math / algorithm", ""))


def test_create_issue_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ToolError, match="GITHUB_TOKEN"):
        server.create_issue("Implement X", filled_body())


def test_create_issue_uses_defaults(monkeypatch):
    calls = {}

    def fake_create_issue(**kwargs):
        calls.update(kwargs)
        return {"number": 7, "html_url": "https://github.com/o/r/issues/7"}

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("PAPER_MCP_LABELS", "paper, enhancement")
    monkeypatch.delenv("PAPER_MCP_REPO", raising=False)
    monkeypatch.setattr(github, "create_issue", fake_create_issue)

    result = server.create_issue("Implement X", filled_body())

    assert result == {"number": 7, "url": "https://github.com/o/r/issues/7"}
    assert calls["repo"] == server.DEFAULT_REPO
    assert calls["labels"] == ["paper", "enhancement"]
    assert calls["token"] == "token"


def test_github_create_issue_request():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse(b'{"number": 1, "html_url": "u"}')

    issue = github.create_issue(
        "o/r", "Title", "Body", token="t", labels=["paper"], urlopen=fake_urlopen
    )

    request = captured["request"]
    assert issue["number"] == 1
    assert request.full_url == "https://api.github.com/repos/o/r/issues"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer t"
    assert json.loads(request.data) == {"title": "Title", "body": "Body", "labels": ["paper"]}


def test_github_create_issue_error():
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, FakeResponse(b'{"message": "Not Found"}')
        )

    with pytest.raises(github.GitHubError, match="404"):
        github.create_issue("o/r", "T", "B", token="t", urlopen=fake_urlopen)
