"""MCP server: read a research paper and file a deltaflow GitHub issue from the blueprint.

Workflow for the MCP client (its LLM does the writing):

1. ``read_paper(source)`` returns the paper's metadata, text and the blueprint.
2. The client fills every blueprint section from the paper.
3. ``create_issue(title, body)`` checks that all sections are present and opens the issue.

Configuration (environment variables):

- ``GITHUB_TOKEN``: token allowed to create issues in the target repo (required).
- ``PAPER_MCP_REPO``: default ``owner/name`` repo, ``phrugsa-limbunlom/deltaflow``.
- ``PAPER_MCP_LABELS``: comma-separated default labels, ``paper``.
- ``GITHUB_API_URL``: REST API root, ``https://api.github.com``.
"""

import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator, List, Optional

import truststore
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pypdf.errors import PdfReadError

from . import github
from .blueprint import load_blueprint, missing_sections
from .papers import load_paper

DEFAULT_REPO = "phrugsa-limbunlom/deltaflow"
DEFAULT_LABELS = "paper"
DEFAULT_MAX_CHARS = 60_000

# Failures caused by input, network or GitHub; their message is safe to show the client.
_EXPECTED_ERRORS = (ValueError, OSError, PdfReadError, github.GitHubError)

server = MCPServer(
    name="deltaflow-paper",
    instructions=(
        "Turn a research paper into a deltaflow GitHub issue. Call read_paper, fill every "
        "section of the returned blueprint using the paper, then call create_issue."
    ),
)


@contextmanager
def _report_errors() -> Iterator[None]:
    """Turn expected failures into ``ToolError`` so the client sees the reason."""
    try:
        yield
    except _EXPECTED_ERRORS as exc:
        raise ToolError(str(exc)) from exc


@server.tool()
def get_blueprint() -> str:
    """Return the Markdown blueprint that every paper issue must follow."""
    return load_blueprint()


@server.tool()
def read_paper(source: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Read a paper and return its metadata, text and the issue blueprint.

    Args:
        source: arXiv ID (``2210.02747``), arXiv URL, or path to a local PDF.
        max_chars: maximum number of characters of paper text to return.
    """
    with _report_errors():
        paper = load_paper(source)
    result = asdict(paper)
    result["truncated"] = len(paper.text) > max_chars
    result["text"] = paper.text[:max_chars]
    result["blueprint"] = load_blueprint()
    return result


@server.tool()
def create_issue(
    title: str,
    body: str,
    labels: Optional[List[str]] = None,
    repo: Optional[str] = None,
) -> dict:
    """Create a GitHub issue whose body follows the blueprint.

    Args:
        title: issue title, e.g. ``"Implement <method> (<paper short name>)"``.
        body: Markdown body containing every ``##`` section of the blueprint.
        labels: issue labels; defaults to ``PAPER_MCP_LABELS``.
        repo: target ``owner/name``; defaults to ``PAPER_MCP_REPO``.
    """
    missing = missing_sections(body, load_blueprint())
    if missing:
        raise ToolError(f"Issue body is missing blueprint sections: {', '.join(missing)}")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ToolError("GITHUB_TOKEN is not set")

    if labels is None:
        labels = [
            label.strip()
            for label in os.environ.get("PAPER_MCP_LABELS", DEFAULT_LABELS).split(",")
            if label.strip()
        ]
    with _report_errors():
        issue = github.create_issue(
            repo=repo or os.environ.get("PAPER_MCP_REPO", DEFAULT_REPO),
            title=title,
            body=body,
            token=token,
            labels=labels,
            api_url=os.environ.get("GITHUB_API_URL", github.GITHUB_API_URL),
        )
    return {"number": issue["number"], "url": issue["html_url"]}


@server.prompt()
def paper_to_issue(source: str) -> str:
    """Guide the client through turning ``source`` into a GitHub issue."""
    return (
        f"Call read_paper with source={source!r}. Using only the returned paper, fill in "
        "every section of the returned blueprint: keep each '##' heading, replace the "
        "HTML comments with content, and write a concrete implementation-task checklist "
        "that fits deltaflow's layout. Show me the draft, then call create_issue with a "
        "title of the form 'Implement <method> (<paper short name>)'."
    )


def main() -> None:
    """Run the server over stdio."""
    # Verify HTTPS with the OS certificate store (needed behind corporate proxies).
    truststore.inject_into_ssl()
    server.run()


if __name__ == "__main__":
    main()
