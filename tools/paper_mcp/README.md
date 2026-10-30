# deltaflow paper MCP server

An [MCP](https://modelcontextprotocol.io) server that turns a research paper into
a GitHub issue for deltaflow. You send it an arXiv paper or a local PDF, and your
MCP client's LLM fills in the [issue blueprint](src/paper_mcp/blueprint.md). The
server then opens the issue.

This package is separate from `deltaflow`; it is not installed with the library.

## How it works

```text
you: "Create an issue for arXiv 2210.02747"
 └─ client calls read_paper("2210.02747")  -> metadata + full text + blueprint
 └─ client LLM fills every blueprint section from the paper
 └─ client calls create_issue(title, body) -> checks sections, opens the issue
```

The server does no LLM calls itself, so it needs no model API key.

| Tool / prompt | Purpose |
| --- | --- |
| `read_paper(source, max_chars=60000)` | Load a paper from an arXiv ID (`2210.02747`, `arXiv:2210.02747`), an arxiv.org URL, or a local `.pdf` path. Returns `title`, `authors`, `abstract`, `published`, `url`, `text`, `truncated` and `blueprint`. |
| `get_blueprint()` | Return the blueprint Markdown. |
| `create_issue(title, body, labels=None, repo=None)` | Open the issue. It is rejected if any `##` section of the blueprint is missing from `body`. |
| prompt `paper_to_issue(source)` | Ready-made instructions that run the whole workflow. |

## Install

From the repository root:

```bash
pip install -e tools/paper_mcp
```

This adds the `deltaflow-paper-mcp` command (equivalently `python -m paper_mcp`).

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GITHUB_TOKEN` | *(required for `create_issue`)* | Token with *Issues: write* on the target repo. |
| `PAPER_MCP_REPO` | `phrugsa-limbunlom/deltaflow` | Default `owner/name` for new issues. |
| `PAPER_MCP_LABELS` | `paper` | Comma-separated default labels. |
| `GITHUB_API_URL` | `https://api.github.com` | REST API root (change for GitHub Enterprise). |

HTTPS uses the operating system's certificate store (via `truststore`), so it
works behind corporate proxies.

### VS Code (`.vscode/mcp.json`)

```json
{
  "inputs": [
    { "id": "github-token", "type": "promptString", "description": "GitHub token", "password": true }
  ],
  "servers": {
    "deltaflow-paper": {
      "command": "deltaflow-paper-mcp",
      "env": { "GITHUB_TOKEN": "${input:github-token}" }
    }
  }
}
```

### Other clients (Claude Desktop, Copilot CLI, ...)

```json
{
  "mcpServers": {
    "deltaflow-paper": {
      "command": "deltaflow-paper-mcp",
      "env": { "GITHUB_TOKEN": "<token>" }
    }
  }
}
```

If the command is not on `PATH`, use the full path to the virtual environment's
Python with `"args": ["-m", "paper_mcp"]`.

## Changing the blueprint

Edit [`src/paper_mcp/blueprint.md`](src/paper_mcp/blueprint.md). Each `##`
heading is a required section, and the HTML comments are guidance for the LLM.
Update `SECTIONS` in `tests/test_server.py` to match.

## Development

```bash
pip install -e "tools/paper_mcp[dev]"
cd tools/paper_mcp
pytest
```

The tests use no network: arXiv and GitHub are replaced with fake responses.
