"""Minimal GitHub REST client for creating issues."""

import json
import urllib.error
import urllib.request
from typing import Callable, List, Optional

GITHUB_API_URL = "https://api.github.com"


class GitHubError(RuntimeError):
    """Raised when the GitHub API rejects a request."""


def create_issue(
    repo: str,
    title: str,
    body: str,
    token: str,
    labels: Optional[List[str]] = None,
    api_url: str = GITHUB_API_URL,
    urlopen: Callable = urllib.request.urlopen,
) -> dict:
    """Create an issue in ``repo`` (``owner/name``) and return the API response."""
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    request = urllib.request.Request(
        f"{api_url}/repos/{repo}/issues",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GitHub API returned {err.code} for {repo}: {detail}") from err
