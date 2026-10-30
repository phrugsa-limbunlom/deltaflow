"""The issue blueprint: the Markdown template every paper issue must follow."""

import re
from importlib.resources import files
from typing import List

_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def load_blueprint() -> str:
    """Return the blueprint Markdown shipped with the package."""
    return files(__package__).joinpath("blueprint.md").read_text(encoding="utf-8")


def required_sections(blueprint: str) -> List[str]:
    """Return the ``##`` section titles of ``blueprint`` in order."""
    return _HEADING.findall(blueprint)


def missing_sections(body: str, blueprint: str) -> List[str]:
    """Return blueprint sections that ``body`` does not contain (case-insensitive)."""
    present = {title.lower() for title in _HEADING.findall(body)}
    return [title for title in required_sections(blueprint) if title.lower() not in present]
