"""Load a research paper from an arXiv ID/URL or a local PDF."""

import io
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from pypdf import PdfReader

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
MAX_PDF_BYTES = 50 * 1024 * 1024

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
# New-style IDs (2301.12345v2) and old-style IDs (hep-th/9901001v1).
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")
_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


@dataclass
class Paper:
    """A paper's metadata and extracted full text."""

    title: str
    url: str
    text: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""


def parse_arxiv_id(source: str) -> Optional[str]:
    """Return the arXiv ID in ``source`` (bare ID, ``arXiv:`` ID, or arxiv.org URL)."""
    source = source.strip()
    if source.lower().startswith("arxiv:"):
        source = source[len("arxiv:") :]
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        if parsed.hostname not in _ARXIV_HOSTS:
            return None
        source = re.sub(r"^/(abs|pdf|html)/", "", parsed.path).removesuffix(".pdf")
    match = _ARXIV_ID.fullmatch(source)
    return match.group(1) if match else None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from all pages of a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _read_limited(response, limit: int) -> bytes:
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Download exceeds {limit} bytes")
    return data


def _collapse(text: str) -> str:
    return " ".join(text.split())


def fetch_arxiv(arxiv_id: str, urlopen: Callable = urllib.request.urlopen) -> Paper:
    """Download metadata and full text for ``arxiv_id``."""
    query = urllib.parse.urlencode({"id_list": arxiv_id})
    with urlopen(f"{ARXIV_API_URL}?{query}", timeout=30) as response:
        feed = ET.fromstring(_read_limited(response, 5 * 1024 * 1024))
    entry = feed.find("atom:entry", _ATOM)
    if entry is None or entry.find("atom:title", _ATOM) is None:
        raise ValueError(f"arXiv paper {arxiv_id!r} not found")

    with urlopen(ARXIV_PDF_URL.format(arxiv_id=arxiv_id), timeout=60) as response:
        text = extract_pdf_text(_read_limited(response, MAX_PDF_BYTES))

    return Paper(
        title=_collapse(entry.findtext("atom:title", "", _ATOM)),
        url=ARXIV_ABS_URL.format(arxiv_id=arxiv_id),
        text=text,
        authors=[
            _collapse(author.findtext("atom:name", "", _ATOM))
            for author in entry.findall("atom:author", _ATOM)
        ],
        abstract=_collapse(entry.findtext("atom:summary", "", _ATOM)),
        published=entry.findtext("atom:published", "", _ATOM)[:10],
    )


def load_local_pdf(path: str) -> Paper:
    """Read a local PDF file."""
    pdf_path = Path(path).expanduser().resolve()
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        raise ValueError(f"Not a PDF file: {pdf_path}")
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes: {pdf_path}")
    pdf_bytes = pdf_path.read_bytes()
    metadata = PdfReader(io.BytesIO(pdf_bytes)).metadata
    title = (metadata.title if metadata else None) or pdf_path.stem
    return Paper(title=title, url=pdf_path.as_uri(), text=extract_pdf_text(pdf_bytes))


def load_paper(source: str, urlopen: Callable = urllib.request.urlopen) -> Paper:
    """Load a paper from an arXiv ID/URL or a local PDF path."""
    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        return fetch_arxiv(arxiv_id, urlopen=urlopen)
    return load_local_pdf(source)
