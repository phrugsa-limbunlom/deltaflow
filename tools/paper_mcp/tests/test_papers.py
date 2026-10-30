import pytest
from conftest import FakeResponse

from paper_mcp import papers

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Flow Matching for
      Generative Modeling</title>
    <summary>  We introduce flow matching.  </summary>
    <published>2022-10-06T15:09:58Z</published>
    <author><name>Yaron Lipman</name></author>
    <author><name>Ricky T. Q. Chen</name></author>
  </entry>
</feed>
"""


@pytest.mark.parametrize(
    "source, expected",
    [
        ("2210.02747", "2210.02747"),
        ("2210.02747v2", "2210.02747v2"),
        ("arXiv:2210.02747", "2210.02747"),
        ("https://arxiv.org/abs/2210.02747", "2210.02747"),
        ("https://arxiv.org/pdf/2210.02747v1.pdf", "2210.02747v1"),
        ("https://www.arxiv.org/html/2210.02747", "2210.02747"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("https://example.com/abs/2210.02747", None),
        ("paper.pdf", None),
    ],
)
def test_parse_arxiv_id(source, expected):
    assert papers.parse_arxiv_id(source) == expected


def test_fetch_arxiv_reads_metadata_and_pdf(pdf_bytes):
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return FakeResponse(ATOM.encode() if "api/query" in url else pdf_bytes)

    paper = papers.load_paper("arXiv:2210.02747", urlopen=fake_urlopen)

    assert requested == [
        "https://export.arxiv.org/api/query?id_list=2210.02747",
        "https://arxiv.org/pdf/2210.02747",
    ]
    assert paper.title == "Flow Matching for Generative Modeling"
    assert paper.authors == ["Yaron Lipman", "Ricky T. Q. Chen"]
    assert paper.abstract == "We introduce flow matching."
    assert paper.published == "2022-10-06"
    assert paper.url == "https://arxiv.org/abs/2210.02747"
    assert "Flow matching with deltas" in paper.text


def test_fetch_arxiv_unknown_id():
    empty_feed = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with pytest.raises(ValueError, match="not found"):
        papers.fetch_arxiv("2210.99999", urlopen=lambda url, timeout: FakeResponse(empty_feed))


def test_load_local_pdf(tmp_path, pdf_bytes):
    path = tmp_path / "paper.pdf"
    path.write_bytes(pdf_bytes)

    paper = papers.load_paper(str(path))

    assert paper.title == "Delta Flow Paper"
    assert paper.url.startswith("file:")
    assert "Flow matching with deltas" in paper.text


def test_load_local_pdf_rejects_non_pdf(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a paper")
    with pytest.raises(ValueError, match="Not a PDF"):
        papers.load_paper(str(path))
