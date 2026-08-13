# Contributing

Contributions are welcome. This project is small and research-oriented, so
please open an issue to discuss non-trivial changes before sending a PR.

## Setup

```bash
git clone https://github.com/phrugsa-limbunlom/deltaflow
cd deltaflow
pip install -e ".[dev]"
```

## Before opening a PR

```bash
black deltaflow tests examples
isort deltaflow tests examples
pytest
```

Run a single test while iterating:

```bash
pytest tests/test_solvers.py::test_name
```

CI runs `pytest --cov=deltaflow` on Python 3.10 / 3.11 / 3.12.

## Adding an example

Examples live under `examples/<tier>/<NN-name>/main.py`, where `<tier>` is one
of `00-foundations`, `10-sampling`, `20-training`, `30-inverse`,
`90-showcase`. Each example should run standalone in well under a minute on
CPU.

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site into ./site
```
