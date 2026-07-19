# Contributing

Contributions are welcome. This project is small and research-oriented, so
please open an issue to discuss non-trivial changes before sending a PR.

## Setup

```bash
git clone https://github.com/yourname/deltaflow
cd deltaflow
pip install -e ".[dev]"
```

## Before opening a PR

```bash
black deltaflow tests examples
isort deltaflow tests examples
pytest
```

## Adding an example

Examples live under `examples/<tier>/<NN-name>/main.py`, where `<tier>` is
one of `00-foundations`, `10-sampling`, `20-training`, `90-showcase`. Each
example should run standalone in well under a minute on CPU.
