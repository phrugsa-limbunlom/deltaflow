# Installation

DeltaFlow requires **Python ≥ 3.9** and **PyTorch ≥ 2.0**. The PyPI
distribution name is `torchdeltaflow`, and the import name is `deltaflow`.

## From PyPI

```bash
pip install torchdeltaflow
```

## From source (editable)

```bash
git clone https://github.com/phrugsa-limbunlom/deltaflow
cd deltaflow
pip install -e .
```

## Optional extras

The core runtime is intentionally minimal (`torch`, `numpy`, `einops`,
`tqdm`). Domain-specific functionality lives behind optional extras:

| Extra | Enables | Pulls in |
|---|---|---|
| `images` | Image dataset streaming (`ImageFolderStream`, `RadiographDataset`) | `pillow` |
| `ot` | Exact mini-batch OT coupling (Hungarian); falls back to a greedy matcher when absent | `scipy` |
| `all` | All optional runtime extras | `pillow`, `scipy` |
| `dev` | Test/lint/type tooling + all runtime extras | `pytest`, `black`, `isort`, `mypy`, ... |
| `docs` | Build this documentation site | `mkdocs-material`, `mkdocstrings`, ... |

```bash
pip install -e ".[all]"     # full runtime
pip install -e ".[dev]"     # contributor setup
pip install -e ".[docs]"    # build the docs
```

## Verify

```python
import deltaflow
print(deltaflow.__version__)
```

Next, see the [Quickstart](quickstart.md).
