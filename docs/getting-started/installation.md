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
`tqdm`, `scipy`). `scipy` ships by default so mini-batch OT coupling
(`OTInterpolant`, `OTCoupling`) uses the exact Hungarian assignment out of
the box, a greedy nearest-neighbour matcher remains only as a defensive
fallback if `scipy` is somehow missing from the environment. Everything else
domain-specific lives behind optional extras:

| Extra | Enables | Pulls in |
|---|---|---|
| `images` | Image dataset streaming (`ImageFolderStream`, `RadiographDataset`) | `pillow` |
| `ot` | Kept for backwards compatibility, `scipy` is now a core dependency | `scipy` |
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
