# torch-ref

Generates the PyTorch reference fixtures the engine tests compare against
(CONTRACTS.md section 11).

```bash
pnpm torch-ref   # = uv run tools/torch-ref/gen.py
```

`uv` reads the pinned `torch` and `numpy` versions from the script's inline
metadata and downloads them on first use. Output goes to
`packages/engine/test/fixtures/<name>.json`. Everything is float64 and seeded,
so a second run is byte-identical. Never edit a fixture by hand: change
`gen.py` and regenerate. Tests never run Python.

Each case is built as a real `nn.Linear`/`nn.*Loss` model, so the numbers come
from torch, not from a re-implementation. See the docstring in `gen.py` for the
conventions (weight layout `[in, out]`, init, loss reductions, penalty, dropout
mask, feature column order).

The same script also writes the decision-tree fixtures (scikit-learn) to
`packages/core/test/fixtures/tree_sklearn.json` (CONTRACTS.md section 11).
