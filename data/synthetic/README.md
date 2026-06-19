# Synthetic data

Generated synthetic GRB fields are written here under `run/`, which is
gitignored to keep the repo light. Regenerate any time with:

```bash
python -m grb_afterglow.synthetic data/synthetic/run --alpha 1.10 --seed 42
```

or simply run the full demo, which generates a fresh dataset first:

```bash
python scripts/run_demo.py
```
