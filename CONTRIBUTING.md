# Contributing

Thanks for your interest in improving **Photons to Power Laws**. This is an
open, educational astronomy project, so clear explanations are valued as highly
as working code.

## Getting set up

```bash
git clone https://github.com/Harishik/photons-to-power-laws.git
cd photons-to-power-laws
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: lint and format on every commit
pytest                      # everything should pass before you start
```

## Ground rules

- **Keep modules focused.** Each pipeline stage lives in its own module and
  is usable on its own.
- **Explain the science.** Docstrings say not just *what* a function does but
  *why*: which physical quantity it produces and how.
- **Always carry uncertainties.** A measurement without an error bar cannot be
  fitted meaningfully. Propagate errors through any new stage, and test that
  they are honest (see `tests/test_photometry.py` for the pattern).
- **Test against a known answer.** The synthetic generator injects known
  physics; new stages should be validated by recovering it.
- **No heavyweight external binaries** in the core path. Optional adapters can
  live behind extras.

## Workflow

1. Open an issue describing the change, especially for new pipeline stages.
2. Branch from `main`: `git checkout -b feature/short-description`.
3. Make the change and add tests. Run `ruff check .`, `ruff format .`, and
   `pytest`.
4. If you changed anything that affects results, run `python scripts/run_demo.py`
   and `python case_studies/grb221009a/analyze.py` and commit the updated
   outputs.
5. Add an entry under `## [Unreleased]` in `CHANGELOG.md`.
6. Open a pull request with a clear description and, for visual changes, a
   before/after figure.

## Style

- Ruff handles linting and formatting (configured in `pyproject.toml`).
- Type hints on public functions; NumPy-style docstrings.

## Good first issues

See the roadmap in the README. A colour-term option in `reduce_frame` and
Galactic extinction corrections are well-scoped next steps. New photometry or
subtraction methods can be scored against the existing ones with the harness
in `dev/bakeoff/`.
