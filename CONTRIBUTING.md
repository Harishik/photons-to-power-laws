# Contributing

Thanks for your interest in improving **Photons to Power Laws**. This is an
open, educational astronomy project, so clear explanations are valued as highly
as working code.

## Getting set up

```bash
git clone https://github.com/Harishik/photons-to-power-laws.git
cd photons-to-power-laws
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"     # editable install + pytest
pytest -q                   # everything should pass before you start
```

## Ground rules

- **Keep modules focused.** Each pipeline stage lives in its own module and
  should be usable on its own.
- **Explain the science.** Docstrings should say not just *what* a function does
  but *why* — what physical quantity it produces and how. New contributors learn
  the subject by reading this code.
- **Always carry uncertainties.** A measurement without an error bar can't be
  fit meaningfully. Propagate errors through any new stage.
- **Add a test.** New behaviour needs a test in `tests/`. Where possible, test
  against a known answer. The synthetic generator makes this easy.
- **No heavyweight external binaries** in the core path. Prefer pip-installable
  Python so the project stays runnable everywhere. Optional adapters, such as
  `astrometry.net` or `astroquery`, can live behind extras.

## Workflow

1. Open an issue describing the change, especially for new pipeline stages.
2. Branch from `main`: `git checkout -b feature/short-description`.
3. Make the change, add tests, run `pytest -q`.
4. Update `CHANGELOG.md` under an `## [Unreleased]` heading.
5. Open a pull request with a clear description and, for visual changes, a
   before/after figure.

## Style

- Follow PEP 8 and keep lines reasonable.
- Type hints on public functions.
- Prefer NumPy-style docstrings, as used throughout `src/grb_afterglow/`.

## Good first issues

Look at the unchecked items in the README roadmap. The catalog adapters and the
real-data WCS matching path are well-scoped entry points.
