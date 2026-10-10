<div align="center">

# Photons to Power Laws

### From raw telescope frames to the physics of gamma-ray burst afterglows

[![CI](https://github.com/Harishik/photons-to-power-laws/actions/workflows/ci.yml/badge.svg)](https://github.com/Harishik/photons-to-power-laws/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<img src="docs/figures/hero.gif" alt="A simulated afterglow fading in the image while its g, r, i light curve builds up and a jet-break model is fitted" width="840">

</div>

**Photons to Power Laws** is a Python pipeline that takes optical images of a
gamma-ray burst (GRB) afterglow and turns them into physics. It calibrates raw
frames, measures the seeing, calibrates photometry against a reference star
catalog through the WCS, subtracts a deep template image to remove the host
galaxy and blended stars, measures the burst with empirical-PSF or aperture
photometry, fits single and broken power laws jointly across bands, and tests
the fitted slopes against the closure relations of the fireball model to infer
the electron energy index, the circumburst medium, and the jet break.

It is validated two ways: on a simulated campaign where the true physics is
known, and on **public ZTF images of GRB 221009A**, the brightest gamma-ray
burst ever recorded, where it is checked against published photometry.

## Results

### Simulated campaign: injected physics recovered from raw pixels

The demo simulates 36 dithered g/r/i frames with variable seeing and
transparency, saturating stars, a structured bias, a vignetted flat, and an
r = 21 host galaxy under the afterglow, plus a deep template image per band.
The injected afterglow has `p = 2.2` in a constant-density medium with a jet
break at half a day. Each frame is template-subtracted and measured with PSF
photometry; everything below is recovered from the raw FITS files.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/lightcurve_dark.png">
  <img src="docs/figures/lightcurve_light.png" alt="Three-band light curve with a jet-break fit and normalised residuals" width="100%">
</picture>

| Parameter | Injected | Recovered |
|---|---:|---:|
| Early decay, alpha1 | 0.900 | 0.901 ± 0.001 |
| Late decay, alpha2 | 2.200 | 2.187 ± 0.038 |
| Spectral index, beta | 0.600 | 0.608 ± 0.006 |
| Jet-break time | 0.500 d | 0.494 ± 0.011 d |
| Electron index, p | 2.2 | 2.216 ± 0.012 |
| Preferred model | broken power law | broken power law (ΔBIC = 17,041) |

Every parameter is within 1.4σ of the injected value. The host galaxy is why
image subtraction matters: measured the simple way, with apertures on the
science frames, the same data give `alpha2 = 1.34 ± 0.01`, a 70σ error,
because once the afterglow fades below the host the aperture measures the host.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/host_comparison_dark.png">
  <img src="docs/figures/host_comparison_light.png" alt="r-band light curve with and without template subtraction: without it the late points flatten onto the host galaxy" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/subtraction_dark.png">
  <img src="docs/figures/subtraction_light.png" alt="Science cutout minus the PSF-matched template equals a difference image in which only the afterglow remains" width="100%">
</picture>

The closure-relation test picks the injected regime on its own: an ISM in the
`nu_m < nu < nu_c` regime before the break and a post-jet-break decay after it.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/closure_dark.png">
  <img src="docs/figures/closure_light.png" alt="Measured alpha and beta before and after the break against the closure relations" width="62%">
</picture>
</p>

### Real data: GRB 221009A in public ZTF images

The same `reduce_frame` code runs on 120″ ZTF cutouts of GRB 221009A,
calibrated against Pan-STARRS1. The deep pre-burst ZTF reference image is
subtracted from every epoch and the afterglow is measured on the difference
image with PSF photometry. It is detected at 1.6, 3.6, 5.6 and 10.6 days in
r band; later epochs give forced-photometry limits. One epoch is rejected
after inspection because a satellite trail crosses the aperture.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="case_studies/grb221009a/figures/lightcurve_dark.png">
  <img src="case_studies/grb221009a/figures/lightcurve_light.png" alt="GRB 221009A ZTF r-band light curve with Liverpool Telescope photometry" width="100%">
</picture>

| Measurement | Value |
|---|---|
| Decay index from ZTF alone (t ≥ 1 d) | 1.34 ± 0.09 |
| Same, v0.2 method (aperture minus reference flux) | 1.53 ± 0.15 |
| Decay index from ZTF + Liverpool Telescope | 1.40 ± 0.02 |
| Published r′ decay after 0.4 d (Laskar et al. 2023) | 1.42 ± 0.01 |
| ZTF − Liverpool at the first three detections | +0.00, +0.15, −0.17 mag |

Difference imaging with PSF photometry raises the signal-to-noise of each
detection by a factor of 1.4 to 1.9 over the v0.2 measurement and shrinks the
error on the ZTF-only decay index by 40 %. Both agree with the published value.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="case_studies/grb221009a/figures/subtraction_dark.png">
  <img src="case_studies/grb221009a/figures/subtraction_light.png" alt="ZTF cutout at 1.6 days, PSF-matched to the reference image, minus the reference, leaving the afterglow" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="case_studies/grb221009a/figures/filmstrip_dark.png">
  <img src="case_studies/grb221009a/figures/filmstrip_light.png" alt="ZTF r-band cutouts around GRB 221009A, with one epoch flagged for a satellite trail" width="100%">
</picture>

The full write-up, including what limits the measurement, is in
[`case_studies/grb221009a`](case_studies/grb221009a/README.md).

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture_dark.png">
  <img src="docs/assets/architecture_light.png" alt="Pipeline architecture in ten stages: data inputs, calibration, detection and seeing, photometric calibration, image subtraction, photometry, light curve, model fitting, physics, outputs" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/pipeline_stages_dark.png">
  <img src="docs/figures/pipeline_stages_light.png" alt="One frame at four stages: raw, calibrated, detected and matched, forced photometry" width="100%">
</picture>

| Stage | Module | What it does |
|---|---|---|
| Calibration | `calibration` | Sigma-clipped master bias and flats; `(raw − bias) / flat` |
| Detection | `detection` | Background and RMS maps, DAOFIND detection, seeing from Gaussian fits to isolated stars, source masks |
| Calibration to AB | `zeropoint` | One-to-one WCS cross-match; robust weighted zero point with optional colour term |
| Image subtraction | `subtraction` | Template resampled through the WCS; Bramich (2008) delta-basis kernel with errors-in-variables correction, regularisation and clipping; difference image in science counts |
| Photometry | `photometry`, `psf` | Aperture photometry with masked, sigma-clipped sky annuli and CCD-equation errors; empirical PSF (Moffat + oversampled residual) fitted to all sources at once; forced PSF fits on difference images with errors from blank positions |
| Reduction | `pipeline` | Seeing-matched apertures, per-frame zero point, astrometric correction, template subtraction, forced photometry at the burst |
| Light curve | `lightcurve` | Multi-band light curve in flux density, so faint and negative forced fluxes are kept |
| Fitting | `fitting` | Single and smoothly broken power laws, joint across bands with spectral index beta; AIC/BIC comparison |
| Physics | `closure` | Ranks the closure relations against the measured (alpha, beta) and infers `p` |
| Data | `synthetic`, `archives` | Simulated campaigns with known truth; ZTF (IRSA) and Pan-STARRS1 (MAST) access |

Photometric calibration is solved separately for every frame, and the
recovered zero points track the simulated atmosphere:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/zeropoints_dark.png">
  <img src="docs/figures/zeropoints_light.png" alt="Zero-point solution for one frame and zero point against time compared with the true transparency" width="100%">
</picture>

## Quickstart

```bash
git clone https://github.com/Harishik/photons-to-power-laws.git
cd photons-to-power-laws
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run the validation demo (simulate, reduce, fit, validate, draw figures):

```bash
p2pl demo --out my_demo
```

Fit your own light curve (CSV with `time_s` and `flux_ujy`/`flux_err_ujy`, or `mag`/`mag_err`, and an optional `band` column):

```bash
p2pl fit my_lightcurve.csv --plot fit.png
```

Reproduce the GRB 221009A case study from the committed ZTF data:

```bash
python case_studies/grb221009a/analyze.py
```

## Python API

```python
from glob import glob

from astropy.coordinates import SkyCoord
from astropy.table import Table

from grb_afterglow import (
    PipelineConfig,
    build_master_bias,
    build_master_flat,
    compare_models,
    evaluate_closure_relations,
    fit_lightcurve,
    run_pipeline,
)
from grb_afterglow.calibration import load_fits


def frames(pattern):
    return [load_fits(path)[0] for path in sorted(glob(pattern))]


bias = build_master_bias(frames("raw/bias_*.fits"))
flats = {band: build_master_flat(frames(f"raw/flat_{band}_*.fits"), master_bias=bias) for band in "gri"}
catalog = Table.read("catalog.csv")  # ra, dec, and per-band AB magnitudes (g, g_err, r, r_err, ...)
target = SkyCoord(150.10330, 22.39806, unit="deg")

result = run_pipeline(
    sorted(glob("raw/sci_*.fits")),
    target,
    catalog,
    master_bias=bias,
    master_flats=flats,
    templates={band: f"templates/template_{band}.fits" for band in "gri"},  # optional: difference imaging
    config=PipelineConfig(photometry="psf"),  # or "aperture" (the default)
)
lc = result.lightcurve

single = fit_lightcurve(lc, "single", reference_band="r")
broken = fit_lightcurve(lc, "broken", reference_band="r")
print(compare_models(single, broken).summary())

p, e = broken.params, broken.errors
best = evaluate_closure_relations(p["alpha2"], e["alpha2"], p["beta"], e["beta"])[0]
print(f"{best.relation.label}: p = {best.p:.2f} ± {best.p_err:.2f}")
```

`p2pl simulate my_data` writes a simulated campaign in exactly this layout (run the snippet inside `my_data/`).

## Repository layout

```text
src/grb_afterglow/      the package (pipeline stages, fitting, closure relations, plots, CLI)
tests/                  91 tests: unit, statistical error-bar checks, end-to-end recovery
case_studies/grb221009a fetch_data.py, analyze.py, committed public data, results, figures
examples/synthetic/     demo outputs: light curve, per-frame table, validation summary
docs/                   science background, figures, architecture diagram (make_architecture.py)
dev/bakeoff/            harness and results of the PSF-photometry and image-subtraction method comparison
scripts/run_demo.py     regenerates the demo outputs and README figures
```

## Testing and quality

```bash
pip install -e ".[dev]"
pytest --cov          # 91 tests
ruff check . && ruff format --check .
```

Beyond unit tests, the suite checks that the error bars are honest: forced
photometry on blank sky averages to zero with unit-width pulls, repeated noise
realisations match the quoted uncertainties, PSF fluxes are unbiased with
unit-width pulls, image subtraction cancels static stars and the host while
keeping the transient's flux in both seeing directions, and an end-to-end run
recovers the injected afterglow physics. CI runs on Python 3.10 to 3.13 and also runs the
demo and the case study.

## Scope and roadmap

Done in 0.3: PSF photometry and image subtraction against a template. Next:

- [ ] Colour terms between instrument and catalog bandpasses in `reduce_frame`
- [ ] Galactic and host extinction corrections, so beta can be measured on real data
- [ ] Bayesian (MCMC) fitting for posterior distributions of the break time and `p`
- [ ] More case studies from public archives

## Documentation

- [Science background](docs/science_background.md): afterglows, closure relations, jet breaks, image subtraction, PSF photometry, extinction
- [GRB 221009A case study](case_studies/grb221009a/README.md)
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff)

## Citation

If this project helps your work, please cite it using [`CITATION.cff`](CITATION.cff):

```bibtex
@software{jamwal_photons_to_power_laws,
  author  = {Jamwal, Harishik Dev Singh},
  title   = {Photons to Power Laws: an optical photometry pipeline for gamma-ray burst afterglows},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/Harishik/photons-to-power-laws},
}
```

## Acknowledgements

Built on [Astropy](https://www.astropy.org/) and
[photutils](https://photutils.readthedocs.io/). The case study uses public data
from the Zwicky Transient Facility via the NASA/IPAC Infrared Science Archive
and from Pan-STARRS1 via MAST, and Liverpool Telescope photometry published by
Laskar et al. (2023); see
[`case_studies/grb221009a/data`](case_studies/grb221009a/data/README.md) for
full acknowledgements.

Released under the [MIT License](LICENSE).
