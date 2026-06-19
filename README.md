# Decoding the Decay: An Optical Photometry Pipeline for Gamma-Ray Burst Afterglows

> *Photons to Power Laws* — from raw detector counts to the afterglow's decay index.

A small, readable, end-to-end pipeline that turns **raw optical telescope images
of a Gamma-Ray Burst (GRB) field** into a **calibrated light curve** and fits the
afterglow's **power-law decay** — the same chain of steps used in the
professional GRB literature, implemented with only pip-installable, open tools
(`astropy`, `photutils`, `numpy`, `scipy`, `matplotlib`).

> **Status:** v0.1.0 — the full pipeline runs end-to-end on built-in synthetic
> data and recovers an injected decay index to within the fit error. Real-data
> adapters are on the roadmap below.

![Recovered afterglow light curve](examples/lightcurve.svg)

*Demo output: a synthetic afterglow with injected decay index α = 1.10 is
recovered as α = 1.11 ± 0.004 (reduced χ² ≈ 1.0).*

---

## Why this project

Gamma-Ray Bursts are the most energetic explosions known. Each is followed by a
rapidly fading **afterglow** — synchrotron emission from a decelerating
relativistic blast wave — that we can watch across the electromagnetic spectrum.
In the optical, that afterglow fades roughly as a power law in time,
`F(t) ∝ t^(-α)`. Measuring the decay index `α` and any later steepening from a
*jet break* is how raw pixels become physics. Those slopes test the *closure
relations* linking the observed behaviour to the microphysics of the shock.

This repository walks through that transformation explicitly and reproducibly,
so it doubles as a teaching tool and a starting point for real reductions.

## The pipeline

```
 raw frames ─► calibration ─► detection ─► aperture ─► zero-point ─► light ─► power-law
 (bias/flat)   (master       (DAOStarFinder) photometry  (vs catalog)  curve    fit (α)
                bias/flat)                   (+ errors)
```

Each stage is one self-contained module in `src/grb_afterglow/`:

| Module | Role |
|---|---|
| `calibration.py` | Build master bias/flat; apply `(raw − bias) / flat`. |
| `detection.py` | Spatially varying background + DAOFIND-style source detection; FWHM estimate. |
| `photometry.py` | Circular-aperture photometry with a local sky annulus and a full CCD-equation error budget. |
| `zeropoint.py` | Match field stars to a reference catalog; solve a robust photometric zero point. |
| `lightcurve.py` | A `LightCurve` container with AB magnitude ↔ flux-density conversions. |
| `fitting.py` | Single and smoothly broken jet-break power-law fits via `scipy.curve_fit`. |
| `plotting.py` | Publication-style light-curve and zero-point diagnostic figures. |
| `synthetic.py` | A realistic synthetic-field generator with ground truth. |

## Quickstart

```bash
git clone https://github.com/Harishik/photons-to-power-laws.git
cd photons-to-power-laws
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_demo.py
```

You should see per-epoch photometry, a fitted decay index, a PASS/CHECK against
the injected truth, and figures plus a CSV light curve written to `examples/`.

### Use it as a library

```python
from grb_afterglow import (
    build_master_bias, build_master_flat, calibrate_frame,
    detect_sources, aperture_photometry_at,
    compute_zeropoint, calibrate_magnitudes,
    LightCurve, fit_single_power_law,
)
from grb_afterglow.calibration import load_fits

raw, hdr = load_fits("science_0001.fits")
cal = calibrate_frame(raw, master_bias, master_flat)
sources = detect_sources(cal, fwhm=3.5, threshold_sigma=5.0)
fit = fit_single_power_law(lc.time, lc.flux, lc.flux_err)
print(fit.summary())
```

## Working with real data

The synthetic generator exists so the project runs with zero downloads and can
be validated. To point the pipeline at real observations, supply your own FITS
frames and a real reference catalog. Good public starting points:

- **Swift/UVOT GRB afterglow archive** for UV/optical FITS images of bursts.
- **Pan-STARRS1 DR2** or **SDSS** for field-star reference magnitudes.
- **Gaia** for astrometric reference and image coordinate solving.

Real-world refinements you will likely want include astrometric plate-solving,
image stacking for faint epochs, PSF photometry, host-galaxy subtraction, and
forced photometry at the GRB position.

## Roadmap

- [x] Core reduction → photometry → light curve → power-law fit
- [x] Synthetic data generator with ground truth + end-to-end validation
- [x] Single and broken jet-break power-law fitting
- [ ] `astroquery` adapters for Pan-STARRS / SDSS / Gaia reference catalogs
- [ ] WCS / sky-coordinate matching path wired into the demo
- [ ] PSF photometry option (`photutils.psf`)
- [ ] Image differencing for host-galaxy subtraction
- [ ] Multi-band light curves and colour evolution
- [ ] Closure-relation diagnostics from `α` and the spectral index `β`
- [ ] Notebook tutorial walking through a real burst

See [CHANGELOG.md](CHANGELOG.md) for released changes.

## Tests

```bash
pip install pytest
pytest -q
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built on the [Astropy](https://www.astropy.org/) ecosystem and
[Photutils](https://photutils.readthedocs.io/). The reduction sequence mirrors
standard practice described across the GRB afterglow literature.
