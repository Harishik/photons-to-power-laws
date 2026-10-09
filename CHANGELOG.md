# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - 2026-10-10

A rewrite of the reduction core, a physically realistic validation suite, and
the first application to real telescope data.

### Fixed
- **The demo no longer recovered the injected decay index.** With current
  photutils it returned `alpha = 0.09` instead of `1.10`: the FWHM estimate
  (derived from DAOFIND "sharpness", which is not a width) came out at 7.6 px
  for 3.2 px stars, so the aperture swallowed a neighbouring star. The seeing is
  now measured by fitting 2-D Gaussians to isolated stars.
- CI failed on Python 3.10 because photutils renamed `xcentroid` to
  `x_centroid`. Column names and renamed keyword arguments are now handled for
  every supported photutils version (1.x, 2.x, 3.x).
- Sky annuli were biased high by the wings of faint neighbouring stars, which
  biased faint forced photometry low by about 0.5 sigma. Sky annuli now exclude
  a dilated source mask; blank-sky photometry is unbiased (tested).
- The sky-estimation term was missing from the photometric error budget.
- Plots assumed `t_ref = 1000 s` regardless of the value used in the fit.
- Pixel-space matching could assign several reference stars to one source.

### Added
- `grb_afterglow.pipeline`: `reduce_frame` and `run_pipeline` take calibrated
  or raw frames with a WCS to a calibrated, multi-band light curve, with seeing
  matched apertures, WCS cross-matching, per-frame zero points, saturation
  handling, and a local astrometric correction.
- `grb_afterglow.closure`: the standard closure relations (ISM and wind, slow
  cooling, before and after the jet break) and a test that ranks them against
  a measured `(alpha, beta)` and infers the electron index `p`.
- Joint multi-band fitting with a shared temporal model and spectral index
  `beta`; multi-start broken power-law fits; AIC/BIC model comparison; an
  optional systematic error floor.
- A realistic synthetic campaign: WCS with dithers, Moffat PSFs with variable
  seeing, variable transparency, saturating stars, column-structured bias,
  vignetted flats, exposure times that grow with time, and a g/r/i afterglow
  with a jet break that obeys a closure relation.
- `grb_afterglow.archives`: ZTF image search and cutouts (IRSA) and
  Pan-STARRS1 cone searches (MAST), using only the standard library.
- Case study: **GRB 221009A** reduced from public ZTF images, validated
  against published Liverpool Telescope photometry
  (`case_studies/grb221009a/`).
- `p2pl` command-line tool: `demo`, `simulate`, and `fit`.
- Publication-style figures in light and dark themes (light curve with
  residuals, zero-point diagnostics, closure-relation plane, pipeline stages,
  cutout filmstrip) and an animated GIF.
- Test suite split by module (70 tests), including statistical checks that
  error bars are honest and an end-to-end recovery test.
- Ruff linting and formatting, pre-commit hooks, coverage, and a CI matrix on
  Python 3.10 to 3.13.

### Changed
- `LightCurve` stores flux density (µJy) rather than magnitudes, so negative
  forced-photometry fluxes are kept; use `LightCurve.from_magnitudes` for
  magnitude input. It now carries a band per point.
- `estimate_background` returns a `BackgroundModel` with the background and its
  RMS map.
- Photometry uses the measured sky scatter instead of a nominal read noise;
  the `read_noise` argument was removed.
- `generate_dataset` takes a `SyntheticConfig` and writes a sky-coordinate
  catalog (`catalog.csv`) and a `truth.json`.

### Removed
- Hand-drawn SVG figures; every figure is now produced by the pipeline.

## [0.1.0] - 2026-06-20

### Added
- Initial calibration, detection, aperture photometry, zero-point, light-curve,
  fitting, plotting, and synthetic-data modules, a demo script, and tests.

[0.2.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Harishik/photons-to-power-laws/releases/tag/v0.1.0
