# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0] - 2026-10-10

Image subtraction and PSF photometry, the two "future extensions" of the
original design, chosen by a documented method comparison (`dev/bakeoff/`).

### Added
- `grb_afterglow.subtraction.subtract_images`: difference imaging against a
  deep template. The template is resampled through the WCS and the sharper
  image is convolved with a delta-function-basis kernel (Bramich 2008) fitted
  with a smooth differential background, an errors-in-variables correction
  for the noise in the convolved image, Laplacian regularisation, sigma
  clipping, and a bright-pixel refit of the photometric scale. Returns a
  `DifferenceImage` in science counts.
- `grb_afterglow.psf`: empirical PSF photometry. A pixel-integrated elliptical
  Moffat plus an oversampled, regularised residual map built from the frame's
  own stars; all sources fitted simultaneously (deblending) with a smooth
  background; PSF-model errors. `build_psf`, `psf_photometry`, `ConvolvedPSF`
  and `forced_psf_flux` (forced fits on difference images with errors from
  blank positions).
- `PipelineConfig.photometry = "aperture" | "psf"` (zero-point stars and the
  burst), `Template`, `reduce_frame(template=...)` and
  `run_pipeline(templates={band: path})`.
- The simulator adds an r = 21 host galaxy under the afterglow and writes a
  deep template per band. The demo reduces with templates and PSF photometry,
  and also without subtraction for comparison: the host flattens the late
  decay from alpha2 = 2.2 to 1.34 unless it is subtracted.
- Figures: `plot_subtraction` (science, matched template, difference) for the
  demo and the ZTF case study, and a with/without-subtraction light curve.
- A redrawn architecture diagram, rendered from HTML for light and dark themes
  (`docs/assets/make_architecture.py`).
- 21 new tests (91 in total) for subtraction, PSF photometry, the template
  pipeline in both seeing directions, and the seeing estimate.

### Changed
- GRB 221009A case study: the ZTF reference image is subtracted and the burst
  is measured with PSF photometry. Detections rise from three to four, their
  S/N by 1.4 to 1.9 times; the ZTF-only decay index is 1.34 ± 0.09 (was
  1.53 ± 0.15; Laskar et al. 2023: 1.42 ± 0.01). The v0.2 measurement is kept
  alongside for comparison.

### Fixed
- When the template has the worse seeing, the flux on the difference image is
  now calibrated with the zero-point stars on the kernel-convolved science
  image (or, for PSF photometry, with the science PSF seen through the
  kernel). The earlier curve-of-growth correction from template stars was
  inflated by crowding and gave ZTF fluxes about twice too high.
- The seeing estimate could be set by a single object: when few sources passed
  the isolation cut (with photutils 2.0 in a frame with many detections, only
  the host galaxy did), its width was returned. The isolation radius is now
  relaxed until at least five stars are fitted.

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

[0.3.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Harishik/photons-to-power-laws/releases/tag/v0.1.0
