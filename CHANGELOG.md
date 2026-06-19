# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project
follows semantic versioning.

## [0.1.0] - 2026-06-20

### Added
- Calibration module: master bias/flat construction and frame reduction.
- Detection module: spatially varying background estimation and DAOFIND-style
  source detection, plus a marginal-profile FWHM estimator.
- Photometry module: circular-aperture photometry with a local sky annulus and
  a full CCD-equation uncertainty budget (source + sky + read noise).
- Zero-point module: pixel- and sky-coordinate catalog matching and a robust
  sigma-clipped photometric zero-point solution.
- Light-curve container with AB magnitude to flux-density conversions and CSV
  export.
- Fitting module: single and smoothly broken jet-break power-law models.
- Plotting module: publication-style light-curve and zero-point diagnostics.
- Synthetic GRB-field generator with a realistic detector model and ground
  truth for validation.
- End-to-end demo (`scripts/run_demo.py`) that recovers an injected decay index.
- Test suite (`pytest`) covering each stage and end-to-end index recovery.

[0.1.0]: https://github.com/Harishik/photons-to-power-laws/releases/tag/v0.1.0
