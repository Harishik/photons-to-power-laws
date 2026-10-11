# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [0.5.0] - 2026-10-11

The four roadmap items of 0.4: light curves from several instruments through
their filter curves, near-infrared data, physical afterglow models, and a PSF
and subtraction kernel that vary across full CCD frames. Also two corrections
to 0.4.0 results.

### Added
- `grb_afterglow.filters`: a registry of 35 system transmission curves (ZTF,
  Pan-STARRS1, SDSS, Liverpool IO:O, Gemini GMOS-S and Flamingos-2, Palomar
  WIRC, 2MASS, HST WFC3) from the SVO Filter Profile Service, committed as
  package data with their provenance (`scripts/fetch_filters.py` rebuilds
  them). Bands are named `"instrument:filter"`. Energy-counter curves are
  converted to photon counting; every pivot wavelength matches SVO's to 0.001 %.
  `register_bandpass` adds your own.
- `grb_afterglow.spectral.SpectralModel`: synthetic photometry of a power law
  behind Galactic (CCM89) and host (SMC) dust through any filter curve,
  precomputed as quadrature weights so that a model evaluation is one weighted
  sum.
- Multi-instrument fitting: `fit_lightcurve(..., ebv_mw=, bandpasses=,
  calibration_offsets=)`. The model is integrated over each band's filter
  curve, the Galactic dust is forward-modelled instead of removed from the
  data, and zero-point offsets per instrument or band are fitted with Gaussian
  priors. `FitResult.predict_band`, `FitResult.offsets` and
  `FitResult.evaluator` give the model as the data see it; MCMC keeps the
  priors.
- `grb_afterglow.mcmc.grid_posterior`: the posterior on a grid of one or two
  difficult parameters, the rest marginalised in the Laplace approximation, for
  posteriors that ensemble MCMC does not converge on. `sample_posterior` gains
  `moves` and `spread_break`.
- `plot_spectrum`: the broadband spectrum of a multi-instrument fit at one
  time, as observed and with the Galactic dust removed, with residuals.
- `grb_afterglow.physical`: physical afterglow models with `afterglowpy`
  (optional extra `[models]`): top-hat and Gaussian jets at any redshift and
  viewing angle behind Galactic dust, fitted by least squares (several starts,
  covariance from steps matched to the uncertainties, because the model is
  computed numerically) and sampled with `emcee` under the usual priors.
- Full CCD frames: `PipelineConfig(tiles=(nx, ny))`.
  - `subtract_images(..., tiles=)` fits a kernel per tile and interpolates it
    across the frame (`DifferenceImage.kernel_at`).
  - `build_spatial_psf` and `psf_photometry_tiled` give a `SpatialPSF`, whose
    Moffat parameters are planes in position, fitted jointly to all PSF stars
    and kept only where significant. The scene is fitted tile by tile, each
    source with the PSF at its own position.
  - Kernel fits on large regions use the pixels near stars plus a sample of
    sky.
  - `dev/full_frame` runs a full 3072 x 3080 ZTF quadrant.
- GRB 210204A, `physical_model.py`:
  - Kumar et al.'s published top-hat jet describes the ZTF light curve up to
    one flux scale (16 %).
  - With their density and epsilon_B, ZTF alone gives p = 2.16 and
    theta_core = 0.022 (theirs: 2.18 and 0.024).
  - With epsilon_B free and the density fixed at values 50 times apart
    (10^-5.67 and 10^-4 cm^-3), the fits are equally good: energy, density and
    epsilon_B are degenerate in optical data.
- `query_panstarrs(..., constraints=)` for wide searches.
- GRB 221009A: the spectral index from Liverpool (Laskar et al. 2023), Lowell
  and Gemini (O'Connor et al. 2023) photometry from g to Ks, 0.66 ± 0.03 (stat) ± 0.13 (E(B−V)) ±
  0.05 (photometric system), in agreement with O'Connor et al. (0.1σ) and
  Laskar et al. (0.7σ). The near infrared shrinks the ambiguity of which
  photometric system the published optical magnitudes are in from 0.46 vs 0.91
  (griz alone) to 0.66 vs 0.76.

### Changed
- Least-squares fits use `scipy.optimize.least_squares` directly (results are
  unchanged to 1e-9), so that priors can enter as extra residuals.
- `power_law_color` integrates over filter curves where the registry has them.
- Plots draw each band's model through its filter curve and calibration
  offset; bands are ordered by wavelength.

### Fixed
- The Liverpool IO:O r-band curve is an energy-counter response (SVO); 0.4.0
  treated it as photon counting, which put its pivot at 618.5 instead of
  616.2 nm.
- GRB 210204A: the 0.4.0 posterior came from an MCMC chain that had not
  converged. The posterior is bimodal (a break between 1 and 2 days, or right
  after the only early epoch with the early decay unconstrained) and is now
  integrated on a grid: t_b = 0.95 ± 0.88 d (was 1.3 +0.7 −0.8 d),
  alpha2 = 1.28 +0.47 −0.16 (was 1.44 +0.43 −0.26).

## [0.4.0] - 2026-10-10

The remaining roadmap items: colour terms, dust extinction, Bayesian
posteriors, a second real-data case study, and the completed method bake-off.

### Added
- `grb_afterglow.extinction`: the Milky Way extinction curve of Cardelli,
  Clayton & Mathis (1989) and the SMC curve of Pei (1992) for host galaxies
  (both checked against `dust_extinction` to machine precision); `Bandpass`
  transmission curves with pivot wavelengths and band-averaged extinction for
  a power-law source; `deredden`; `power_law_color` for an afterglow's colour
  behind dust.
- Host-galaxy extinction in the fit: `fit_lightcurve(..., host_redshift=z)`
  fits `host_av` alongside the spectral index.
- `grb_afterglow.mcmc.sample_posterior`: posterior sampling of any power-law
  fit with `emcee` (new dependency), with credible intervals and `p` derived
  sample by sample; `plot_posterior` corner plots and `plot_sed`.
- Colour terms: `PipelineConfig.color_terms` and `target_color`; zero points
  evaluated at the burst's colour (`ZeroPoint.at_color`), with the slope's
  uncertainty propagated; slopes can be fixed, e.g. after measuring them once
  from many frames.
- GRB 221009A: the optical spectral index from Liverpool griz photometry with
  the Galactic extinction averaged over the IO:O filter curves,
  0.41 ± 0.06 (stat) ± 0.18 (E(B−V)), within 1.4σ of O'Connor et al. (2023);
  ZTF zero points with a colour term measured once from all frames.
- A second case study, GRB 210204A, from 47 public ZTF g, r, i images: a jet
  break, beta = 0.92 ± 0.19 and the ISM closure relation with
  p = 2.84 ± 0.38, consistent with Kumar et al. (2022). Both case studies run
  in CI.
- The demo samples the posterior of the jet-break fit and draws its corner plot.
- The three unfinished bake-off candidates were completed and scored:
  an analytic Moffat PSF, Alard-Lupton and ZOGY subtraction (`dev/bakeoff/`).
- 15 new tests (106 in total).

### Fixed
- Bake-off harness: the Alard-Lupton kernel was applied in the wrong
  orientation for the real-data calibration, and ZOGY was scored through the
  crowding-biased curve-of-growth fallback (chi²/pt 16.7 instead of 0.42).

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

[0.5.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Harishik/photons-to-power-laws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Harishik/photons-to-power-laws/releases/tag/v0.1.0
