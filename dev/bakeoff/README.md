# PSF-photometry and image-subtraction bake-off

How the methods in `grb_afterglow.psf` and `grb_afterglow.subtraction` were chosen. `harness.py`
scores a candidate on the same synthetic data (with known truth) and real data (the ZTF epochs of
GRB 221009A, checked against Liverpool Telescope photometry):

```bash
cd dev/bakeoff
python harness.py psf psf_empirical/method.py           # add --quick while iterating
python harness.py subtraction sub_bramich/method.py
```

Each candidate directory holds `method.py` and its `results_*.json`. The two winners now point at
the shipped package code, so the harness doubles as a validation of what the pipeline runs (the
packaged versions reproduce the prototypes' scores). `HARNESS_CHANGELOG.md` records two fixes to
the harness itself; results from before a fix are not comparable.

## PSF photometry: `psf_empirical`

An empirical PSF (pixel-integrated elliptical Moffat plus an oversampled, regularised residual)
built from each image, fitted to all sources at once. Synthetic fields at three seeings, sparse and
crowded, 2226 stars; real data are the 11 ZTF frames.

| Metric | Empirical PSF | Apertures (`baseline_aperture`) |
|---|---:|---:|
| Bias on bright stars | +0.02 % | |
| Robust spread of the pulls (all / crowded) | 1.02 / 1.01 | |
| Median error of faint stars | 9.2 % | 30.1 % |
| ZTF zero-point scatter against Pan-STARRS1 | 0.055 mag | 0.075 mag |

An analytic-PSF variant (`photutils.psf` with a fitted Moffat) was started but not finished.

## Image subtraction: `sub_bramich`

A delta-function-basis kernel (Bramich 2008) with an errors-in-variables correction, Laplacian
regularisation, sigma clipping and a bright-pixel scale refit. Synthetic pairs with a transient on
a PSF-convolved host galaxy, in four seeing configurations (science sharper, template sharper,
matched, undersampled), 10 noise realisations, three transient fluxes.

| Metric | Bramich | No PSF matching (`baseline_naive`, 4 realisations) |
|---|---:|---:|
| Pulls in a 1.2 FWHM aperture: mean / spread | +0.02 / 1.13 | −3.5 / 11.9 |
| Pulls with no transient: spread | 1.07 | 12.1 |
| Noise relative to the ideal (no correlated noise) | 0.995 | 1.32 |
| Residual of bright static stars | 0.5 % of their flux | 21 % |
| Real ZTF data, chi²/pt against Liverpool (detected / late epochs) | 0.23 / 0.65 | |

Alard-Lupton (Gaussian basis) and ZOGY prototypes were started but not finished; the Bramich kernel
met every target first, without assumptions about either PSF.

### The 2x flux excess on ZTF data was the harness

The first real-data run gave fluxes about twice the Liverpool prediction (chi²/pt ≈ 26). The
method was fine; the harness's calibration was not. In the detected epochs the seeing is better
than in the ZTF reference, so the science image is convolved and the difference image carries the
reference PSF. The harness corrected for that with the curve of growth of stars in the deep
reference image, but in this crowded Galactic-plane coadd the large "total" apertures also collect
neighbouring stars: the curve of growth claimed 37 % of a point source's light within 1.2 FWHM,
where a median-stacked PSF of the same stars gives 80 %. Calibrating instead with the zero-point
stars measured on the kernel-convolved science image (exactly the difference image's PSF) gives
81, 25 and 14 µJy at 1.6, 3.6 and 5.6 days against a predicted 76, 24 and 13. The pipeline uses
the same approach, and for PSF photometry carries the science PSF through the kernel
(`grb_afterglow.psf.ConvolvedPSF`).
