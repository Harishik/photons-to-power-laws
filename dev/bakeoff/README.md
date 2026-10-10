# PSF-photometry and image-subtraction bake-off

How the methods in `grb_afterglow.psf` and `grb_afterglow.subtraction` were chosen. Five candidates
were written independently and scored by the same harness, `harness.py`, on synthetic data with known
truth and on real data (the ZTF epochs of GRB 221009A, checked against Liverpool Telescope
photometry):

```bash
cd dev/bakeoff
python harness.py psf psf_empirical/method.py           # add --quick while iterating
python harness.py subtraction sub_bramich/method.py
```

Each candidate directory holds `method.py` and its `results_*.json` (full runs; the two baselines
were scored with `--quick`). The two winners point at the shipped package code, so the harness
doubles as a validation of what the pipeline runs. `HARNESS_CHANGELOG.md` records the fixes to the
harness itself; every result here was produced after the last of them.

## PSF photometry

Synthetic fields at three seeings, sparse and crowded (2226 stars); real data are the 11 ZTF frames.

| Metric | `psf_empirical` (shipped) | `psf_analytic` | Apertures (`baseline_aperture`) |
|---|---:|---:|---:|
| Bias on bright stars | +0.02 % | −0.13 % | |
| Robust spread of the pulls (all / crowded) | 1.02 / 1.01 | 1.03 / 1.02 | |
| Median error of faint stars | 9.2 % | 9.4 % | 30.1 % |
| ZTF zero-point scatter against Pan-STARRS1 | **0.055 mag** | 0.073 mag | 0.075 mag |
| Run time (full harness) | 27 s | 109 s | |

* **`psf_empirical`**: a pixel-integrated elliptical Moffat plus an oversampled, regularised residual
  map, built from each image and fitted to all sources at once.
* **`psf_analytic`**: the same Moffat without the residual map, fitted jointly to isolated stars, with
  box-wise forced photometry, neighbour fitting and a PSF-model error from the shape covariance.

On the synthetic fields, whose stars *are* Moffat profiles, the two are equivalent. On real ZTF
frames only the empirical model improves on apertures: real PSFs are not Moffat profiles, and the
residual map is what captures the difference. It is also four times faster.

## Image subtraction

Synthetic pairs with a transient on a PSF-convolved host galaxy, in four seeing configurations
(science sharper, template sharper, matched, undersampled), 10 noise realisations and three
transient fluxes each.

| Metric | `sub_bramich` (shipped) | `sub_alard` | `sub_zogy` | No PSF matching (`baseline_naive`, quick) |
|---|---:|---:|---:|---:|
| Pulls in a 1.2 FWHM aperture: mean / spread | +0.02 / 1.13 | −0.05 / 1.17 | +0.04 / 1.17 | −3.5 / 11.9 |
| Pulls with no transient: spread | 1.07 | 1.12 | 1.15 | 12.1 |
| Bias on a bright transient | −1.7 % | −2.0 % | −0.7 % | −16 % |
| Noise relative to the ideal | 0.995 | 0.993 | 1.07 | 1.32 |
| Residual of bright static stars | 0.5 % | 0.5 % | 0.6 % | 21 % |
| Real ZTF data: chi²/pt against Liverpool, detected / late epochs | 0.23 / 0.65 | 0.20 / 0.46 | 0.42 / 0.31 | |
| Real ZTF data: median flux error | 4.1 µJy | 3.9 µJy | 6.1 µJy | |
| Run time (full harness) | 170 s | 155 s | 551 s | |

* **`sub_bramich`**: a delta-function-basis kernel (Bramich 2008) with an errors-in-variables
  correction, Laplacian regularisation, sigma clipping and a bright-pixel scale refit.
* **`sub_alard`**: the Alard & Lupton (1998) basis of Gaussians times polynomials, with a fourth, wide
  Gaussian for Moffat wings, the same errors-in-variables idea (corrected least squares in the
  noise-orthonormal eigenbasis) and quintic-spline resampling.
* **`sub_zogy`**: ZOGY proper image subtraction (Zackay, Ofek & Gal-Yam 2016) with Moffat PSFs fitted to
  both images and the flux ratio measured on the cross-convolved images.

Bramich and Alard-Lupton are statistically indistinguishable: Bramich's errors are a little better
calibrated on synthetic data (pull spreads 1.13 and 1.07 against 1.17 and 1.12), Alard-Lupton's
difference images are a little less noisy on ZTF. Bramich was kept because its kernel needs no
choice of basis widths and degrees. ZOGY has the smallest bright-flux bias, but on real data it
is the noisiest of the three, because its analytic Moffat PSFs do not describe the ZTF PSF (the
proper-subtraction optimality assumes known PSFs), and it is three times slower.

### Problems found while finishing the candidates

* **Kernel orientation.** The harness calibrates real data by convolving the science image with the
  method's kernel, using `scipy.ndimage.correlate`. The Alard-Lupton kernel was fitted for true
  convolution; returned unflipped, any asymmetry from residual misregistration was mirrored. Flipping
  it moved the real-data chi²/pt of the detected epochs from 0.24 to 0.20.
* **ZOGY and the real-data calibration.** ZOGY returned no kernel, so the harness fell back to the
  template's curve of growth (below) and scored chi²/pt = 16.7 on the detected ZTF epochs. ZOGY now
  returns the filter it applies to the science image, `K_n`, and the PSF of its difference image,
  whose encircled energy replaces the Moffat profile the harness assumed: 0.42.
* **The analytic PSF re-centres bright targets** (S/N > 10) instead of keeping their positions fixed.
  The harness passes exact positions, so this can only add scatter; it is visible as a slightly
  wider pull distribution but did not change the ranking.

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
