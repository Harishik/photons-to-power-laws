# Case study: GRB 221009A in public ZTF images

GRB 221009A, the "brightest of all time", went off on 2022 October 9 close to
the Galactic plane. Its optical afterglow sits behind several magnitudes of
dust in a crowded star field: a hard target for simple aperture photometry,
and a good test of whether this pipeline holds up on real data.

This case study runs the pipeline, unchanged, on the public Zwicky Transient
Facility (ZTF) survey images of the field and compares the result with
independent Liverpool Telescope photometry published by
[Laskar et al. (2023)](https://doi.org/10.3847/2041-8213/acbfad).

```bash
python case_studies/grb221009a/analyze.py      # runs offline on the committed data
python case_studies/grb221009a/fetch_data.py   # optional: rebuild data/ from IRSA and MAST
```

## Data

- **ZTF**: every public image covering the burst from one day before to 40
  days after the trigger (field 590, CCD 11, quadrant 4; 30 s exposures), as
  120″ cutouts, plus the deep ZTF reference image in each band. IRSA serves
  22 of the 27 epochs it lists.
- **Pan-STARRS1 DR1** stars within 1.6′, for the zero points.
- **Liverpool Telescope** griz photometry from Table 2 of Laskar et al. (2023),
  used only for comparison.

Provenance and acknowledgements are in [`data/README.md`](data/README.md).

## Method

1. Each ZTF cutout goes through `grb_afterglow.reduce_frame`: background and
   source detection, seeing from isolated stars (typically within 0.1 px of
   the value in the ZTF header), WCS cross-match with Pan-STARRS1, and a
   per-frame zero point from empirical-PSF photometry of the matched stars.
2. The deep pre-burst ZTF reference image (16 frames co-added, FWHM 2.2 px) is
   used as the template: it is resampled onto each epoch's pixel grid,
   PSF-matched with a Bramich (2008) kernel, and subtracted, which removes the
   blended field stars and any host light. In the three detected epochs the
   seeing is better than in the reference, so the science image is the one
   convolved and the PSF model is carried through the same kernel.
3. The afterglow is measured on the difference image by a forced PSF fit at the
   Swift/UVOT position, with an error from identical fits at blank positions
   (convolution correlates the pixel noise).
4. Epochs are inspected in the cutout filmstrip below. The 8.6-day r-band
   image is rejected because a satellite trail crosses the aperture.
5. A single power law is fitted to the r-band data after one day, where the
   published light curve has already steepened (break at 0.40 ± 0.04 d).
6. For comparison, the v0.2 measurement is repeated: aperture photometry of
   each epoch minus the flux in the same aperture on the reference image
   (6.5 ± 1.0 µJy in r).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/lightcurve_dark.png">
  <img src="figures/lightcurve_light.png" alt="GRB 221009A ZTF r-band light curve with Liverpool Telescope photometry and a power-law fit" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/subtraction_dark.png">
  <img src="figures/subtraction_light.png" alt="ZTF cutout at 1.6 days PSF-matched to the reference image, the reference, and the difference image with only the afterglow left" width="100%">
</picture>

Detections in ZTF r (difference imaging, PSF photometry), with the v0.2
measurement for comparison:

| Days after trigger | Flux density (µJy) | AB mag | S/N | v0.2: aperture − reference (µJy) | v0.2 S/N |
|---:|---:|---:|---:|---:|---:|
| 1.58 | 76.2 ± 3.6 | 19.20 | 21.0 | 69.8 ± 4.9 | 14.1 |
| 3.57 | 21.1 ± 2.3 | 20.59 | 9.1 | 20.7 ± 3.3 | 6.4 |
| 5.61 | 15.0 ± 2.2 | 20.96 | 6.7 | 10.3 ± 2.9 | 3.6 |
| 10.56 | 10.4 ± 3.0 | 21.36 | 3.5 | 6.7 ± 3.5 | 1.9 |

| Fit (t ≥ 1 d, r band) | alpha | chi²/dof |
|---|---:|---:|
| ZTF only, difference imaging + PSF | 1.34 ± 0.09 | 6.7/7 |
| ZTF only, v0.2 (aperture − reference flux) | 1.53 ± 0.15 | 7.2/7 |
| Liverpool Telescope only (refitted here) | 1.40 ± 0.02 | 15.7/17 |
| ZTF + Liverpool | 1.40 ± 0.02 | 22.8/26 |
| Laskar et al. (2023), r′ after 0.40 d | 1.42 ± 0.01 | |

Image subtraction with PSF photometry raises the signal-to-noise of every
detection by a factor of 1.4 to 1.9, adds a fourth detection, and shrinks the
uncertainty of the ZTF-only decay index by 40 %; both methods agree with the
published value (0.9σ and 0.8σ), and refitting the Liverpool data with this
package's fitter reproduces the published index. At the first three ZTF
detections the ZTF magnitudes are within ±0.17 mag of the Liverpool power law
(+0.00, +0.15, −0.17 mag), comparable to the per-point errors (0.05 to
0.16 mag); the marginal 10.6-day point lies 1.7σ above it. Some offset is
expected: ZTF r and the Liverpool r′ filter are different bandpasses, and this
afterglow is much redder than the field stars that set the zero point, so a
colour term would shift it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/filmstrip_dark.png">
  <img src="figures/filmstrip_light.png" alt="ZTF r-band cutouts of the burst position over time, with one epoch flagged for a satellite trail" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/zeropoints_dark.png">
  <img src="figures/zeropoints_light.png" alt="Zero-point solution for one ZTF frame and per-frame zero points" width="100%">
</picture>

Per-epoch measurements are in [`results/ztf_photometry.csv`](results/ztf_photometry.csv)
and the numbers above in [`results/summary.json`](results/summary.json).

## Limitations

- **Survey cadence.** ZTF's public survey visits the field every two to
  three nights with 30 s exposures, so the afterglow is detected only three
  times before it fades below about r = 21.5.
- **Calibrating the difference image.** The reference image has worse seeing
  than the detected epochs, so the difference image carries the reference's
  PSF and its flux scale has to be tied back to the science image's zero
  point. The obvious shortcut, an aperture correction from the curve of
  growth of stars in the deep reference, fails here: in this crowded field,
  neighbouring stars inside the large "total" apertures inflate it, and the
  fluxes came out twice too high. The pipeline instead carries the science
  PSF through the matching kernel (see
  [`docs/science_background.md`](../../docs/science_background.md)).
- **No colour term or extinction correction.** Fine for the decay index, which
  is unaffected by constant offsets, but it is why this case study does not
  measure the spectral index beta.
- **g band.** The g-band frames are too shallow for a detection given the
  extinction, and several have too few calibration stars to be calibrated.
