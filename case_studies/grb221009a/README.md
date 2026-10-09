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
   the value in the ZTF header), WCS cross-match with Pan-STARRS1, a per-frame zero
   point, and forced aperture photometry at the Swift/UVOT position.
2. The same measurement on the deep reference image gives the flux already at
   the burst position before the burst (blended field stars and host light):
   6.5 ± 1.0 µJy in r. It is subtracted from every epoch.
3. Epochs are inspected in the cutout filmstrip below. The 8.6-day r-band
   image is rejected because a satellite trail crosses the aperture.
4. A single power law is fitted to the r-band data after one day, where the
   published light curve has already steepened (break at 0.40 ± 0.04 d).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/lightcurve_dark.png">
  <img src="figures/lightcurve_light.png" alt="GRB 221009A ZTF r-band light curve with Liverpool Telescope photometry and a power-law fit" width="100%">
</picture>

Detections in ZTF r (reference-subtracted):

| Days after trigger | Flux density (µJy) | AB mag | S/N |
|---:|---:|---:|---:|
| 1.58 | 69.8 ± 4.9 | 19.29 | 14.1 |
| 3.57 | 20.7 ± 3.3 | 20.61 | 6.4 |
| 5.61 | 10.3 ± 2.9 | 21.36 | 3.6 |

| Fit (t ≥ 1 d, r band) | alpha | chi²/dof |
|---|---:|---:|
| ZTF only, this pipeline | 1.53 ± 0.15 | 7.2/7 |
| Liverpool Telescope only (refitted here) | 1.40 ± 0.02 | 15.7/17 |
| ZTF + Liverpool | 1.40 ± 0.02 | 27.5/26 |
| Laskar et al. (2023), r′ after 0.40 d | 1.42 ± 0.01 | |

The decay index measured from ZTF alone agrees with the published value, and
refitting the Liverpool data with this package's fitter reproduces the
published index. At the three ZTF detections the ZTF magnitudes are
+0.10 to +0.24 mag fainter than the Liverpool power law, comparable to the
per-point errors (0.08 to 0.3 mag). Some offset is expected: ZTF r and the
Liverpool r′ filter are different bandpasses, and this afterglow is much redder
than the field stars that set the zero point, so a colour term would shift it.

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
- **Aperture photometry in a crowded field.** Subtracting the reference flux
  removes static blends on average, but differences in seeing between the
  reference and each epoch leave residuals. Image subtraction would do better.
- **No colour term or extinction correction.** Fine for the decay index, which
  is unaffected by constant offsets, but it is why this case study does not
  measure the spectral index beta.
- **g band.** The g-band frames are too shallow for a detection given the
  extinction, and several have too few calibration stars to be calibrated.
