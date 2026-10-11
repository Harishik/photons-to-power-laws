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
  for comparison and for the spectral index.
- **Lowell, Gemini, Palomar and HST** optical and near-infrared photometry
  from [O'Connor et al. (2023)](https://doi.org/10.1126/sciadv.adi1405). The
  Lowell and Gemini points before 20 days go into the spectral index.
- The **transmission curves** of every filter involved come from the package's
  filter registry (SVO Filter Profile Service).

Provenance and acknowledgements are in [`data/README.md`](data/README.md).

## Method

1. Each ZTF cutout goes through `grb_afterglow.reduce_frame`: background and
   source detection, seeing from isolated stars (typically within 0.1 px of
   the value in the ZTF header), WCS cross-match with Pan-STARRS1, and a
   per-frame zero point from empirical-PSF photometry of the matched stars,
   with a colour term (below).
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
7. The spectral index comes from all the optical and near-infrared
   photometry between 1.7 and 20 days (below). A power law with a common decay
   and spectral index is fitted to every band at once. The Galactic extinction
   (E(B−V) = 1.32 ± 0.06, Schlafly & Finkbeiner 2011 as derived by Shrestha et
   al. 2023; Cardelli et al. 1989 curve) is applied to the model and integrated
   over each instrument's filter curve, so the data are fitted as observed.
   The posterior is sampled with MCMC.
8. ZTF r and Pan-STARRS1 r are different bandpasses, and behind this much dust
   the afterglow (Pan-STARRS1 g − r = 1.33 for the measured beta, integrated
   over both filters) is far redder than the calibration stars. The colour
   term, 0.141 ± 0.010 per magnitude of g − r, is measured once as the median
   of the per-frame fits over all r frames, and each frame's zero point is
   evaluated at the afterglow's colour.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/lightcurve_dark.png">
  <img src="figures/lightcurve_light.png" alt="GRB 221009A ZTF r-band light curve with Liverpool Telescope photometry and a power-law fit" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/subtraction_dark.png">
  <img src="figures/subtraction_light.png" alt="ZTF cutout at 1.6 days PSF-matched to the reference image, the reference, and the difference image with only the afterglow left" width="100%">
</picture>

Detections in ZTF r (difference imaging, PSF photometry, colour term), with
the v0.2 measurement for comparison:

| Days after trigger | Flux density (µJy) | AB mag | S/N | v0.2: aperture − reference (µJy) | v0.2 S/N |
|---:|---:|---:|---:|---:|---:|
| 1.58 | 78.5 ± 3.9 | 19.16 | 20.1 | 69.8 ± 4.9 | 14.1 |
| 3.57 | 21.9 ± 2.4 | 20.55 | 9.1 | 20.7 ± 3.3 | 6.4 |
| 5.61 | 15.3 ± 2.3 | 20.94 | 6.7 | 10.3 ± 2.9 | 3.6 |
| 10.56 | 10.9 ± 3.2 | 21.30 | 3.5 | 6.7 ± 3.5 | 1.9 |

| Fit (t ≥ 1 d, r band) | alpha | chi²/dof |
|---|---:|---:|
| ZTF only, difference imaging + PSF | 1.34 ± 0.09 | 6.5/7 |
| ZTF only, v0.2 (aperture − reference flux) | 1.53 ± 0.15 | 7.2/7 |
| Liverpool Telescope only (refitted here) | 1.40 ± 0.02 | 15.7/17 |
| ZTF + Liverpool | 1.40 ± 0.02 | 22.8/26 |
| Laskar et al. (2023), r′ after 0.40 d | 1.42 ± 0.01 | |

Image subtraction with PSF photometry raises the signal-to-noise of every
detection by a factor of 1.4 to 1.9, adds a fourth detection, and shrinks the
uncertainty of the ZTF-only decay index by 40 %; both methods agree with the
published value (0.9σ and 0.8σ), and refitting the Liverpool data with this
package's fitter reproduces the published index. At the first three ZTF
detections the ZTF magnitudes are within 0.19 mag of the Liverpool power law
(−0.03, +0.11, −0.19 mag), comparable to the per-point errors (0.05 to
0.16 mag); the marginal 10.6-day point lies 1.7σ above it. The colour term
makes the afterglow 2 to 3 % brighter than a zero point at the stars' average
colour would; the remaining differences are within the errors, and ZTF r and
the Liverpool r′ filter are still different bandpasses.

### The spectral index behind the Galactic plane

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/sed_dark.png">
  <img src="figures/sed_light.png" alt="Optical to near-infrared spectrum of GRB 221009A at 4.4 days, as observed and with the Galactic extinction removed" width="85%">
</picture>

The Galactic dust removes 4.7 magnitudes in g but only 0.5 in Ks. Between 1.7
and 20 days
the spectral index is measured from 62 points in 12 bands:

- Liverpool griz (Laskar et al. 2023);
- Lowell LMI griz, Gemini GMOS-S i, and Gemini Flamingos-2 J, H and Ks
  (O'Connor et al. 2023).

After 20 days a supernova may start to contribute ([Fulton et al.
2023](https://arxiv.org/abs/2301.11170) discuss one emerging at about that
time), so the later Palomar and HST photometry is
not used. The bands come from four instruments calibrated against two
catalogues, so the fit is set up as follows:

- **Filter curves.** Each measurement is modelled through its own
  instrument's filter curve (`lt:r`, `f2:Ks`, ...). Lowell's LMI filters are
  SDSS-type and not in the SVO service, so the SDSS curves stand in for them.
- **Calibration offsets.** Every band except Liverpool r, which sets the flux
  scale, has a zero-point offset with a Gaussian prior: 0.03 mag for the
  optical bands (Liverpool and Lowell are calibrated to Pan-STARRS1, without
  colour terms), and 0.05 mag for the near-infrared bands, which are
  calibrated to 2MASS and converted from Vega to AB.
- **Error floor.** A 2 % error floor is added to every point.

| Data, model | beta | ± E(B−V) | chi²/dof |
|---|---:|---:|---:|
| Liverpool griz, instrument filter curves | 0.46 ± 0.09 | 0.18 | 37/37 |
| Liverpool griz, Pan-STARRS1 curves | 0.91 ± 0.10 | 0.19 | 39/37 |
| **Optical + near-infrared, instrument filter curves (MCMC)** | **0.66 ± 0.03** | **0.13** | 83/48 |
| Optical + near-infrared, Pan-STARRS1 and 2MASS curves | 0.76 ± 0.03 | 0.12 | 93/48 |
| O'Connor et al. (2023), optical/IR after 1.7 d, same E(B−V) | 0.68 ± 0.05 | | |
| Laskar et al. (2023), NIR to optical at 4.4 d | 0.76 ± 0.04 | | |

With the near infrared, beta = 0.66 ± 0.03 (statistical and calibration)
± 0.13 (E(B−V)) ± 0.05 (photometric system). That agrees with O'Connor et al.
at 0.1σ and Laskar et al. at 0.7σ. Neither comparison is independent:

- 19 of the 62 points are O'Connor et al.'s: the Lowell and Gemini ones,
  including all of the near infrared;
- the other 43 are Laskar et al.'s Liverpool photometry.

The table shows what the near infrared buys:

- **Photometric system.** Published magnitudes calibrated against
  Pan-STARRS1 stars without colour terms lie close to the instrument's own
  system (what is fitted here). With a colour term evaluated at the
  afterglow's colour they would be in the Pan-STARRS1 system instead. For a
  source this red the two readings disagree badly: from Liverpool's griz
  alone, beta is 0.46 or 0.91 depending on which one is assumed. The
  near-infrared bands pin down the slope over a factor of four in wavelength,
  and the difference shrinks to 0.66 against 0.76.
- **Dust.** The near infrared barely helps with the uncertain E(B−V): the
  extinction difference between g and Ks still changes with it, and ±0.06
  moves beta by ±0.13 instead of ±0.18.

The fit is not perfect (chi²/dof = 1.7): a few bands need offsets of 0.06 to
0.09 mag, beyond their priors (Lowell i, Gemini i, Liverpool z, Flamingos-2 J
and Ks). These are the band-to-band calibration differences expected from
heterogeneous photometry of a source this red.

The decay index from the same fit is alpha = 1.368 ± 0.009. The posterior is
in [`figures/posterior_light.png`](figures/posterior_light.png). With beta
still uncertain by 0.14, the closure relations cannot single out an emission
regime:

- a wind medium below the cooling frequency agrees at 0.6σ;
- a uniform medium agrees at 1.8σ;
- the post-jet-break relation above the cooling frequency would need p = 1.3,
  below the p > 2 it assumes.

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
- **The spectral index is limited by the dust.** E(B−V) toward this
  low-latitude field is itself uncertain (Shrestha et al. 2023 discuss values
  from 0.71 to 1.32), and the model assumes Milky Way dust with R_V = 3.1
  and no dust in the host (O'Connor et al. 2023 find E(B−V) < 0.1 there). Even
  with the near infrared, E(B−V) ± 0.06 is the largest error on beta.
- **Calibration of literature photometry.** The published magnitudes have no
  colour terms. The calibration offsets absorb grey shifts per band, but not
  the colour-dependent difference between an instrument and its calibration
  catalogue, which is why the photometric-system systematic is quoted. The
  Lowell LMI filters are modelled with the SDSS curves.
- **g band.** The g-band frames are too shallow for a detection given the
  extinction, and several have too few calibration stars to be calibrated.
