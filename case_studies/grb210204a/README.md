# Case study: GRB 210204A, an afterglow ZTF found itself

GRB 210204A was a long burst detected by Fermi/GBM on 2021 February 4. Its
coarse gamma-ray localisation would have made the afterglow hard to find, but
the Zwicky Transient Facility's survey happened to image the field 38 minutes
later and caught it at r = 17.1 (ZTF21aagwbjr, GCN Circular 29405). ZTF kept
observing the field in g, r and i for a month. At redshift 0.876 it is also
known for some of the most delayed optical flares ever seen in an afterglow
([Kumar et al. 2022](https://arxiv.org/abs/2204.07587)).

Unlike GRB 221009A, this field is far from the Galactic plane
(E(B−V) = 0.028), so ZTF's three bands measure the spectral index directly.

```bash
python case_studies/grb210204a/analyze.py      # runs offline on the committed data (~2 minutes)
python case_studies/grb210204a/fetch_data.py   # optional: rebuild data/ from IRSA and MAST
```

## Method

1. All 47 public ZTF images from two days before to 40 days after the
   trigger (field 515, CCD 5, quadrant 1; 30 s exposures) go through
   `grb_afterglow.reduce_frame` with the deep ZTF reference image of each band
   as the template: difference imaging and empirical-PSF photometry, with zero
   points from Pan-STARRS1. Two frames have too few calibration stars.
2. ZTF and Pan-STARRS1 bandpasses differ, so each band's zero point has a
   colour term. It is measured once per band, as the median of the per-frame
   fits over all frames, and applied at the afterglow's own colour (computed
   from its spectral index): g −0.081 ± 0.020 and r +0.068 ± 0.025 per
   magnitude of g − r, i +0.151 ± 0.032 per magnitude of r − i.
3. The light curve is corrected for the Galactic extinction (Schlafly &
   Finkbeiner 2011, CCM89 curve) and a smoothly broken power law with a common
   spectral index is fitted jointly to g, r and i before 7 days, where the late
   flares begin. The posterior is integrated on a grid of break time and early
   decay index (`grb_afterglow.mcmc.grid_posterior`; see below for why not
   MCMC).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/lightcurve_dark.png">
  <img src="figures/lightcurve_light.png" alt="GRB 210204A g, r, i light curve from ZTF with a jet-break fit" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/subtraction_dark.png">
  <img src="figures/subtraction_light.png" alt="ZTF discovery image of the afterglow 0.6 hours after the burst, minus the reference image" width="100%">
</picture>

The afterglow is detected above 3σ in 12 frames, from 0.6 hours to 5 days.
The discovery image gives r = 17.15 ± 0.03 (502 ± 15 µJy), in agreement with
the r = 17.11 of the discovery circular.

| | This pipeline (ZTF only, posterior median and 68 %) | Kumar et al. (2022) |
|---|---:|---:|
| Preferred model | broken power law (ΔBIC = 83 over a single one) | jet break |
| Early decay, alpha1 | 0.50 +0.05 −1.46 | ≈ 0.33 |
| Break time | 0.95 ± 0.88 d (two modes, below) | 0.37 ± 0.30 d |
| Late decay, alpha2 | 1.28 +0.47 −0.16 | 1.18 ± 0.03 |
| Spectral index, beta | 0.93 ± 0.19 | ≈ 0.75 (optical to X-ray) |
| Closure relation after the break | ISM, ν_m < ν < ν_c, p = 2.86 ± 0.38 (0.3σ) | ISM, p ≈ 2.5 |

The survey saw the afterglow once, at 0.6 hours, and then not until
0.94 days. That leaves two ways to place the break, and the posterior has a
mode for each:

- **A break between 1 and 2 days**, with an early decay of 0.50 set by the
  first point. This is the least-squares solution: t_b = 1.46 ± 0.45 d,
  alpha2 = 1.51 ± 0.27.
- **A break right after the first point.** Everything after it is then a
  single power law with alpha ≈ 1.2, and the first point says nothing about
  the early decay, which can take any value its prior allows. This mode holds
  40 % of the probability (break before 0.5 d). How much depends on the prior
  range of alpha1, uniform from −3 to 6 here: the data alone cannot choose
  between the two.

Kumar et al.'s break at 0.37 d, from denser data, lies between the modes. The
late decay, spectral index and electron index agree with theirs within 1σ. Their light curve combines ZTF with several other telescopes and
constrains the late decay far better. The spectral index from ZTF's three
bands and the closure relations independently point to the same
constant-density medium and electron index that Kumar et al. infer from the
optical and X-ray data.

**Why a grid, not MCMC.** An ensemble MCMC sampler mixes badly on this
posterior: the early-break mode is a long, thin funnel. Its autocorrelation
time grew with the length of the chain, from about 200 steps in a 6,000-step
chain to about 2,000 in a 30,000-step one. Version 0.4 of this case study
quoted t_b = 1.3 +0.7 −0.8 d from such an unconverged chain. The posterior is
therefore integrated on a grid instead:

- the grid covers break time and early decay index;
- amplitude, late decay and spectral index are integrated as Gaussians at
  each grid point;
- the result is unchanged on a grid twice as fine.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/posterior_dark.png">
  <img src="figures/posterior_light.png" alt="Corner plot of the posterior of the jet-break fit" width="70%">
</picture>
</p>

**The late flares.** Kumar et al. report optical flares at 8.8 and 12.7 days.
The five ZTF points between 8 and 14 days lie on average 1.8 ± 1.9 µJy above
the extrapolated power law: consistent with flares of the published size, but
ZTF's 30 s exposures are not deep enough to detect them on their own.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/filmstrip_dark.png">
  <img src="figures/filmstrip_light.png" alt="ZTF r-band cutouts of the afterglow fading over a month" width="100%">
</picture>

Per-epoch measurements are in [`results/ztf_photometry.csv`](results/ztf_photometry.csv)
and every number above in [`results/summary.json`](results/summary.json).

## A physical model of the jet

Kumar et al. fitted a top-hat jet with `afterglowpy` to radio, X-ray and
optical data. `physical_model.py` fits the same model to the ZTF light curve
alone, before 8 days:

```bash
pip install -e ".[models]"
python case_studies/grb210204a/physical_model.py   # ~20 minutes, mostly MCMC
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/physical_model_dark.png">
  <img src="figures/physical_model_light.png" alt="GRB 210204A ZTF light curve with a top-hat jet model fitted to it and the Kumar et al. model for comparison" width="100%">
</picture>

**Their model against our photometry.** Kumar et al.'s posterior medians,
evaluated at the ZTF epochs, give chi² = 58 for 16 points. The shape matches,
but the ZTF fluxes are 16 % (0.17 mag) brighter. With that single scale
factor, chi² = 16 for 16 points. Their paper does not state a cosmology; the
distance here is from Planck 2018.

**What ZTF alone measures.** The density, epsilon_B and viewing angle are held
at Kumar et al.'s values, epsilon_e = 0.1 and xi_N = 1 as they had them, and
the energy, opening angle and electron index are fitted. The posterior medians
and 68 % intervals are:

| | ZTF only, this pipeline | Kumar et al. (2022) |
|---|---:|---:|
| Electron index, p | 2.16 +0.02 −0.01 | 2.18 ± 0.03 |
| Jet half-opening angle, θ_core (rad) | 0.022 +0.002 −0.004 | 0.024 +0.003 −0.002 |
| Energy, log10 E0 (erg) | 54.16 +0.02 −0.02 | 54.06 ± 0.03 |
| chi²/dof | 10.2/13 | |

The electron index and the opening angle agree. The energy is 0.1 dex
higher, which absorbs the 16 % offset in flux.

The posterior has a tail toward narrower jets. Its autocorrelation time
(about 140 steps) is longer than the emcee guideline allows for this chain
(16 walkers × 8,000 steps). An independent shorter run gave the same intervals
to within 0.01.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/physical_posterior_dark.png">
  <img src="figures/physical_posterior_light.png" alt="Corner plot of energy, opening angle and electron index from ZTF, with Kumar et al.'s values marked" width="70%">
</picture>
</p>

**What it cannot measure.** With the density fixed and epsilon_B free as well,
two densities 50 times apart fit equally well:

| Density n0 (cm⁻³), fixed | log10 E0 | θ_core | p | log10 epsilon_B | chi²/dof |
|---|---:|---:|---:|---:|---:|
| 10⁻⁵·⁶⁷ (Kumar et al.) | 55.1 ± 0.3 | 0.023 ± 0.005 | 2.43 ± 0.22 | −2.8 ± 0.6 | 7.7/12 |
| 10⁻⁴ | 53.3 ± 0.2 | 0.049 ± 0.007 | 2.40 ± 0.15 | −1.0 ± 0.5 | 7.7/12 |
| 10⁻² | | | | | 210/12 |

The energies differ by a factor of 50 and the opening angles by 2, and the
electron index is now 2.4. A light curve in three optical bands measures two
slopes and a break, and those leave energy, density, epsilon_B and p trading
off against each other. Kumar et al. break the degeneracy with their X-ray and
radio data. Only the dense case, 10⁻² cm⁻³, fails to fit at all.

## Limitations

- **One epoch before the break.** The ZTF survey visited the field 0.6 hours
  after the burst and then not until 0.94 days, so the break time and the early
  slope rest on a single point and their posterior depends on the priors.
- **Survey depth.** After about 5 days the afterglow falls below the depth of
  30 s ZTF exposures; around the full moon at 21 to 25 days the frames are almost
  two magnitudes shallower and have few calibration stars.
- **No host galaxy** is detected at the position, so the reference images
  contain only field stars; the subtraction is still needed for them and for
  the zero-point consistency between epochs.
