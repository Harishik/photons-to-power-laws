# Science background: GRB optical afterglows

This note gives just enough astrophysics to understand *why* the pipeline does
what it does. It is written for someone comfortable with undergraduate physics
but new to gamma-ray bursts.

## What is a GRB afterglow?

A gamma-ray burst (GRB) is a brief, extraordinarily luminous flash of
gamma-rays from a relativistic jet launched by a collapsing massive star or a
neutron-star merger. The **afterglow** comes from the jet ploughing into the
surrounding medium and decelerating. That external shock accelerates electrons
into a power-law energy distribution, `N(E) ∝ E^-p`, and they radiate
**synchrotron emission** from radio to X-rays. Optical telescopes follow it
from minutes to weeks after the trigger as it fades.

## Why power laws?

A decelerating blast wave has no characteristic scale, so the synchrotron
emission is a broken power law in both frequency and time:

    F_nu(t) ∝ t^(-alpha) nu^(-beta)

- **alpha**, the temporal index, says how fast the source fades. Typical
  optical values are roughly 0.7 to 1.5 before any jet break.
- **beta**, the spectral index, says how the brightness changes with
  frequency. The pipeline measures it by fitting several bands at once.

## Closure relations

alpha and beta are not independent. Both are set by the electron index `p`,
and the relation between them depends on

- the **circumburst medium**: a constant-density interstellar medium (ISM) or
  the `rho ∝ r^-2` wind of the progenitor star;
- the **spectral regime**: whether the observed band lies below or above the
  cooling frequency `nu_c`;
- whether the **jet break** has happened (below).

For slow cooling and `p > 2` (`grb_afterglow.closure`):

| Regime | beta | alpha | alpha(beta) |
|---|---|---|---|
| ISM, nu_m < nu < nu_c | (p-1)/2 | 3(p-1)/4 | 3beta/2 |
| ISM, nu > nu_c | p/2 | (3p-2)/4 | (3beta-1)/2 |
| Wind, nu_m < nu < nu_c | (p-1)/2 | (3p-1)/4 | (3beta+1)/2 |
| Wind, nu > nu_c | p/2 | (3p-2)/4 | (3beta-1)/2 |
| After jet break, nu_m < nu < nu_c | (p-1)/2 | p | 2beta+1 |
| After jet break, nu > nu_c | p/2 | p | 2beta |

Measuring alpha and beta and asking which relation they satisfy is how an
optical light curve constrains the physics of the explosion. Note that above
the cooling frequency the ISM and wind cases are degenerate: optical data alone
cannot separate them.

## Jet breaks

The outflow is a narrow jet, not a sphere. Early on, relativistic beaming hides
the jet's edges, so the afterglow behaves as if it were spherical. As the blast
wave decelerates, the beaming cone widens until it exceeds the jet's opening
angle. The light curve then steepens to `alpha ≈ p`, at the same time in every
band. This achromatic **jet break** is why the pipeline fits a smoothly broken
power law with a shared break time across bands, and the break time encodes the
jet opening angle and the true, beaming-corrected energy.

## From pixels to those numbers

1. **Calibration** removes instrument signatures, `(raw − bias) / flat`, so a
   pixel value reflects light from the sky rather than the detector.
2. **Detection and seeing.** Stars are found with DAOFIND and the seeing is
   measured from isolated stars. Apertures scale with the seeing.
3. **Photometry.** Either aperture sums minus a local sky, with a
   CCD-equation error that includes source shot noise, sky noise, and the
   uncertainty of the sky estimate itself (sky annuli exclude other sources),
   or PSF fitting (below).
4. **Photometric calibration.** Detected stars are matched to a reference
   catalog (Pan-STARRS1 for real data) through the WCS, and the zero point is
   solved per frame because the atmosphere changes between exposures. When the
   instrument's bandpass differs from the catalog's, a colour term relates them
   (below).
5. **Image subtraction and forced photometry at the burst position.** A deep
   template is subtracted (below) and the afterglow is measured on the
   difference image, even after it has faded below the detection threshold.
   Those measurements can be zero or negative, which is why light curves are
   kept in flux density.
6. **Fitting.** Flux-space weighted least squares for single and broken power
   laws, compared with the Bayesian information criterion, then the posterior
   sampled with MCMC and the slopes tested against the closure relations.

Every step propagates uncertainty, because the scientific statement is not
`alpha = 1.4` but `alpha = 1.40 ± 0.02`. Only the version with an error bar can
be compared with theory.

## Host galaxies and image subtraction

A faded afterglow sits on top of its host galaxy and, in crowded fields, on
the wings of neighbouring stars. An aperture measures all of that light
together, so once the afterglow is fainter than the host the light curve
flattens and the late decay index comes out far too shallow. The simulated
demo shows the size of the effect: without subtraction a host galaxy of
r = 21 turns an injected post-break decay of alpha = 2.2 into 1.3.

**Image subtraction** removes everything that did not change. A deep
reference image of the field (the *template*, taken before the burst or long
after it) is resampled onto the science image's pixel grid through the WCS,
convolved with a kernel that matches its PSF and photometric scale to the
science image, and subtracted. Static stars and the host cancel; only the
transient remains in the *difference image*. The kernel is fitted by linear
least squares over the whole image (Alard & Lupton 1998; Bramich 2008), or
the subtraction is done optimally in Fourier space (Zackay, Ofek & Gal-Yam
2016).

This package uses the **delta-function kernel of Bramich (2008)**, in which
every kernel pixel is a free parameter, so no assumption about either PSF is
needed. The sharper image is the one convolved (a kernel can only broaden).
Fitting the kernel by least squares when the convolved image is itself noisy
biases it towards zero, the classic errors-in-variables problem, which would
leave a residual of every static source, the host included; the expected noise
term is subtracted from the normal equations (Fuller 1987), a Laplacian
penalty (Becker et al. 2012) keeps the solution stable, outliers such as the
transient itself are clipped, and the photometric scale is refitted on bright
pixels. In a bake-off against an Alard-Lupton Gaussian basis and ZOGY on
synthetic pairs with known transients and on the real ZTF data
(`dev/bakeoff/`), Bramich and Alard-Lupton performed equivalently: over
synthetic pairs in four seeing configurations (science sharper, template
sharper, matched, undersampled) Bramich's flux pulls have mean +0.02 and
spread 1.13, the noise in its difference images is within 1 % of the ideal,
and on the real ZTF epochs the fluxes agree with the Liverpool Telescope light
curve (chi²/pt = 0.2). Bramich was kept because its kernel needs no choice of
basis; ZOGY, whose optimality assumes known PSFs, was noisier on the real
data with analytic Moffat PSFs.

The flux scale needs care when the *template* has the worse seeing, so that
the science image is the one convolved and the difference image carries the
template's PSF. The zero point refers to the science PSF, so the pipeline
transfers it with the zero-point stars themselves: measured on the science
image convolved with the fitted kernel, they have exactly the difference
image's PSF. (For PSF photometry the science PSF seen through the kernel is
used directly.) An apparently equivalent shortcut, an aperture correction from
the curve of growth of stars in the template, fails in crowded fields: on the
deep ZTF reference image of GRB 221009A, neighbouring stars inside the large
"total" apertures made a point source look as if it had 37 % of its light
within 1.2 FWHM instead of 80 %, and the fluxes came out twice too high.

Resampling and convolution correlate the noise of neighbouring pixels, so the
per-pixel variance underestimates the noise of an aperture sum on a difference
image. The pipeline therefore measures the noise empirically, by placing many
apertures (or PSF fits) on empty parts of the difference image.

## PSF photometry

An aperture weights every pixel equally, but the outer pixels of a faint
source carry mostly sky noise. Fitting a model of the point-spread function
(PSF) weights pixels by how much of the source's light they hold, which gives
the best attainable signal-to-noise for faint sources, and fitting several
overlapping sources at once separates blends that an aperture would add
together. Because the fitted PSF is normalised, the result is a total flux
with no aperture correction.

The PSF is **empirical**, built from the stars in each frame (in the spirit of
DAOPHOT, Stetson 1987, and Anderson & King 2000): a pixel-integrated elliptical
Moffat profile carries the core and the wings to infinity, so fluxes are total
fluxes, and a residual map on a grid oversampled four times per pixel absorbs
whatever the analytic profile misses; regularisation pulls it to zero where the
data are uninformative, so with few PSF stars the model falls back to the
Moffat. All detections, plus sources found in the fit residuals, are fitted
simultaneously with a smooth background, so neighbours are deblended rather
than added to the target, and the errors include a PSF-model term measured on
the PSF stars. In the bake-off this gave a flux bias below 0.1 %, pulls with a
robust spread of 1.02 (1.01 in crowded fields), and a median error of 9 % on
faint stars against 30 % for apertures; on the ZTF frames the zero-point
scatter drops from 0.075 to 0.055 mag. A purely analytic Moffat PSF matched
it on synthetic stars, which are Moffat profiles, but only reached 0.073 mag
on the ZTF frames: real PSFs need the empirical part. On a difference image the transient is
fitted with the same PSF and a local background, and its error comes from the
scatter of identical fits at blank positions.

## Colour terms

A survey's filters never match the catalog's exactly, so stars of different
colours need slightly different zero points: `m_cat = m_inst + ZP + C·colour`.
The pipeline fits `C` and evaluates the zero point at the colour of the burst,
not at the average colour of the calibration stars. Afterglows can be far
redder than those stars: a `β = 0.4` power law behind the Galactic dust toward
GRB 221009A has `g − r = 1.3`. A colour term is a property of the instrument,
so it is best measured once from many frames (the case studies use the median
of the per-frame fits) and then fixed: fitted to the few dozen stars of a
single frame its uncertainty, extrapolated to a red burst, can exceed the
correction itself.

## Extinction

Dust in the Milky Way and in the host galaxy dims and reddens the afterglow. A
temporal index is unaffected (extinction is constant in time), but a spectral
index measured from optical colours is biased unless extinction is corrected.
The Milky Way curve is that of Cardelli, Clayton & Mathis (1989) with
`R_V = 3.1`, scaled by the colour excess `E(B − V)` of the Schlafly &
Finkbeiner (2011) dust map. GRB host galaxies usually have dust like the Small
Magellanic Cloud's, without the 2175 Å bump, so host extinction uses the SMC
curve of Pei (1992) at the rest-frame wavelength; with several bands the fit
can solve for the host's `A_V` alongside `β`.

Behind a lot of dust two details matter. The correction should be averaged
over each filter's transmission curve rather than taken at one wavelength,
because the reddened source puts most of its photons in the red side of a
broad band. And the wavelength must be the actual filter's: for GRB 221009A
(`E(B − V) = 1.32`, `A_V ≈ 4.1`), using SDSS rather than Liverpool IO:O pivot
wavelengths moves the g-band pivot by 13 nm and overcorrects g by 0.18 mag
(i by 0.09), which lowers `β` by 0.08; evaluating the extinction at the pivot
instead of averaging over the filter lowers it by another 0.08. Even done
properly, `β` from optical bands alone is dominated by the uncertainty of the
dust: ±0.06 in `E(B − V)` moves `β` from Liverpool's griz by ±0.18.

## Several instruments, one spectrum

Photometry from different telescopes raises two problems when it goes into one
fit. The first is bandpasses. Every instrument's "r" is a different band, and
behind heavy dust the difference is large. The package therefore does not
deredden each band by one number. It applies the dust to the model spectrum
and integrates that through each instrument's own transmission curve:

```text
F_band = ∫ T(λ) F_ν(λ) 10^(−0.4 A(λ)) dλ/λ / ∫ T(λ) dλ/λ
```

This is synthetic photometry of the model. The curves come from the SVO
Filter Profile Service (Rodrigo et al. 2012, 2020, 2024), converted to photon
counting where SVO gives an energy response.

The second problem is zero points. Each data set is calibrated against a
catalogue, Pan-STARRS1 in the optical and 2MASS in the near infrared, usually
without colour terms. The fit gives each instrument, or each band, a
zero-point offset with a Gaussian prior as wide as the expected calibration
error, anchored to a reference band.

Without colour terms, catalogue-calibrated magnitudes are in the instrument's
own system, up to a grey offset set by the colours of the calibration stars.
They are not in the catalogue's system. For a source as red as GRB 221009A the
distinction is not academic: read either way, Liverpool's griz give `β` = 0.46
or 0.91. Near-infrared bands, where the dust is weak, fix the slope over a
factor of four in wavelength and shrink that ambiguity to 0.66 against 0.76.
They help less with the dust itself: E(B − V) ± 0.06 still moves `β` by ±0.13,
compared with ±0.18 for griz alone.

## Bayesian posteriors

Least squares returns a best fit and a covariance matrix, an adequate summary
only when the likelihood is close to Gaussian in the parameters. A jet break
often violates that: with few points on one side of the break, its time is
poorly constrained, asymmetric, and correlated with the decay indices. The
pipeline therefore also samples the posterior with the affine-invariant
ensemble sampler of Goodman & Weare (2010) (`emcee`, Foreman-Mackey et al.
2013), using the same flux-space likelihood and flat priors (in the logarithm
for the amplitude and the break time). Derived quantities such as `p` follow
sample by sample. For the synthetic demo, where the data constrain everything
well, posterior and least squares agree.

Samplers have limits. For GRB 210204A, with one ZTF epoch before the break,
the posterior has two modes:

- a break between 1 and 2 days;
- a break right after that epoch, where the early decay index is
  unconstrained and the posterior narrows into a long funnel.

The ensemble sampler's autocorrelation time keeps growing with the chain
length, the sign that it has not converged. There, `grid_posterior`
integrates the two difficult parameters on a grid, which is exact for them
whatever the shape of the posterior. The well-constrained rest are integrated
in the Laplace approximation, as Gaussians around their best fit at each grid
point, and samples are drawn from the result.

## References

- Sari, Piran & Narayan 1998, ApJ 497, L17: spectra and light curves of afterglows.
- Sari, Piran & Halpern 1999, ApJ 519, L17; Rhoads 1999, ApJ 525, 737: jet breaks.
- Chevalier & Li 2000, ApJ 536, 195: afterglows in a stellar wind.
- Zhang & Mészáros 2004, IJMPA 19, 2385; Racusin et al. 2009, ApJ 698, 43:
  closure-relation tables.
- Stetson 1987, PASP 99, 191: DAOPHOT, DAOFIND, and PSF-fitting photometry.
- Anderson & King 2000, PASP 112, 1360: empirical, oversampled PSFs.
- Alard & Lupton 1998, ApJ 503, 325; Bramich 2008, MNRAS 386, L77;
  Becker et al. 2012, MNRAS 425, 1341; Zackay, Ofek & Gal-Yam 2016, ApJ 830, 27:
  image subtraction.
- Fuller 1987, *Measurement Error Models* (Wiley): errors-in-variables regression.
- Tonry et al. 2012, ApJ 750, 99: the Pan-STARRS1 photometric system.
- Cardelli, Clayton & Mathis 1989, ApJ 345, 245: the Milky Way extinction curve.
- Pei 1992, ApJ 395, 130: extinction curves of the Milky Way and the Magellanic Clouds.
- Schlafly & Finkbeiner 2011, ApJ 737, 103: recalibrated Galactic dust map.
- Goodman & Weare 2010, CAMCoS 5, 65; Foreman-Mackey et al. 2013, PASP 125, 306:
  affine-invariant ensemble sampling and `emcee`.
- O'Connor et al. 2023, Science Advances 9, eadi1405; Shrestha et al. 2023,
  arXiv:2302.03829: the spectral index of, and the extinction toward, GRB 221009A.
- Kumar et al. 2022, MNRAS, arXiv:2204.07587: the afterglow and late flares of GRB 210204A.
- Bellm et al. 2019, PASP 131, 018002; Masci et al. 2019, PASP 131, 018003:
  the Zwicky Transient Facility and its data system.
- Laskar et al. 2023, ApJL 946, L23, [doi:10.3847/2041-8213/acbfad](https://doi.org/10.3847/2041-8213/acbfad):
  the radio to GeV afterglow of GRB 221009A.
- Fulton et al. 2023, arXiv:2301.11170: a possible supernova in GRB 221009A at about 20 days.
- Rodrigo, Solano & Bayo 2012, IVOA Working Draft; Rodrigo & Solano 2020, XIV.0 Scientific
  Meeting of the SEA; Rodrigo et al. 2024, A&A 689, A93: the SVO Filter Profile Service.
