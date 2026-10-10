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
   solved per frame because the atmosphere changes between exposures.
5. **Image subtraction and forced photometry at the burst position.** A deep
   template is subtracted (below) and the afterglow is measured on the
   difference image, even after it has faded below the detection threshold.
   Those measurements can be zero or negative, which is why light curves are
   kept in flux density.
6. **Fitting.** Flux-space weighted least squares for single and broken power
   laws, compared with the Bayesian information criterion, then tested against
   the closure relations.

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
pixels. This method was chosen over an Alard-Lupton Gaussian basis and ZOGY in
a bake-off on synthetic pairs with known transients and on the real ZTF data
(`dev/bakeoff/`): over synthetic pairs in four seeing configurations (science
sharper, template sharper, matched, undersampled) its flux pulls have mean
+0.02 and spread 1.13, the noise in the difference image is within 1 % of the
ideal, and on the real ZTF epochs the fluxes agree with the Liverpool
Telescope light curve (chi²/pt = 0.2).

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
scatter drops from 0.075 to 0.055 mag. On a difference image the transient is
fitted with the same PSF and a local background, and its error comes from the
scatter of identical fits at blank positions.

## Extinction

Dust in the Milky Way and in the host galaxy dims and reddens the afterglow. A
temporal index is unaffected (extinction is constant in time), but a spectral
index measured from optical colours is biased unless extinction is corrected.
GRB 221009A lies close to the Galactic plane behind several magnitudes of
extinction, which is why the case study measures alpha only.

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
- Bellm et al. 2019, PASP 131, 018002; Masci et al. 2019, PASP 131, 018003:
  the Zwicky Transient Facility and its data system.
- Laskar et al. 2023, ApJL, [doi:10.3847/2041-8213/acbfad](https://doi.org/10.3847/2041-8213/acbfad):
  the radio to GeV afterglow of GRB 221009A.
