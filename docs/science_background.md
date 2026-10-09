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
3. **Photometry.** Aperture sums minus a local sky, with a CCD-equation
   error that includes source shot noise, sky noise, and the uncertainty of
   the sky estimate itself. Sky annuli exclude other sources.
4. **Photometric calibration.** Detected stars are matched to a reference
   catalog (Pan-STARRS1 for real data) through the WCS, and the zero point is
   solved per frame because the atmosphere changes between exposures.
5. **Forced photometry at the burst position** measures the afterglow even
   after it has faded below the detection threshold. Those measurements can be
   zero or negative, which is why light curves are kept in flux density.
6. **Fitting.** Flux-space weighted least squares for single and broken power
   laws, compared with the Bayesian information criterion, then tested against
   the closure relations.

Every step propagates uncertainty, because the scientific statement is not
`alpha = 1.4` but `alpha = 1.40 ± 0.02`. Only the version with an error bar can
be compared with theory.

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
- Stetson 1987, PASP 99, 191: DAOPHOT and the DAOFIND algorithm.
- Tonry et al. 2012, ApJ 750, 99: the Pan-STARRS1 photometric system.
- Bellm et al. 2019, PASP 131, 018002; Masci et al. 2019, PASP 131, 018003:
  the Zwicky Transient Facility and its data system.
- Laskar et al. 2023, ApJL, [doi:10.3847/2041-8213/acbfad](https://doi.org/10.3847/2041-8213/acbfad):
  the radio to GeV afterglow of GRB 221009A.
