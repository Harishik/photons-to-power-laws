# Science background: GRB optical afterglows

This note gives just enough astrophysics to understand *why* the pipeline does
what it does. It is written for someone comfortable with undergraduate physics
but new to gamma-ray bursts.

## What is a GRB afterglow?

A Gamma-Ray Burst (GRB) is a brief, extraordinarily luminous flash of
gamma-rays. The leading picture for the burst itself is internal shocks within a
relativistic jet. For the **afterglow**, it is the external shock, the jet
ploughing into the surrounding medium and decelerating. That shock accelerates
electrons, which radiate **synchrotron emission** across radio, optical, and
X-ray bands. The afterglow is what we follow up from the ground in the optical,
sometimes within minutes of the trigger and for days to weeks afterward.

## Why a power law?

Synchrotron emission from a decelerating blast wave produces, to good
approximation, emission that is a **broken power law in both frequency and
time**:

    F_nu(t) ∝ t^(-α) ν^(-β)

- **α** is the temporal decay index, how fast the source fades. Typical
  pre-break optical values are roughly 0.7 to 1.5.
- **β** is the spectral index, how the brightness changes with frequency.

The temporal and spectral slopes are not independent. Standard afterglow theory
predicts specific relationships between them, the **closure relations**, that
depend on the energy distribution of the shocked electrons, the density profile
of the surrounding medium, and which part of the synchrotron spectrum the
optical band falls in. Measuring α and β and checking which closure relation
they satisfy is how a light curve constrains the physics of the explosion.

## Jet breaks

The emitting material is not isotropic. It is a **jet**. Early on, relativistic
beaming hides the jet's edges, so the source behaves as if emission were
spherical. As the blast wave decelerates, the beaming cone widens until it
exceeds the jet's opening angle. At that point the observer sees the edge and
the light curve **steepens**, an achromatic break called a **jet break**. This
is why the pipeline includes a broken power-law model. The late decay index
`α₂` is steeper than the early `α₁`, and the break time `t_break` encodes the
jet opening angle and the true, beaming-corrected energy of the burst.

## From pixels to those numbers

Getting α from images is a chain of careful measurements:

1. **Calibration** removes instrument signatures so a pixel value reflects light
   from the sky, not the detector. `(raw − bias) / flat`.
2. **Detection + photometry** turn the calibrated image into counts for each
   source, with an error bar from the CCD-equation noise budget.
3. **Photometric calibration** ties those counts to a physical magnitude scale
   by comparison with catalog stars in the same frame.
4. **Forced photometry at the GRB position** measures the afterglow even once it
   has faded below the blind-detection threshold.
5. **Light-curve fitting** converts magnitudes to flux density and fits
   `F ∝ t^(-α)`, optionally with a jet break, returning α and its uncertainty.

Every step preserves and propagates uncertainty, because the final scientific
statement is not only `α = 1.1`, but `α = 1.10 ± 0.03`. Only the version with the
error bar can be compared against theory.

## Further reading

These are good entry points into the primary literature:

- Sari, Piran & Narayan (1998), "Spectra and Light Curves of GRB Afterglows".
- Rhoads (1999) and Sari, Piran & Halpern (1999), jet breaks.
- Reviews of GRB afterglows for the broader observational picture.

The pipeline's reduction sequence follows what real GRB follow-up papers do:
bias/flat calibration, source extraction, calibration against Pan-STARRS or SDSS
field stars, forced photometry at the burst position, and power-law fitting.
