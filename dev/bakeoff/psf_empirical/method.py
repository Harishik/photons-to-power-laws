"""Bake-off entry for the shipped PSF photometry, grb_afterglow.psf.psf_photometry.

An empirical PSF (pixel-integrated elliptical Moffat plus an oversampled, regularised residual)
built from the image and fitted to all sources at once. The prototype scored here was moved into
the package; this adapter keeps the harness testing the shipped code.
"""

from grb_afterglow.psf import psf_photometry as _psf_photometry


def psf_photometry(image, positions, *, fwhm, source_xy, gain=None, var=None):
    r = _psf_photometry(image, positions, source_xy=source_xy, fwhm=fwhm, var=var, gain=gain)
    return {"flux": r.flux, "flux_err": r.flux_err}
