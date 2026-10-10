"""Bake-off entry for the shipped image subtraction, grb_afterglow.subtraction.subtract_images.

The Bramich (2008) delta-basis kernel with errors-in-variables correction, Laplacian regularisation,
sigma clipping and a bright-pixel scale refit. The prototype scored here was moved into the package
unchanged (identical harness results); this adapter keeps the harness testing the shipped code.
"""

from grb_afterglow.subtraction import subtract_images


def subtract(sci, sci_wcs, ref, ref_wcs, *, sci_var=None, ref_var=None, sci_fwhm=None, ref_fwhm=None):
    d = subtract_images(
        sci,
        sci_wcs,
        ref,
        ref_wcs,
        science_fwhm=sci_fwhm,
        template_fwhm=ref_fwhm,
        science_var=sci_var,
        template_var=ref_var,
    )
    convolved = "science" if d.psf_from == "template" else "reference"
    return {"diff": d.data, "var": d.variance, "fwhm": d.fwhm, "kernel": d.kernel, "convolved": convolved}
