# Harness changes

- **09:10** make_pair: the host galaxy is now convolved with each image's PSF. Before this fix the host was
  injected without PSF convolution in both images, so any correct PSF-matching kernel produced a structured
  host residual at the transient position: small-aperture pulls were biased (+ when the reference is sharper,
  - when the science image is sharper) even though large-aperture fluxes were fine. Results produced before
  09:10 are not comparable; re-run. Do not tune methods to the old behaviour.
- **19:30** eval_subtraction, real ZTF data: the burst flux on the difference image is now calibrated with the
  zero-point stars themselves, measured in the measurement aperture on the science image convolved with the
  method's kernel (the PSF of the difference image), relative to the zero-point aperture. Before, the
  correction came from the curve of growth of the ZTF reference image, whose 4-FWHM "total" apertures collect
  light from neighbouring stars in this deep Galactic-plane coadd (frac = 0.375 at 1.2 FWHM, against 0.80 from
  a median-stacked PSF of the same stars), so every flux measured with the reference PSF came out ~2x too
  high. Methods that return no kernel keep the old (unreliable) curve-of-growth path.
