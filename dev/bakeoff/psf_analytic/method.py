"""PSF photometry with an analytic, pixel-integrated elliptical Moffat model.

Two stages:

1. **PSF model.** An elliptical Moffat profile with free core width, ellipticity
   and power-law index ``beta`` is fitted jointly to bright, reasonably isolated
   stars picked from the detection list. The profile is integrated over each
   pixel with Gauss-Legendre quadrature, so it stays accurate for undersampled
   images (FWHM ~ 1.5 px). Each star has its own sub-pixel centre; its flux, a
   local sky level and the fluxes of detected neighbours inside its stamp are
   solved linearly (variable projection). The covariance of the shape
   parameters is kept so that PSF-model uncertainty can be propagated.
2. **Forced photometry.** At each requested position the flux is the solution of
   a weighted *linear* least-squares problem over a square box: the target, the
   detected neighbours whose centres fall in the box and a planar sky are fitted
   simultaneously; neighbours whose wings reach into the box are subtracted with
   fluxes from a first pass over all sources. Pixels are weighted by their
   variance (the supplied map, or the sky RMS plus model-based shot noise). The
   flux error combines the least-squares covariance with the PSF-model error,
   obtained by re-solving the box with the PSF shape perturbed along the
   principal axes of its covariance (this matters for bright stars and blends).

The Moffat profile is normalised analytically to unit integral over the plane,
so fitted fluxes are *total* fluxes and need no aperture correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from grb_afterglow.detection import estimate_background

ISO = 1.5
__all__ = ["MoffatPSF", "fit_moffat_psf", "forced_psf_photometry", "psf_photometry"]


@dataclass(frozen=True)
class MoffatPSF:
    """Elliptical Moffat point-spread function with unit total flux.

    The profile is ``(beta - 1) / (pi alpha^2) * (1 + u^2)^(-beta)`` with
    ``u^2 = r^T M r / alpha^2``, where ``M = [[1+e1, e2], [e2, 1-e1]] / q`` and
    ``q = sqrt(1 - e1^2 - e2^2)`` so that ``det M = 1`` and ``alpha`` is the
    geometric-mean core radius.

    Attributes
    ----------
    alpha : float
        Core radius in pixels.
    beta : float
        Power-law index (> 1); smaller values mean broader wings.
    e1, e2 : float
        Ellipticity components (``e1^2 + e2^2 < 1``).
    order : int
        Gauss-Legendre nodes per axis used to integrate the profile over a pixel.
    """

    alpha: float
    beta: float
    e1: float = 0.0
    e2: float = 0.0
    order: int = 3

    @classmethod
    def from_params(cls, theta, order=3) -> MoffatPSF:
        """Build from ``(log FWHM, log(beta - 1), e1, e2)``, the fitting parametrisation."""
        beta = 1.0 + float(np.exp(theta[1]))
        alpha = float(np.exp(theta[0])) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        return cls(alpha, beta, float(theta[2]), float(theta[3]), order)

    @property
    def params(self) -> np.ndarray:
        """``(log FWHM, log(beta - 1), e1, e2)`` (FWHM is far less correlated with beta than alpha)."""
        return np.array([np.log(self.fwhm), np.log(self.beta - 1.0), self.e1, self.e2])

    @property
    def fwhm(self) -> float:
        """Geometric-mean full width at half maximum in pixels."""
        return float(2.0 * self.alpha * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0))

    def profile(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Surface brightness (per unit area) at offsets ``(dx, dy)`` from the centre."""
        q = np.sqrt(max(1.0 - self.e1**2 - self.e2**2, 1e-6))
        u2 = ((1.0 + self.e1) * dx**2 + 2.0 * self.e2 * dx * dy + (1.0 - self.e1) * dy**2) / (q * self.alpha**2)
        return (self.beta - 1.0) / (np.pi * self.alpha**2) * np.exp(-self.beta * np.log1p(u2))

    def pixel_values(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Unit-flux PSF integrated over pixels whose centres are offset ``(dx, dy)`` from the source."""
        node, weight = np.polynomial.legendre.leggauss(self.order)
        ox, oy = (g.ravel() / 2.0 for g in np.meshgrid(node, node))
        w = np.outer(weight, weight).ravel() / 4.0
        return self.profile(np.asarray(dx)[..., None] + ox, np.asarray(dy)[..., None] + oy) @ w

    def design(self, px: np.ndarray, py: np.ndarray, xs, ys) -> np.ndarray:
        """``(len(px), len(xs))`` matrix: column ``k`` is a unit-flux source at ``(xs[k], ys[k])``.

        Pixel ``(i, j)`` covers ``[j - 0.5, j + 0.5] x [i - 0.5, i + 0.5]``.
        """
        return self.pixel_values(
            px[:, None] - np.asarray(xs, float)[None, :], py[:, None] - np.asarray(ys, float)[None, :]
        )


def _gl_order(fwhm: float) -> int:
    """Gauss-Legendre order per axis: higher for sharper (undersampled) PSFs."""
    return 4 if fwhm < 2.0 else 3


def _box(shape, x, y, half):
    """Slices and flattened pixel coordinates of a box clipped to the image (None if empty)."""
    ny, nx = shape
    ix, iy = int(np.rint(x)), int(np.rint(y))
    x0, x1, y0, y1 = max(ix - half, 0), min(ix + half + 1, nx), max(iy - half, 0), min(iy + half + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (slice(y0, y1), slice(x0, x1)), xx.ravel().astype(float), yy.ravel().astype(float)


def _weighted_lstsq(a, b, w):
    """Weighted linear least squares; returns coefficients and their covariance."""
    sw = np.sqrt(w)
    aw = a * sw[:, None]
    coef, *_ = np.linalg.lstsq(aw, b * sw, rcond=None)
    return coef, np.linalg.pinv(aw.T @ aw)


def _scene(positions, source_xy, radius):
    """Requested positions plus the detections that are not within ``radius`` of one."""
    if len(source_xy) == 0 or len(positions) == 0:
        return np.vstack([positions, source_xy])
    d, _ = cKDTree(positions).query(source_xy)
    return np.vstack([positions, source_xy[d > radius]])


def _peak_snr(image, sky, rms, xy):
    """Peak sky-subtracted value in the 3x3 pixels around each position, and its S/N."""
    ny, nx = image.shape
    ix = np.clip(np.rint(xy[:, 0]).astype(int), 1, nx - 2)
    iy = np.clip(np.rint(xy[:, 1]).astype(int), 1, ny - 2)
    peak = np.full(len(xy), -np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            v = image[iy + di, ix + dj] - sky[iy + di, ix + dj]
            peak = np.maximum(peak, np.nan_to_num(v, nan=-np.inf))
    return peak, peak / rms[iy, ix]


def fit_moffat_psf(image, source_xy, fwhm, *, sky, var, known_xy=None, max_stars=20, min_snr=15.0, stamp_fwhm=3.5):
    """Fit an elliptical Moffat PSF jointly to bright, isolated stars.

    Parameters
    ----------
    image : numpy.ndarray
        Image including sky.
    source_xy : numpy.ndarray
        ``(K, 2)`` detected source positions (PSF-star candidates).
    fwhm : float
        Approximate seeing FWHM in pixels (initial guess and stamp size).
    sky, var : numpy.ndarray
        Sky map (for star selection) and per-pixel variance map.
    known_xy : numpy.ndarray, optional
        All known source positions, used for the isolation test and modelled as
        point sources inside the stamps. Defaults to ``source_xy``.
    max_stars : int
        Maximum number of PSF stars.
    min_snr : float
        Minimum peak signal-to-noise ratio of a PSF star.
    stamp_fwhm : float
        Stamp half-size in units of ``fwhm``.

    Returns
    -------
    psf : MoffatPSF
        The fitted model; a circular ``beta = 3`` Moffat with the given FWHM if
        fewer than two usable stars are found.
    cov : numpy.ndarray or None
        ``4 x 4`` covariance of :attr:`MoffatPSF.params` (marginalised over the
        star positions, scaled by the reduced chi-square when it exceeds one),
        or None for the fallback model.
    """
    order = _gl_order(fwhm)
    beta0 = 3.0
    alpha0 = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta0) - 1.0))
    fallback = MoffatPSF(alpha0, beta0, order=order)
    xy = np.asarray(source_xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return fallback, None
    half = int(max(np.ceil(stamp_fwhm * fwhm), 4))
    ny, nx = image.shape
    known = xy if known_xy is None else np.asarray(known_xy, float).reshape(-1, 2)
    peak, snr = _peak_snr(image, sky, np.sqrt(var), xy)
    kpeak, _ = _peak_snr(image, sky, np.sqrt(var), known)
    tree = cKDTree(known)
    stars, rows, nbrs = [], [], []
    for k in np.argsort(-peak):
        x, y = xy[k]
        if snr[k] < min_snr or not (half <= x < nx - 1 - half and half <= y < ny - 1 - half):
            continue
        nb = np.array(tree.query_ball_point((x, y), np.hypot(half, half) + fwhm), dtype=int)
        d = np.hypot(*(known[nb] - xy[k]).T)
        nb, d = nb[d > 0.5], d[d > 0.5]  # drop the star itself
        # Reject stars with a close neighbour or a comparably bright one nearby.
        if np.any(d < 1.5 * fwhm) or np.any((d < 2.5 * fwhm) & (kpeak[nb] > 0.3 * peak[k])):
            continue
        sl, px, py = _box(image.shape, x, y, half)
        data, w = image[sl].ravel(), 1.0 / var[sl].ravel()
        ok = np.isfinite(data) & np.isfinite(w)
        stars.append((x, y))
        rows.append(np.column_stack([px[ok], py[ok], data[ok], np.sqrt(w[ok])]))
        nbrs.append(known[nb])
        if len(stars) >= max_stars:
            break
    if len(stars) < 2:
        return fallback, None

    def residuals(theta, pix, owner, bounds, sxy, pairs, ncol):
        """Weighted residuals of all stamps; per-stamp flux, sky and neighbours are profiled out."""
        psf = MoffatPSF.from_params(theta[:4], order)
        c = sxy + theta[4:].reshape(-1, 2)
        a = np.zeros((len(pix), int(ncol.max())))
        a[:, 0] = psf.pixel_values(pix[:, 0] - c[owner, 0], pix[:, 1] - c[owner, 1])
        a[:, 1] = 1.0
        k, col, nxy = pairs
        a[k, col] = psf.pixel_values(pix[k, 0] - nxy[:, 0], pix[k, 1] - nxy[:, 1])
        a *= pix[:, 3:4]
        b = pix[:, 2] * pix[:, 3]
        out = np.empty(len(pix))
        for j, nc in enumerate(ncol):
            r = slice(bounds[j], bounds[j + 1])
            coef, *_ = np.linalg.lstsq(a[r, :nc], b[r], rcond=None)
            out[r] = b[r] - a[r, :nc] @ coef
        return out

    def run(use, theta0):
        n = len(use)
        pix = np.vstack([rows[i] for i in use])
        sizes = np.array([len(rows[i]) for i in use])
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        # (pixel, neighbour) pairs of every stamp: pixel index, model column, neighbour position
        pairs = [
            (np.arange(bounds[j], bounds[j + 1]), np.full(sizes[j], c + 2), np.tile(xy_nb, (sizes[j], 1)))
            for j, i in enumerate(use)
            for c, xy_nb in enumerate(nbrs[i])
        ]
        pairs = (
            tuple(np.concatenate(v) for v in zip(*pairs, strict=True))
            if pairs
            else (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)))
        )
        args = (
            pix,
            np.repeat(np.arange(n), sizes),
            bounds,
            np.array([stars[i] for i in use]),
            pairs,
            np.array([2 + len(nbrs[i]) for i in use]),
        )
        lo = np.concatenate([[np.log(0.3 * fwhm), np.log(0.2), -0.6, -0.6], np.full(2 * n, -1.5)])
        hi = np.concatenate([[np.log(3.0 * fwhm), np.log(20.0), 0.6, 0.6], np.full(2 * n, 1.5)])
        sparsity = lil_matrix((len(pix), len(theta0)), dtype=int)
        for j in range(n):
            sparsity[bounds[j] : bounds[j + 1], :4] = 1
            sparsity[bounds[j] : bounds[j + 1], 4 + 2 * j : 6 + 2 * j] = 1
        res = least_squares(
            residuals,
            theta0,
            bounds=(lo, hi),
            args=args,
            jac_sparsity=sparsity,
            x_scale="jac",
            loss="huber",
            f_scale=3.0,
            xtol=1e-6,
            ftol=1e-6,
            max_nfev=200,
        )
        chi2 = np.array([np.mean(res.fun[bounds[j] : bounds[j + 1]] ** 2) for j in range(n)])
        return res, chi2

    use = np.arange(len(stars))
    res, chi2 = run(use, np.concatenate([fallback.params, np.zeros(2 * len(use))]))
    good = chi2 <= 3.0 * np.median(chi2)
    if not good.all() and good.sum() >= 2:
        use = use[good]
        res, _ = run(use, np.concatenate([res.x[:4], res.x[4:].reshape(-1, 2)[good].ravel()]))
    jac = res.jac.toarray() if hasattr(res.jac, "toarray") else np.asarray(res.jac)
    dof = max(len(res.fun) - len(res.x), 1)
    # Profile out the star positions; dimensions pinned at a bound are not constrained by the data.
    cov = np.linalg.pinv(jac.T @ jac)[:4, :4] * max(1.0, 2.0 * res.cost / dof)
    cov[res.active_mask[:4] != 0, :] = 0.0
    cov[:, res.active_mask[:4] != 0] = 0.0
    return MoffatPSF.from_params(res.x[:4], order), cov


def _render(shape, psf, xy, flux, half):
    """Image of point sources with the given fluxes (non-finite or zero fluxes are skipped)."""
    out = np.zeros(shape)
    for (x, y), f in zip(xy, flux, strict=True):
        box = _box(shape, x, y, half) if np.isfinite(f) and f != 0 else None
        if box is not None:
            sl, px, py = box
            out[sl] += f * psf.design(px, py, [x], [y])[:, 0].reshape(out[sl].shape)
    return out


def _refine_position(psf, px, py, data, w, xy, others, lo, hi):
    """Best-fit position of the first source in a box (other columns solved linearly)."""
    sw = np.sqrt(w)

    def resid(p):
        a = np.column_stack([psf.design(px, py, [p[0]], [p[1]]), others])
        coef, *_ = np.linalg.lstsq(a * sw[:, None], data * sw, rcond=None)
        return (data - a @ coef) * sw

    x0 = np.clip(xy, lo + 1e-6, hi - 1e-6)
    return least_squares(resid, x0, bounds=(lo, hi), xtol=1e-4, ftol=1e-8, diff_step=1e-3).x


def forced_psf_photometry(
    image,
    positions,
    psf,
    *,
    source_xy,
    sky,
    rms,
    var=None,
    gain=None,
    psf_cov=None,
    box_fwhm=2.5,
    planar_sky=True,
    merge_fwhm=0.6,
    recenter_snr=None,
    neighbour_recenter_snr=10.0,
    isolation_fwhm=ISO,
):
    """PSF photometry at given positions with simultaneous fitting of neighbours.

    Parameters
    ----------
    image : numpy.ndarray
        Image including sky.
    positions : numpy.ndarray
        ``(M, 2)`` ``(x, y)`` positions to measure.
    psf : MoffatPSF
        PSF model with unit total flux.
    source_xy : numpy.ndarray
        ``(K, 2)`` detected sources. Those farther than ``merge_fwhm * FWHM``
        from every requested position are fitted as neighbours.
    sky, rms : numpy.ndarray
        Sky map (first-pass noise model only; every box fits its own sky) and
        sky RMS map.
    var : numpy.ndarray, optional
        Per-pixel variance. If omitted it is ``rms^2`` plus the model source
        counts divided by ``gain`` (when the gain is known).
    gain : float, optional
        Detector gain in electrons per count.
    psf_cov : numpy.ndarray, optional
        Covariance of ``psf.params``; if given, the PSF-model error is added to
        the flux errors in quadrature.
    box_fwhm : float
        Half-size of the fitting box in units of the PSF FWHM.
    planar_sky : bool
        Fit a plane (otherwise a constant) as the local sky in each box.
    merge_fwhm : float
        Detections closer than this (in FWHM) to a requested position are
        treated as that source.
    recenter_snr : float, optional
        Requested positions whose signal-to-noise ratio exceeds this are
        re-centred (by at most ``max(0.5 FWHM, 0.5 px)``). By default the
        positions are held fixed (strictly forced photometry).
    neighbour_recenter_snr : float
        Detected neighbours above this signal-to-noise ratio are re-centred.

    Returns
    -------
    flux, flux_err : numpy.ndarray
        Total fluxes and 1-sigma errors in image counts (NaN if unmeasurable).
    xy : numpy.ndarray
        Positions used for the requested sources.
    """
    pos = np.asarray(positions, float).reshape(-1, 2)
    fwhm = psf.fwhm
    comps = _scene(pos, np.asarray(source_xy, float).reshape(-1, 2), max(merge_fwhm * fwhm, 0.5))
    cur, m = comps.copy(), len(pos)
    thr = np.full(len(comps), float(neighbour_recenter_snr))
    thr[:m] = np.inf if recenter_snr is None else float(recenter_snr)
    if len(comps) > 1:  # blended sources keep their positions: centroid and flux are degenerate
        thr[cKDTree(comps).query(comps, k=2)[0][:, 1] < isolation_fwhm * fwhm] = np.inf
    max_shift = max(0.5 * fwhm, 0.5)
    half = int(max(np.ceil(box_fwhm * fwhm), 3))
    r_free = half + 0.5 * fwhm
    tree = cKDTree(comps)
    flux, err = np.zeros(len(comps)), np.full(len(comps), np.nan)
    perturbed = []
    if psf_cov is not None:
        lam, vec = np.linalg.eigh(psf_cov)
        perturbed = [
            MoffatPSF.from_params(psf.params + np.sqrt(lv) * v, psf.order)
            for lv, v in zip(lam, vec.T, strict=True)
            if lv > 0
        ]
    var = None if var is None else np.asarray(var, float)

    for final in (False, True):
        # Model image of every source from the previous pass; inside a box, the sources that
        # are not refitted (``fixed``) are represented by it.
        model_img = _render(image.shape, psf, cur, flux, int(np.ceil(half + 6.0 * fwhm)))
        prev_xy, new = cur.copy(), flux.copy()
        for i in range(m) if final else range(len(comps)):
            box = _box(image.shape, *comps[i], half)
            if box is None:
                new[i] = err[i] = np.nan
                continue
            sl, px, py = box
            raw = image[sl].ravel().astype(float)
            near = np.array(tree.query_ball_point(comps[i], r_free * np.sqrt(2.0)), dtype=int)
            near = near[np.max(np.abs(cur[near] - np.rint(comps[i])), axis=1) <= r_free]
            free = np.concatenate([[i], near[near != i]])
            prev_free = psf.design(px, py, prev_xy[free, 0], prev_xy[free, 1]) @ np.nan_to_num(flux[free])
            data = raw - (model_img[sl].ravel() - prev_free)
            sky_cols = [np.ones_like(px)]
            if planar_sky:
                sky_cols += [(px - np.rint(comps[i][0])) / half, (py - np.rint(comps[i][1])) / half]

            def design(model, px=px, py=py, free=free, sky_cols=sky_cols):
                return np.column_stack([model.design(px, py, cur[free, 0], cur[free, 1]), *sky_cols])

            a = design(psf)
            if var is not None:
                v = var[sl].ravel()
            else:
                src_model = model_img[sl].ravel() if final else raw - sky[sl].ravel()
                v = rms[sl].ravel() ** 2 + (np.maximum(src_model, 0.0) / gain if gain else 0.0)
            ok = np.isfinite(data) & np.isfinite(v) & (v > 0)
            if ok.sum() <= a.shape[1] or a[ok, 0].sum() < 0.05:
                new[i] = err[i] = np.nan
                continue
            w = 1.0 / v[ok]
            coef, cov = _weighted_lstsq(a[ok], data[ok], w)
            if cov[0, 0] > 0 and coef[0] > thr[i] * np.sqrt(cov[0, 0]):
                cur[i] = _refine_position(
                    psf, px[ok], py[ok], data[ok], w, cur[i], a[ok, 1:], comps[i] - max_shift, comps[i] + max_shift
                )
                a = design(psf)
                coef, cov = _weighted_lstsq(a[ok], data[ok], w)
                # The position was fitted too: take the flux variance from the linearised
                # model including the centroid derivatives (matters for blends).
                h = 1e-3
                grad = [
                    (
                        psf.design(px, py, [cur[i, 0] + dx], [cur[i, 1] + dy])
                        - psf.design(px, py, [cur[i, 0] - dx], [cur[i, 1] - dy])
                    )[:, 0]
                    * coef[0]
                    / (2 * h)
                    for dx, dy in ((h, 0.0), (0.0, h))
                ]
                cov = _weighted_lstsq(np.column_stack([a, *grad])[ok], data[ok], w)[1]
            var_f = cov[0, 0]
            if final:  # PSF-model error: re-solve with the shape moved along each principal axis
                for model in perturbed:
                    var_f += (_weighted_lstsq(design(model)[ok], data[ok], w)[0][0] - coef[0]) ** 2
            new[i] = coef[0]
            err[i] = np.sqrt(var_f) if var_f > 0 else np.nan
        flux = new
    return flux[:m], err[:m], cur[:m]


#: Requested positions brighter than this S/N are re-centred by :func:`psf_photometry`.
RECENTER_SNR = 10.0


def psf_photometry(image, positions, *, fwhm, source_xy, gain=None, var=None):
    """PSF photometry with an analytic, pixel-integrated Moffat PSF.

    Positions of sources detected at S/N > ``RECENTER_SNR`` are refined (detection
    centroids of undersampled stars are biased); fainter ones are measured at
    exactly the given position, as forced photometry of a transient should be.

    Parameters
    ----------
    image : numpy.ndarray
        Image in counts, sky included.
    positions : numpy.ndarray
        ``(M, 2)`` pixel positions ``(x, y)`` at which to measure fluxes.
    fwhm : float
        Approximate seeing FWHM in pixels.
    source_xy : numpy.ndarray
        ``(K, 2)`` detected sources, brightest first; used to choose PSF stars
        and as neighbours in the simultaneous fit.
    gain : float, optional
        Detector gain (e-/count), for the source shot-noise term when ``var`` is
        not given.
    var : numpy.ndarray, optional
        Per-pixel variance of ``image``. Estimated from the sky RMS (plus source
        shot noise) if omitted.

    Returns
    -------
    dict
        ``{"flux": total fluxes, "flux_err": 1-sigma errors, "psf": MoffatPSF}``.
    """
    img = np.asarray(image, dtype=float)
    bkg = estimate_background(np.where(np.isfinite(img), img, np.nanmedian(img)))
    rms = np.maximum(bkg.rms, 1e-12)
    if var is not None:
        fit_var = np.asarray(var, float)
    else:
        fit_var = rms**2
        if gain is not None and gain > 0:
            fit_var = fit_var + np.maximum(img - bkg.background, 0.0) / gain
    pos = np.asarray(positions, float).reshape(-1, 2)
    src = np.asarray(source_xy, float).reshape(-1, 2)
    psf, cov = fit_moffat_psf(
        img, src, float(fwhm), sky=bkg.background, var=fit_var, known_xy=_scene(pos, src, max(0.6 * fwhm, 0.5))
    )
    flux, err, _ = forced_psf_photometry(
        img,
        pos,
        psf,
        source_xy=src,
        sky=bkg.background,
        rms=rms,
        var=var,
        gain=gain,
        psf_cov=cov,
        recenter_snr=RECENTER_SNR,
    )
    return {"flux": flux, "flux_err": err, "psf": psf}
