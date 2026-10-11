"""PSF photometry with an empirical, oversampled point-spread function (ePSF).

The PSF is built from the image itself (cf. DAOPHOT, Stetson 1987; Anderson &
King 2000) as ``PSF = (M + R) / (1 + integral of R)``. ``M`` is a
pixel-integrated elliptical Moffat profile fitted to bright stars; it carries
the wings to infinity, so the normalisation, and hence every returned flux, is
the *total* flux. ``R`` is an empirical residual on a grid oversampled per
pixel, fitted by regularised least squares to the stacked,
sub-pixel-registered, neighbour-subtracted PSF stars. The regularisation pulls
``R`` to zero where the data carry no information, so the model falls back to
the analytic Moffat when PSF stars are few, and pixel integration plus
oversampling handle undersampled images.

Fluxes are measured at *fixed* positions by one global, variance-weighted
linear least-squares fit of the targets, all other detections (deblending),
sources found in the fit residuals, and a smooth background correction; only
the neighbours' positions are refined (Gauss-Newton). Errors come from the fit
covariance, include a PSF-model error measured on the PSF stars, and are
inflated where the local residuals exceed the noise. The scene fit uses dense
linear algebra, so it suits images with up to a few hundred sources.

Across a full CCD frame the PSF changes with position, and a scene fit of
thousands of stars would be too large anyway. :func:`build_spatial_psf` fits
the frame tile by tile, to subtract neighbours and select PSF stars, and then
fits one :class:`SpatialPSF` to the PSF stars of all tiles together: its
Moffat parameters are planes in position, each gradient kept only if
significant, and its empirical residual map is common to the frame.
:func:`psf_photometry_tiled` then fits the scene tile by tile, every source
with the PSF at its own position.

On a difference image, where every static source has been subtracted, the
transient is measured with :func:`forced_psf_flux` instead: a linear fit of the
PSF and a local background whose noise is measured by applying the same
estimator at blank positions, because PSF-matching convolution correlates the
pixel noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, ndimage, optimize, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .detection import estimate_background

N_ITER = 4  # scene fit / PSF rebuild cycles
PSF_SNR_MIN = 20.0  # minimum S/N of a PSF star
PSF_MAX_CONTAM = 0.1  # max. neighbour light within 1.5 FWHM, relative to the star's own
MOVE_SNR_MIN = 10.0  # minimum S/N for refining a neighbour's position
FIND_SNR_MIN = 5.0  # matched-filter S/N of extra sources found in the residuals
BKG_SPACING_FWHM = 8.0  # node spacing of the background correction, in FWHM

# 4-point Gauss-Legendre rule on [-0.5, 0.5]: accurate pixel integration of the Moffat.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)
_GL_X, _GL_W = _GL_X / 2.0, _GL_W / 2.0


def moffat_pixel(dx, dy, alpha: float, beta: float, g1: float = 0.0, g2: float = 0.0) -> np.ndarray:
    """Pixel-integrated elliptical Moffat profile with unit total flux.

    Parameters
    ----------
    dx, dy
        Offsets of pixel centres from the source position, in pixels.
    alpha, beta
        Moffat core width (px) and power-law index (``beta > 1``).
    g1, g2
        Shear-like ellipticity components (``g1**2 + g2**2 < 1``). The profile
        area is preserved, so the normalisation is independent of them.

    Returns
    -------
    numpy.ndarray
        Fraction of the total flux falling in each pixel.
    """
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    s = 1.0 / np.sqrt(1.0 - g1 * g1 - g2 * g2) / alpha**2
    a11, a22, a12 = (1.0 - g1) * s, (1.0 + g1) * s, -g2 * s
    out = np.zeros(np.broadcast(dx, dy).shape)
    for ox, wx in zip(_GL_X, _GL_W, strict=True):
        x = dx + ox
        for oy, wy in zip(_GL_X, _GL_W, strict=True):
            y = dy + oy
            out += wx * wy * (1.0 + a11 * x * x + 2.0 * a12 * x * y + a22 * y * y) ** (-beta)
    return out * (beta - 1.0) / (np.pi * alpha**2)


def _bilinear(dx, dy, half, oversample):
    """Indices and weights of the 4 grid nodes around each offset (zero weight off-grid)."""
    n = 2 * half * oversample + 1
    tx, ty = (np.ravel(dx) + half) * oversample, (np.ravel(dy) + half) * oversample
    kx, ky = np.floor(tx).astype(int), np.floor(ty).astype(int)
    fx, fy = tx - kx, ty - ky
    ok = (kx >= 0) & (ky >= 0) & (kx < n - 1) & (ky < n - 1)
    kx, ky = np.where(ok, kx, 0), np.where(ok, ky, 0)
    idx = np.stack([ky * n + kx, ky * n + kx + 1, (ky + 1) * n + kx, (ky + 1) * n + kx + 1], axis=1)
    w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1) * ok[:, None]
    return idx, w


@dataclass
class EmpiricalPSF:
    """Analytic Moffat core and wings plus an oversampled empirical residual.

    Calling the instance with pixel offsets returns the fraction of the total
    flux in those pixels. ``residual`` (``None`` for a purely analytic model)
    is a square grid of half-width ``half`` px and node spacing
    ``1 / oversample``. ``n_stars`` is the number of PSF stars it was built
    from and ``model_error`` the fractional model error measured on them.
    """

    alpha: float
    beta: float
    g1: float = 0.0
    g2: float = 0.0
    residual: np.ndarray | None = None
    half: int = 0
    oversample: int = 2
    n_stars: int = 0
    model_error: float = 0.0

    @classmethod
    def moffat_from_fwhm(cls, fwhm: float, beta: float = 2.5, **kwargs) -> EmpiricalPSF:
        """A round analytic Moffat profile of the given FWHM (px)."""
        return cls(fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)), beta, **kwargs)

    @property
    def fwhm(self) -> float:
        """Geometric-mean FWHM of the (not pixel-integrated) Moffat component, in pixels."""
        return float(2.0 * self.alpha * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0))

    def moffat(self, dx, dy) -> np.ndarray:
        """The analytic component alone, normalised to unit total flux."""
        return moffat_pixel(dx, dy, self.alpha, self.beta, self.g1, self.g2)

    def __call__(self, dx, dy) -> np.ndarray:
        dx, dy = np.broadcast_arrays(np.asarray(dx, float), np.asarray(dy, float))
        model = self.moffat(dx, dy)
        if self.residual is None:
            return model
        idx, w = _bilinear(dx, dy, self.half, self.oversample)
        res = (self.residual.ravel()[idx] * w).sum(axis=1).reshape(model.shape)
        return (model + res) / (1.0 + self.residual.sum() / self.oversample**2)


class ConvolvedPSF:
    """A PSF seen through a pixel convolution kernel, such as a PSF-matching kernel.

    If an image with PSF ``psf`` is convolved (``scipy.ndimage.correlate``)
    with ``kernel``, a point source in the result has the profile
    ``sum_k kernel_k * psf(dx + ox_k, dy + oy_k)`` exactly, for any sub-pixel
    position, because both the PSF and the kernel act on whole pixels. The
    kernel is normalised, so fluxes remain total fluxes.
    """

    def __init__(self, psf: EmpiricalPSF, kernel: np.ndarray, fwhm: float, *, min_weight: float = 1e-4):
        kernel = np.asarray(kernel, dtype=float) / np.sum(kernel)
        half_y, half_x = (n // 2 for n in kernel.shape)
        oy, ox = np.nonzero(np.abs(kernel) >= min_weight * np.abs(kernel).max())
        self.psf = psf
        self.weights = kernel[oy, ox] / kernel[oy, ox].sum()
        self.offsets = np.column_stack([ox - half_x, oy - half_y]).astype(float)
        self.fwhm = float(fwhm)

    def __call__(self, dx, dy) -> np.ndarray:
        dx, dy = np.broadcast_arrays(np.asarray(dx, float), np.asarray(dy, float))
        out = np.zeros(dx.shape)
        for w, (ox, oy) in zip(self.weights, self.offsets, strict=True):
            out += w * self.psf(dx + ox, dy + oy)
        return out


SLOPE_T_MIN = 3.0  # significance a spatial PSF gradient needs to be kept


class SpatialPSF:
    """A PSF that varies across the frame.

    The analytic part varies smoothly with position (DAOPHOT's variable PSF,
    Stetson 1987, instead keeps the analytic function fixed and lets its
    residual look-up tables vary linearly or quadratically): ``log alpha``, ``log(beta - 1)``, ``g1`` and ``g2``
    of the Moffat profile are planes in ``(x, y)``, fitted jointly to the PSF
    stars of the whole frame (each star with the profile at its own
    position), and a gradient is kept only where it is significant. The
    empirical residual map is common to the frame. ``plane[k]`` holds the
    coefficients ``(c0, c_u, c_v)`` of parameter ``k`` in
    ``u = (x - centre_x) / scale``, ``v = (y - centre_y) / scale``.
    """

    def __init__(self, plane, centre, scale, *, residual=None, half=0, oversample=2, model_error=0.0, n_stars=0):
        self.plane = np.asarray(plane, dtype=float).reshape(4, 3)
        self.centre = (float(centre[0]), float(centre[1]))
        self.scale = float(scale)
        self.residual, self.half, self.oversample = residual, int(half), int(oversample)
        self.model_error, self.n_stars = float(model_error), int(n_stars)

    def parameters(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Moffat ``(alpha, beta, g1, g2)`` at positions ``(x, y)``."""
        u = (np.asarray(x, dtype=float) - self.centre[0]) / self.scale
        v = (np.asarray(y, dtype=float) - self.centre[1]) / self.scale
        la, lb, g1, g2 = (c[0] + c[1] * u + c[2] * v for c in self.plane)
        return np.exp(la), 1.0 + np.exp(lb), np.clip(g1, -0.5, 0.5), np.clip(g2, -0.5, 0.5)

    def at(self, x: float, y: float) -> EmpiricalPSF:
        """The PSF at pixel ``(x, y)``."""
        alpha, beta, g1, g2 = (float(a) for a in self.parameters(x, y))
        return EmpiricalPSF(
            alpha, beta, g1, g2, self.residual, self.half, self.oversample, self.n_stars, self.model_error
        )

    def evaluate(self, dx, dy, x, y) -> np.ndarray:
        """PSF values at offsets ``(dx, dy)`` from sources at ``(x, y)``, each with the PSF at its own position."""
        dx, dy, x, y = (np.asarray(a, dtype=float) for a in np.broadcast_arrays(dx, dy, x, y))
        model = moffat_pixel(dx, dy, *self.parameters(x, y))
        if self.residual is None:
            return model
        idx, w = _bilinear(dx, dy, self.half, self.oversample)
        res = (self.residual.ravel()[idx] * w).sum(axis=1).reshape(model.shape)
        return (model + res) / (1.0 + self.residual.sum() / self.oversample**2)

    def __call__(self, dx, dy) -> np.ndarray:
        """The PSF at the reference position (for matched filtering)."""
        return self.at(*self.centre)(dx, dy)

    def shifted(self, x0: float, y0: float) -> SpatialPSF:
        """The same PSF in the coordinates of a sub-image whose corner is at ``(x0, y0)``."""
        return SpatialPSF(
            self.plane,
            (self.centre[0] - x0, self.centre[1] - y0),
            self.scale,
            residual=self.residual,
            half=self.half,
            oversample=self.oversample,
            model_error=self.model_error,
            n_stars=self.n_stars,
        )

    def fwhm_at(self, x, y) -> np.ndarray:
        """FWHM (px) of the Moffat component at positions ``(x, y)``."""
        alpha, beta, _, _ = self.parameters(x, y)
        return 2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0)

    @property
    def fwhm(self) -> float:
        return float(self.fwhm_at(*self.centre))

    @property
    def gradients(self) -> dict[str, tuple[float, float]]:
        """Fitted gradients ``(d/du, d/dv)`` of each parameter (zero where not significant)."""
        names = ("log_alpha", "log_beta_minus_1", "g1", "g2")
        return {n: (float(c[1]), float(c[2])) for n, c in zip(names, self.plane, strict=True)}


def _fit_moffat_spatial(stamps, positions, centre, scale, start):
    """Fit Moffat parameters that are planes in position jointly to all PSF-star stamps.

    Like :func:`_fit_moffat` (variable projection over each star's flux and
    sky), but every star is modelled with the profile at its own position.
    Gradients are dropped one at a time, least significant first, until all
    that remain exceed ``SLOPE_T_MIN`` sigma. Returns the ``(4, 3)`` plane
    coefficients and the stars' sky offsets.
    """
    sid = np.concatenate([np.full(len(st[0]), i) for i, st in enumerate(stamps)])
    dx, dy, c, w = (np.concatenate([st[k] for st in stamps]) for k in range(4))
    pos = np.asarray(positions, dtype=float)
    u = ((pos[:, 0] - centre[0]) / scale)[sid]
    v = ((pos[:, 1] - centre[1]) / scale)[sid]
    s11, s1c = np.bincount(sid, w), np.bincount(sid, w * c)

    def linear(m):
        smm, sm1, smc = np.bincount(sid, w * m * m), np.bincount(sid, w * m), np.bincount(sid, w * m * c)
        det = smm * s11 - sm1**2
        return (s11 * smc - sm1 * s1c) / det, (smm * s1c - sm1 * smc) / det

    free = np.ones((4, 3), dtype=bool)
    plane = np.zeros((4, 3))
    plane[:, 0] = start
    lo0, hi0 = np.array([np.log(0.2), np.log(0.2), -0.5, -0.5]), np.array([np.log(50.0), np.log(15.0), 0.5, 0.5])
    while True:
        keys, base = np.argwhere(free), plane.copy()

        def full(p, keys=keys, base=base):
            out = base.copy()
            out[tuple(keys.T)] = p
            return out

        def model(p, full=full):
            q = full(p)
            la, lb, g1, g2 = (q[k, 0] + q[k, 1] * u + q[k, 2] * v for k in range(4))
            g1, g2 = np.clip(g1, -0.5, 0.5), np.clip(g2, -0.5, 0.5)
            return moffat_pixel(dx, dy, np.exp(la), 1.0 + np.exp(lb), g1, g2)

        def resid(p, model=model):
            m = model(p)
            f, b = linear(m)
            return np.sqrt(w) * (c - f[sid] * m - b[sid])

        lo = np.array([lo0[k] if j == 0 else -2.0 for k, j in keys])
        hi = np.array([hi0[k] if j == 0 else 2.0 for k, j in keys])
        p0 = np.clip(plane[tuple(keys.T)], lo + 1e-6, hi - 1e-6)
        sol = optimize.least_squares(resid, p0, bounds=(lo, hi), x_scale=0.1)
        plane = full(sol.x)
        slopes = [i for i, (_, j) in enumerate(keys) if j > 0]
        if not slopes:
            break
        dof = max(len(sol.fun) - len(sol.x), 1)
        cov = linalg.pinvh(sol.jac.T @ sol.jac) * float(np.sum(sol.fun**2) / dof)
        t = np.abs(sol.x[slopes]) / np.sqrt(np.maximum(np.diag(cov)[slopes], 1e-300))
        if t.min() >= SLOPE_T_MIN:
            break
        k, j = keys[slopes[int(np.argmin(t))]]
        free[k, j], plane[k, j] = False, 0.0
    return plane, linear(model(sol.x))[1]


@dataclass
class PSFPhotometry:
    """Result of :func:`psf_photometry`.

    ``flux`` (total flux in counts) and ``flux_err`` (1 sigma) per requested
    position (NaN where a position has no usable pixels); ``psf`` is the model
    used and ``xy`` every source in the final scene fit (targets first).
    """

    flux: np.ndarray
    flux_err: np.ndarray
    psf: EmpiricalPSF | SpatialPSF
    xy: np.ndarray


def _stamps(shape, xy, half):
    """Source index, flat pixel index and offsets for square stamps of all sources."""
    ny, nx = shape
    g = np.arange(-half, half + 1)
    px = np.rint(xy[:, 0]).astype(int)[:, None, None] + g[None, None, :]
    py = np.rint(xy[:, 1]).astype(int)[:, None, None] + g[None, :, None]
    px, py = np.broadcast_arrays(px, py)
    src = np.broadcast_to(np.arange(len(xy))[:, None, None], px.shape)
    ok = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    src, px, py = src[ok], px[ok], py[ok]
    return src, py * nx + px, px - xy[src, 0], py - xy[src, 1]


def _tent_basis(shape, spacing):
    """Sparse bilinear-interpolation basis (pixels x nodes) for a smooth background."""
    axes = []
    for n in shape:
        m = max(2, int(np.ceil((n - 1) / spacing)) + 1)
        t = np.arange(n) * (m - 1) / max(n - 1, 1)
        k = np.minimum(np.floor(t).astype(int), m - 2)
        f = t - k
        rows, cols = np.repeat(np.arange(n), 2), np.column_stack([k, k + 1]).ravel()
        axes.append(sparse.csr_matrix((np.column_stack([1 - f, f]).ravel(), (rows, cols)), shape=(n, m)))
    return sparse.kron(axes[0], axes[1], format="csr")


def _psf_values(psf, dx, dy, xy, src):
    """PSF at stamp offsets; a :class:`SpatialPSF` is evaluated at each pixel's own source position."""
    if isinstance(psf, SpatialPSF):
        return psf.evaluate(dx, dy, xy[src, 0], xy[src, 1])
    return psf(dx, dy)


def _scene_fit(data, var, xy, psf, half, basis, bkg_sigma, move=None, step=0.05):
    """Global weighted linear fit of all fluxes plus a smooth background correction.

    If ``move`` flags sources, their PSF derivatives are fitted too and the
    implied Gauss-Newton position shifts (clipped to 0.3 px) are returned.
    """
    shape = data.shape
    src, pix, dx, dy = _stamps(shape, xy, half)
    val = _psf_values(psf, dx, dy, xy, src)
    fit_idx = np.flatnonzero(np.isfinite(var.ravel()))
    row_of = -np.ones(data.size, int)
    row_of[fit_idx] = np.arange(fit_idx.size)
    use = row_of[pix] >= 0
    n_src, n_fit = len(xy), fit_idx.size
    blocks = [sparse.csr_matrix((val[use], (row_of[pix[use]], src[use])), shape=(n_fit, n_src))]
    n_mv = 0 if move is None else int(move.sum())
    if n_mv:
        mv = use & move[src]
        col = np.cumsum(move) - 1
        for ddx, ddy in ((step, 0.0), (0.0, step)):
            g = (
                _psf_values(psf, dx[mv] - ddx, dy[mv] - ddy, xy, src[mv])
                - _psf_values(psf, dx[mv] + ddx, dy[mv] + ddy, xy, src[mv])
            ) / (2 * step)
            blocks.append(sparse.csr_matrix((g, (row_of[pix[mv]], col[src[mv]])), shape=(n_fit, n_mv)))
    blocks.append(basis[fit_idx])
    a = sparse.hstack(blocks, format="csr")
    w = 1.0 / var.ravel()[fit_idx]
    normal = np.asarray((a.T @ a.multiply(w[:, None])).todense())
    rhs = a.T @ (w * data.ravel()[fit_idx])
    nb = basis.shape[1]
    normal[-nb:, -nb:] += np.eye(nb) / bkg_sigma**2  # weak prior: correction ~ 1 sky sigma
    alive = np.diag(normal) > 0  # sources with no usable pixels get NaN
    sub = np.ix_(alive, alive)
    cov = np.full(normal.shape, np.nan)
    try:
        cov[sub] = linalg.cho_solve(linalg.cho_factor(normal[sub]), np.eye(alive.sum()))
    except linalg.LinAlgError:  # exactly coincident sources: use the pseudo-inverse
        cov[sub] = linalg.pinvh(normal[sub])
    par = np.full(normal.shape[0], np.nan)
    par[alive] = cov[sub] @ rhs[alive]
    flux, err = par[:n_src], np.sqrt(np.diag(cov)[:n_src])
    model = np.bincount(pix, weights=np.nan_to_num(flux)[src] * val, minlength=data.size).reshape(shape)
    bkg = (basis @ np.nan_to_num(par[-nb:])).reshape(shape)
    shift = np.zeros_like(xy)
    if n_mv:
        with np.errstate(invalid="ignore", divide="ignore"):
            shift[move] = np.column_stack([par[n_src : n_src + n_mv], par[n_src + n_mv : n_src + 2 * n_mv]])
            shift[move] /= flux[move][:, None]
        shift = np.clip(np.nan_to_num(shift), -0.3, 0.3)
    return {"flux": flux, "err": err, "model": model, "bkg": bkg, "shift": shift, "src": src, "pix": pix, "val": val}


def _select_psf_stars(fit, xy, psf, var, fwhm, n_max=40):
    """Bright sources away from the edges, free of masked pixels, whose cores are not dominated by neighbours."""
    flux, model = fit["flux"], fit["model"]
    shape = model.shape
    h = int(np.ceil(1.5 * fwhm))
    margin = h + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = flux / fit["err"]
    inside = np.all((xy > margin) & (xy < np.array(shape[::-1]) - 1 - margin), axis=1)
    ok = np.isfinite(snr) & (snr > PSF_SNR_MIN) & inside
    keep = []
    g = np.arange(-h, h + 1)
    for i in np.flatnonzero(ok)[np.argsort(-snr[ok])]:
        px = np.rint(xy[i, 0]).astype(int) + g[None, :]
        py = np.rint(xy[i, 1]).astype(int) + g[:, None]
        if not np.all(np.isfinite(var[py, px])):  # saturated or bad pixels in the core
            continue
        own = flux[i] * psf(px - xy[i, 0], py - xy[i, 1])
        if np.sum(model[py, px] - own) < PSF_MAX_CONTAM * np.sum(own):
            keep.append(i)
        if len(keep) >= n_max:
            break
    return np.asarray(keep, int)


def _star_stamps(data, var, fit, xy, psf, idx, radius, nb_err=0.05):
    """Neighbour-subtracted pixels around each PSF star as (dx, dy, counts, weight)."""
    ny, nx = data.shape
    h = int(np.ceil(radius))
    out = []
    for i in idx:
        x0, y0 = xy[i]
        gx = np.arange(max(int(np.rint(x0)) - h, 0), min(int(np.rint(x0)) + h + 1, nx))
        gy = np.arange(max(int(np.rint(y0)) - h, 0), min(int(np.rint(y0)) + h + 1, ny))
        px, py = (a.ravel() for a in np.meshgrid(gx, gy))
        others = fit["model"][py, px] - fit["flux"][i] * psf(px - x0, py - y0)
        c = data[py, px] - fit["bkg"][py, px] - others
        v = var[py, px] + (nb_err * others) ** 2  # 5% error on subtracted neighbours
        ok = (np.hypot(px - x0, py - y0) <= radius) & np.isfinite(v)
        out.append((px[ok] - x0, py[ok] - y0, c[ok], 1.0 / v[ok]))
    return out


def _fit_moffat(stamps, psf, fit_shape):
    """Fit (alpha, beta[, g1, g2]) jointly to all PSF-star stamps.

    Each star's flux and local sky offset are solved linearly at every step
    (variable projection). Returns the shape parameters and the sky offsets.
    """
    sid = np.concatenate([np.full(len(s[0]), i) for i, s in enumerate(stamps)])
    dx, dy, c, w = (np.concatenate([s[k] for s in stamps]) for k in range(4))
    s11, s1c = np.bincount(sid, w), np.bincount(sid, w * c)

    def linear(m):
        smm, sm1, smc = np.bincount(sid, w * m * m), np.bincount(sid, w * m), np.bincount(sid, w * m * c)
        det = smm * s11 - sm1**2
        return (s11 * smc - sm1 * s1c) / det, (smm * s1c - sm1 * smc) / det

    def model(p):
        return moffat_pixel(dx, dy, np.exp(p[0]), 1.0 + np.exp(p[1]), *(p[2:] if fit_shape else (0.0, 0.0)))

    def resid(p):
        m = model(p)
        f, b = linear(m)
        return np.sqrt(w) * (c - f[sid] * m - b[sid])

    p0 = [np.log(psf.alpha), np.log(psf.beta - 1.0)] + ([psf.g1, psf.g2] if fit_shape else [])
    lo = [np.log(0.2), np.log(0.2)] + ([-0.5, -0.5] if fit_shape else [])
    hi = [np.log(50.0), np.log(15.0)] + ([0.5, 0.5] if fit_shape else [])
    p = optimize.least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), x_scale=0.1).x
    g = p[2:] if fit_shape else (0.0, 0.0)
    return (float(np.exp(p[0])), float(1.0 + np.exp(p[1])), float(g[0]), float(g[1])), linear(model(p))[1]


def _fit_residual(stamps, psf, half, oversample, smooth=1.0, ridge=0.02, moffat=None):
    """Regularised least-squares fit of the oversampled empirical residual grid.

    ``moffat`` gives the analytic profile at every stamp pixel directly (for a
    spatially varying PSF); otherwise it is ``psf.moffat``.
    """
    n = 2 * half * oversample + 1
    sid = np.concatenate([np.full(len(s[0]), i) for i, s in enumerate(stamps)])
    dx, dy, c, w = (np.concatenate([s[k] for s in stamps]) for k in range(4))
    m = psf.moffat(dx, dy) if moffat is None else np.asarray(moffat, dtype=float)
    flux = np.bincount(sid, w * c * m) / np.bincount(sid, w * m * m)  # per-star flux, Moffat-normalised
    y = c / flux[sid] - m
    wt = w * flux[sid] ** 2
    g = np.arange(n) / oversample - half
    taper = np.clip((half - np.hypot(*np.meshgrid(g, g))) / (0.3 * half), 0.0, 1.0) ** 2  # zero at the edge
    idx, bw = _bilinear(dx, dy, half, oversample)
    bw = bw * taper.ravel()[idx]
    b = sparse.csr_matrix((bw.ravel(), (np.repeat(np.arange(len(y)), 4), idx.ravel())), shape=(len(y), n * n))
    eye = sparse.identity(n, format="csr")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    d1 = sparse.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n))
    scale = wt.sum() / (np.pi * (half * oversample) ** 2)  # mean data weight per grid node
    normal = b.T @ b.multiply(wt[:, None]) + ridge * scale * sparse.identity(n * n)
    for op in (sparse.kron(eye, d2), sparse.kron(d2, eye), np.sqrt(2.0) * sparse.kron(d1, d1)):
        normal = normal + smooth * scale * (op.T @ op)
    coef = spsolve(normal.tocsc(), b.T @ (wt * y))
    return coef.reshape(n, n) * taper


def _rebuild_psf(fit, data, var, xy, psf, fwhm, half, oversample):
    """Select PSF stars from the current scene model and refit the ePSF to them.

    The returned PSF records the number of PSF stars and the fractional
    PSF-model error (excess core residual power of the PSF stars beyond the
    noise).
    """
    idx = _select_psf_stars(fit, xy, psf, var, fwhm)
    if len(idx) == 0:
        return psf
    stamps = _star_stamps(data, var, fit, xy, psf, idx, max(3.0 * fwhm, 5.0))
    shape, sky = _fit_moffat(stamps, psf, fit_shape=len(stamps) >= 3)
    new = EmpiricalPSF(*shape, half=half, oversample=oversample, n_stars=len(idx))
    stamps = [(s[0], s[1], s[2] - b, s[3]) for s, b in zip(stamps, sky, strict=True)]
    if len(stamps) >= 5:
        inner = [tuple(a[np.hypot(s[0], s[1]) <= half] for a in s) for s in stamps]
        new.residual = _fit_residual(inner, new, half, oversample)
    excess = []
    for dx, dy, c, w in stamps:
        m_all = new(dx, dy)
        core = np.hypot(dx, dy) <= 1.5 * fwhm
        m = np.sum(w * c * m_all) / np.sum(w * m_all * m_all) * m_all[core]
        excess.append(np.sum((c[core] - m) ** 2 - 1.0 / w[core]) / np.sum(m * m))
    new.model_error = float(np.sqrt(max(np.median(excess), 0.0))) if len(excess) >= 3 else 0.0
    return new


def _residual_sources(res, var, psf, xy, fwhm):
    """Point sources missing from the model: matched-filter peaks in the residual image."""
    h = int(np.ceil(1.5 * fwhm))
    g = np.arange(-h, h + 1)
    kern = psf(g[None, :], g[:, None])
    inv = np.where(np.isfinite(var), 1.0 / var, 0.0)
    num = ndimage.correlate(res * inv, kern, mode="constant")
    den = ndimage.correlate(inv, kern**2, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(den > 0, num / np.sqrt(den), 0.0)
        fl = np.where(den > 0, num / den, 0.0)
    peak = (snr == ndimage.maximum_filter(snr, size=3)) & (snr > FIND_SNR_MIN)
    peak[[0, -1], :] = peak[:, [0, -1]] = False
    iy, ix = np.nonzero(peak)
    order = np.argsort(-snr[iy, ix])
    iy, ix = iy[order], ix[order]
    with np.errstate(invalid="ignore", divide="ignore"):  # parabolic sub-pixel peak
        sx = 0.5 * (fl[iy, ix - 1] - fl[iy, ix + 1]) / (fl[iy, ix - 1] - 2 * fl[iy, ix] + fl[iy, ix + 1])
        sy = 0.5 * (fl[iy - 1, ix] - fl[iy + 1, ix]) / (fl[iy - 1, ix] - 2 * fl[iy, ix] + fl[iy + 1, ix])
    cand = np.column_stack([ix + np.clip(np.nan_to_num(sx), -0.5, 0.5), iy + np.clip(np.nan_to_num(sy), -0.5, 0.5)])
    new = []
    for c in cand:  # brightest first; keep peaks more than 1 FWHM from every known source
        others = np.vstack([xy, *new]) if new else xy
        if len(others) == 0 or np.min(np.hypot(*(others - c).T)) > fwhm:
            new.append(c[None, :])
    return np.vstack(new) if new else np.zeros((0, 2))


def _fit_scene(image, positions, source_xy, *, fwhm, var, gain, psf, state=None):
    """Iterate scene fits, neighbour positions, residual sources and (unless ``psf`` is given) the PSF.

    If ``state`` is a dict it receives the final scene fit, the background-subtracted
    data, the variance and the source positions.
    """
    data = np.asarray(image, float)
    pos = np.atleast_2d(np.asarray(positions, float)).reshape(-1, 2)
    det = np.atleast_2d(np.asarray(source_xy, float)).reshape(-1, 2)
    det = det[np.all(np.isfinite(det), axis=1)]
    good = np.all(np.isfinite(pos), axis=1)
    half = int(np.ceil(max(3.0 * fwhm, 5.0)))  # PSF stamp half-width (px)
    oversample = 4 if fwhm < 2.5 else 2

    # Source list: the distinct targets (fixed), then the detections that are not one of them (movable).
    tgt, inv = np.unique(pos[good], axis=0, return_inverse=True)
    if len(det) and len(tgt):
        det = det[cKDTree(tgt).query(det)[0] > 0.75 * fwhm]
    xy = np.vstack([tgt, det])
    movable = np.arange(len(xy)) >= len(tgt)
    xy_start = xy.copy()

    sky = estimate_background(np.where(np.isfinite(data), data, np.nanmedian(data)))
    finite = np.isfinite(data)
    v_base = np.where(finite, sky.rms**2 if var is None else np.asarray(var, float), np.inf)
    v_base = np.where(v_base > 0, v_base, np.inf)
    resid0 = np.where(finite, data - sky.background, 0.0)
    basis = _tent_basis(data.shape, max(BKG_SPACING_FWHM * fwhm, 16.0))
    bkg_sigma = max(sky.median_rms, 1e-6)

    fixed_psf = psf is not None
    if not fixed_psf:  # start from a round beta = 2.5 Moffat of the given FWHM
        psf = EmpiricalPSF.moffat_from_fwhm(fwhm, half=half, oversample=oversample)
    else:
        half = max(half, psf.half)
    v, fit = v_base, {}
    for it in range(N_ITER):
        move = None
        if it > 0:  # refine the positions of neighbours with S/N > MOVE_SNR_MIN in the last fit
            with np.errstate(invalid="ignore", divide="ignore"):
                move = movable & (fit["flux"] / fit["err"] > MOVE_SNR_MIN)
        fit = _scene_fit(resid0, v, xy, psf, half, basis, bkg_sigma, move=move)
        if move is not None and move.any():
            xy = xy + fit["shift"]
            far = np.hypot(*(xy - xy_start).T) > 1.0  # a neighbour may wander at most 1 px
            xy[far] = xy_start[far]
        model = np.maximum(fit["model"], 0.0)
        v = v_base + (model / gain if (var is None and gain) else 0.0) + (psf.model_error * model) ** 2
        if it == N_ITER - 1:
            break
        if it > 0 or fixed_psf:  # add sources missed by the detection list (once the PSF is empirical)
            new = _residual_sources(resid0 - fit["bkg"] - fit["model"], v, psf, xy, fwhm)
            if len(new):
                xy, xy_start = np.vstack([xy, new]), np.vstack([xy_start, new])
                movable = np.concatenate([movable, np.ones(len(new), bool)])
                fit = _scene_fit(resid0, v, xy, psf, half, basis, bkg_sigma)
        if not fixed_psf:
            psf = _rebuild_psf(fit, resid0, v, xy, psf, fwhm, half, oversample)

    fit = _scene_fit(resid0, v, xy, psf, half, basis, bkg_sigma)
    # Inflate errors where the PSF-weighted mean chi^2 of the local residuals exceeds 1.
    chi = ((resid0 - fit["bkg"] - fit["model"]) ** 2 / v).ravel()[fit["pix"]]
    ok = np.isfinite(chi)
    num = np.bincount(fit["src"], weights=fit["val"] * np.where(ok, chi, 0.0), minlength=len(xy))
    den = np.bincount(fit["src"], weights=fit["val"] * ok, minlength=len(xy))
    with np.errstate(invalid="ignore", divide="ignore"):
        err = fit["err"] * np.sqrt(np.maximum(np.where(den > 0, num / den, 1.0), 1.0))
    out_f, out_e = np.full(len(pos), np.nan), np.full(len(pos), np.nan)
    out_f[good], out_e[good] = fit["flux"][inv.ravel()], err[inv.ravel()]
    if state is not None:
        state.update(fit=fit, data=resid0, var=v, xy=xy)
    return PSFPhotometry(out_f, out_e, psf, xy)


def build_psf(
    image: np.ndarray,
    source_xy: np.ndarray,
    *,
    fwhm: float,
    var: np.ndarray | None = None,
    gain: float | None = None,
) -> EmpiricalPSF:
    """Build an empirical PSF from the stars in an image.

    Parameters
    ----------
    image
        Calibrated image in counts, sky included. NaN pixels (e.g. saturated
        stars) are ignored, and stars with NaN in their core are not used.
    source_xy
        ``(M, 2)`` detected sources, brightest first. PSF stars are chosen
        among them; all are fitted simultaneously so that neighbours are
        subtracted from the PSF stars.
    fwhm
        Approximate seeing FWHM in pixels (sets the initial PSF and the radii).
    var, gain
        As in :func:`psf_photometry`.

    Returns
    -------
    EmpiricalPSF
        Normalised to unit *total* flux. A PSF built on one image can be
        fitted on another with the same PSF, such as a difference image.
    """
    xy = np.atleast_2d(np.asarray(source_xy, float)).reshape(-1, 2)
    return _fit_scene(image, np.zeros((0, 2)), xy, fwhm=float(fwhm), var=var, gain=gain, psf=None).psf


def psf_photometry(
    image: np.ndarray,
    positions: np.ndarray,
    *,
    source_xy: np.ndarray,
    fwhm: float | None = None,
    psf: EmpiricalPSF | None = None,
    var: np.ndarray | None = None,
    gain: float | None = None,
) -> PSFPhotometry:
    """Forced PSF photometry at fixed pixel positions, deblending all neighbours.

    Parameters
    ----------
    image
        Calibrated image in counts, sky included. NaN pixels are ignored.
    positions
        ``(N, 2)`` pixel positions ``(x, y)`` at which to measure fluxes. They
        are kept fixed (forced photometry).
    source_xy
        ``(M, 2)`` detected sources, brightest first, fitted simultaneously as
        neighbours (positions refined if S/N > ``MOVE_SNR_MIN``). Detections
        within ``0.75 fwhm`` of a position are taken to be that position's
        source and are not fitted separately.
    fwhm
        Approximate seeing FWHM in pixels; defaults to that of ``psf``.
    psf
        PSF model to use. If omitted it is built from this image together
        with the fit, as in :func:`build_psf`.
    var
        Per-pixel variance of ``image``. If omitted: local sky variance plus,
        if ``gain`` is given, the shot noise of the fitted model.
    gain
        Electrons per count, for the source shot noise when ``var`` is omitted.
    """
    if fwhm is None:
        if psf is None:
            raise ValueError("Give fwhm or psf.")
        fwhm = psf.fwhm
    return _fit_scene(image, positions, source_xy, fwhm=float(fwhm), var=var, gain=gain, psf=psf)


def build_spatial_psf(
    image: np.ndarray,
    source_xy: np.ndarray,
    *,
    fwhm: float,
    tiles: tuple[int, int],
    var: np.ndarray | None = None,
    gain: float | None = None,
    margin: float | None = None,
) -> SpatialPSF:
    """Build a :class:`SpatialPSF` from the stars of a full frame.

    The frame is split into ``tiles = (nx, ny)`` tiles, each fitted as a scene
    (with ``margin`` px around it, default ``max(15, 5 fwhm)``) to subtract
    every star's neighbours and select PSF stars; the PSF model is then fitted
    to the PSF stars of all tiles together.
    """
    from .subtraction import _tile_edges

    data = np.asarray(image, dtype=float)
    ny, nx = data.shape
    src = np.atleast_2d(np.asarray(source_xy, float)).reshape(-1, 2)
    margin = float(max(15.0, 5.0 * fwhm) if margin is None else margin)
    xe, ye = _tile_edges(nx, int(tiles[0])), _tile_edges(ny, int(tiles[1]))
    radius = max(3.0 * fwhm, 5.0)
    stamps, positions, starts, local_psfs = [], [], [], []
    for j in range(len(ye) - 1):
        for i in range(len(xe) - 1):
            x0, x1 = int(max(xe[i] - margin, 0)), int(min(xe[i + 1] + margin, nx))
            y0, y1 = int(max(ye[j] - margin, 0)), int(min(ye[j + 1] + margin, ny))
            inside = (src[:, 0] >= x0) & (src[:, 0] < x1) & (src[:, 1] >= y0) & (src[:, 1] < y1)
            if not inside.any():
                continue
            sub_var = None if var is None else np.asarray(var, float)[y0:y1, x0:x1]
            state: dict = {}
            try:
                res = _fit_scene(
                    data[y0:y1, x0:x1], np.zeros((0, 2)), src[inside] - np.array([x0, y0], float),
                    fwhm=float(fwhm), var=sub_var, gain=gain, psf=None, state=state,
                )  # fmt: skip
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
            xy = state["xy"]
            idx = _select_psf_stars(state["fit"], xy, res.psf, state["var"], fwhm)
            gx, gy = xy[idx, 0] + x0, xy[idx, 1] + y0  # each PSF star once: in the tile it lies in
            idx = idx[(gx >= xe[i]) & (gx < xe[i + 1]) & (gy >= ye[j]) & (gy < ye[j + 1])]
            if len(idx) == 0:
                continue
            stamps += _star_stamps(state["data"], state["var"], state["fit"], xy, res.psf, idx, radius)
            positions += list(xy[idx] + np.array([x0, y0], float))
            starts.append([np.log(res.psf.alpha), np.log(res.psf.beta - 1.0), res.psf.g1, res.psf.g2])
            local_psfs.append(res.psf)
    if len(stamps) < 3:
        raise RuntimeError("Too few PSF stars in the frame for a spatially varying PSF.")
    centre, scale = ((nx - 1) / 2.0, (ny - 1) / 2.0), max(nx, ny) / 2.0
    plane, sky = _fit_moffat_spatial(stamps, positions, centre, scale, np.median(starts, axis=0))
    half, oversample = local_psfs[0].half, local_psfs[0].oversample
    psf = SpatialPSF(plane, centre, scale, half=half, oversample=oversample, n_stars=len(stamps))
    stamps = [(st[0], st[1], st[2] - b, st[3]) for st, b in zip(stamps, sky, strict=True)]
    if len(stamps) >= 5:
        inner = [tuple(a[np.hypot(st[0], st[1]) <= half] for a in st) for st in stamps]
        moffat = np.concatenate([psf.at(*xy).moffat(st[0], st[1]) for st, xy in zip(inner, positions, strict=True)])
        psf.residual = _fit_residual(inner, None, half, oversample, moffat=moffat)
    excess = []
    for (dx, dy, c, w), xy in zip(stamps, positions, strict=True):
        m_all = psf.at(*xy)(dx, dy)
        core = np.hypot(dx, dy) <= 1.5 * fwhm
        m = np.sum(w * c * m_all) / np.sum(w * m_all * m_all) * m_all[core]
        excess.append(np.sum((c[core] - m) ** 2 - 1.0 / w[core]) / np.sum(m * m))
    psf.model_error = float(np.sqrt(max(np.median(excess), 0.0))) if len(excess) >= 3 else 0.0
    return psf


def psf_photometry_tiled(
    image: np.ndarray,
    positions: np.ndarray,
    *,
    source_xy: np.ndarray,
    fwhm: float,
    tiles: tuple[int, int],
    psf: SpatialPSF | None = None,
    var: np.ndarray | None = None,
    gain: float | None = None,
    margin: float | None = None,
) -> PSFPhotometry:
    """:func:`psf_photometry` with a PSF that varies across the frame, for full CCD frames.

    Builds a :class:`SpatialPSF` (:func:`build_spatial_psf`) unless one is
    given, then measures the positions tile by tile (``tiles = (nx, ny)``): each
    tile's positions in a scene fit of every source within ``margin`` of the
    tile, every source with the PSF at its own position. Returns a
    :class:`PSFPhotometry` whose ``psf`` is the :class:`SpatialPSF`.
    """
    from .subtraction import _tile_edges

    data = np.asarray(image, dtype=float)
    ny, nx = data.shape
    pos = np.atleast_2d(np.asarray(positions, float)).reshape(-1, 2)
    src = np.atleast_2d(np.asarray(source_xy, float)).reshape(-1, 2)
    margin = float(max(15.0, 5.0 * fwhm) if margin is None else margin)
    if psf is None:
        psf = build_spatial_psf(data, src, fwhm=fwhm, tiles=tiles, var=var, gain=gain, margin=margin)
    xe, ye = _tile_edges(nx, int(tiles[0])), _tile_edges(ny, int(tiles[1]))
    col = np.clip(np.searchsorted(xe, pos[:, 0], side="right") - 1, 0, len(xe) - 2)
    row = np.clip(np.searchsorted(ye, pos[:, 1], side="right") - 1, 0, len(ye) - 2)
    flux, flux_err = np.full(len(pos), np.nan), np.full(len(pos), np.nan)
    fitted = []
    for j in range(len(ye) - 1):
        for i in range(len(xe) - 1):
            mine = (col == i) & (row == j)
            if not mine.any():
                continue
            x0, x1 = int(max(xe[i] - margin, 0)), int(min(xe[i + 1] + margin, nx))
            y0, y1 = int(max(ye[j] - margin, 0)), int(min(ye[j + 1] + margin, ny))
            inside = (src[:, 0] >= x0) & (src[:, 0] < x1) & (src[:, 1] >= y0) & (src[:, 1] < y1)
            shift = np.array([x0, y0], float)
            sub_var = None if var is None else np.asarray(var, float)[y0:y1, x0:x1]
            res = _fit_scene(
                data[y0:y1, x0:x1], pos[mine] - shift, src[inside] - shift,
                fwhm=float(fwhm), var=sub_var, gain=gain, psf=psf.shifted(x0, y0),
            )  # fmt: skip
            flux[mine], flux_err[mine] = res.flux, res.flux_err
            fitted.append(res.xy + shift)
    xy = np.vstack(fitted) if fitted else src
    return PSFPhotometry(flux, flux_err, psf, xy)


@dataclass
class ForcedPSFFlux:
    """A forced PSF fit of one source on an image with no other sources nearby."""

    x: float
    y: float
    flux: float
    flux_err: float
    background: float
    blank_noise: float


def _forced_estimator(psf, x: float, y: float, radius: float, valid: np.ndarray | None):
    """Linear estimators of (flux, background) for a PSF at (x, y) over a circular stamp."""
    ix, iy = int(np.rint(x)), int(np.rint(y))
    h = int(np.ceil(radius))
    g = np.arange(-h, h + 1)
    gx, gy = np.meshgrid(g, g)
    dx, dy = gx + ix - x, gy + iy - y
    use = np.hypot(dx, dy) <= radius
    if valid is not None:
        use &= valid
    p = psf(dx, dy)
    design = np.column_stack([p[use], np.ones(use.sum())])
    pinv = np.linalg.pinv(design)
    est_f, est_b = np.zeros_like(p), np.zeros_like(p)
    est_f[use], est_b[use] = pinv
    return ix, iy, h, est_f, est_b, p


def forced_psf_flux(
    image: np.ndarray,
    x: float,
    y: float,
    psf: EmpiricalPSF | ConvolvedPSF,
    *,
    radius: float | None = None,
    mask: np.ndarray | None = None,
    gain: float | None = None,
    n_blank: int = 300,
    seed: int = 0,
) -> ForcedPSFFlux:
    """PSF flux of one source at a fixed position, with an empirical error.

    Fits ``flux * PSF + background`` by unweighted linear least squares (the
    optimal estimator in the background-limited regime) within ``radius``
    (default 2.5 FWHM). The noise is measured by applying the identical
    estimator at up to ``n_blank`` blank positions, which captures the
    pixel-to-pixel noise correlation of resampled or convolved images such as
    difference images; the source's own shot noise is added if ``gain`` is
    given.

    Parameters
    ----------
    image
        Typically a difference image (NaN where invalid).
    x, y
        Pixel position of the source.
    psf
        PSF of ``image`` (an :class:`EmpiricalPSF` or :class:`ConvolvedPSF`),
        normalised to unit total flux.
    radius
        Fitting radius in pixels.
    mask
        True where sources or bad pixels are; blank positions avoid them.
    gain
        Electrons per count of ``image``.
    """
    data = np.asarray(image, dtype=float)
    radius = float(radius if radius is not None else max(2.5 * psf.fwhm, 4.0))
    finite = np.isfinite(data)
    ix, iy, h, est_f, est_b, p = _forced_estimator(psf, x, y, radius, None)
    ny, nx = data.shape
    if not (h <= ix < nx - h and h <= iy < ny - h):
        return ForcedPSFFlux(float(x), float(y), np.nan, np.nan, np.nan, np.nan)
    window = (slice(iy - h, iy + h + 1), slice(ix - h, ix + h + 1))
    if not finite[window][est_f != 0].all():  # refit on the usable pixels only
        ix, iy, h, est_f, est_b, p = _forced_estimator(psf, x, y, radius, finite[window])
    stamp = np.where(finite[window], data[window], 0.0)
    flux = float(np.sum(est_f * stamp))
    background = float(np.sum(est_b * stamp))

    # The same estimator at blank positions: correlate the image with it.
    bad = ~finite if mask is None else (~finite | np.asarray(mask, dtype=bool))
    clear = ndimage.distance_transform_edt(~bad) > radius + 1.0
    clear[:h, :] = clear[ny - h :, :] = clear[:, :h] = clear[:, nx - h :] = False
    clear[max(iy - 2 * h, 0) : iy + 2 * h + 1, max(ix - 2 * h, 0) : ix + 2 * h + 1] = False
    ys, xs = np.nonzero(clear)
    blank = np.nan
    if xs.size >= 20:
        rng = np.random.default_rng(seed)
        pick = rng.choice(xs.size, size=min(n_blank, xs.size), replace=False)
        sums = ndimage.correlate(np.where(finite, data, 0.0), est_f, mode="constant")[ys[pick], xs[pick]]
        blank = float(1.4826 * np.median(np.abs(sums - np.median(sums))))
    shot2 = max(flux, 0.0) / gain * float(np.sum(est_f**2 * p)) if gain else 0.0
    return ForcedPSFFlux(float(x), float(y), flux, float(np.sqrt(blank**2 + shot2)), background, blank)
