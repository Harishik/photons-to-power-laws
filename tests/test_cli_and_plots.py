import numpy as np
import pytest

from grb_afterglow import LightCurve, broken_power_law, fit_lightcurve
from grb_afterglow.cli import main
from grb_afterglow.plotting import plot_closure_relations, plot_filmstrip, plot_lightcurve


@pytest.fixture
def lc_csv(tmp_path):
    t = np.logspace(3, 6, 25)
    flux = broken_power_law(t, 100.0, 5e4, 0.9, 2.1)
    rng = np.random.default_rng(0)
    err = 0.03 * flux
    path = tmp_path / "lc.csv"
    LightCurve(t, flux + rng.normal(0, err), err, "r").save_csv(path)
    return path


def test_cli_fit_prefers_broken(lc_csv, capsys, tmp_path):
    assert main(["fit", str(lc_csv), "--plot", str(tmp_path / "fit.png")]) == 0
    out = capsys.readouterr().out
    assert "Preferred: broken_power_law" in out
    assert (tmp_path / "fit.png").stat().st_size > 10_000


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "p2pl" in capsys.readouterr().out


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_figures_render(lc_csv, tmp_path, theme):
    lc = LightCurve.read_csv(lc_csv)
    fit = fit_lightcurve(lc, "broken")
    plot_lightcurve(lc, fit, theme=theme, savepath=tmp_path / "lc.png")
    plot_closure_relations([("x", 2.2, 0.05, 0.6, 0.02)], theme=theme, savepath=tmp_path / "cl.png")
    img = np.random.default_rng(0).normal(0, 1, (60, 60))
    plot_filmstrip(
        [{"image": img, "x": 30, "y": 30, "label": "a"}, {"image": img, "x": 30, "y": 30, "label": "b", "flag": "bad"}],
        theme=theme,
        savepath=tmp_path / "fs.png",
    )
    for name in ("lc.png", "cl.png", "fs.png"):
        assert (tmp_path / name).stat().st_size > 5_000


def test_sed_and_posterior_figures_render(tmp_path):
    from grb_afterglow.extinction import deredden
    from grb_afterglow.mcmc import sample_posterior
    from grb_afterglow.plotting import plot_posterior, plot_sed

    t = np.repeat(np.geomspace(1e4, 1e6, 8), 3)
    bands = np.array(["g", "r", "i"] * 8, dtype=object)
    nu_ratio = np.select([bands == "g", bands == "r"], [617 / 481, 1.0], 617 / 752)
    flux = 50 * (t / 1e5) ** -1.2 * nu_ratio**-0.7
    observed = LightCurve(t, flux * 0.5, flux * 0.02, bands)
    corrected = deredden(observed, ebv_mw=0.2)
    fit = fit_lightcurve(corrected, "single", reference_band="r")
    plot_sed(observed, corrected, fit, 1.0, title="SED", subtitle="test", savepath=tmp_path / "sed.png")
    assert (tmp_path / "sed.png").stat().st_size > 0
    post = sample_posterior(corrected, fit, n_walkers=12, n_steps=300, burn=100)
    _, axes = plot_posterior(
        post, derived={"p": post.p("ism_slow")}, truths={"beta": 0.7}, theme="dark", savepath=tmp_path / "post.png"
    )
    assert axes.shape == (3, 3)  # alpha, beta and the derived p
    assert (tmp_path / "post.png").stat().st_size > 0
    other = sample_posterior(corrected, fit, n_walkers=12, n_steps=200, burn=50, seed=1)
    fig, _ = plot_posterior(post, ["alpha", "beta"], compare=other, labels=("a", "b"), savepath=tmp_path / "two.png")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["a", "b"]
    _, axes = plot_posterior(post, derived={"p": post.p("ism_slow")}, compare=other)  # p is the first's only
    assert axes.shape == (3, 3)
