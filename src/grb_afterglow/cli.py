"""Command-line interface: ``p2pl demo | simulate | fit``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__


def _cmd_demo(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .demo import run_demo

    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    summary = run_demo(args.out, seed=args.seed, figures_dir=args.figures, themes=themes, animate=args.animate)
    return 0 if summary["status"] == "PASS" else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .synthetic import generate_dataset

    truth = generate_dataset(args.out, seed=args.seed)
    print(f"Wrote {len(truth['frames'])} science frames, calibration frames, catalog.csv and truth.json to {args.out}/")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from .fitting import compare_models, fit_lightcurve
    from .lightcurve import LightCurve

    lc = LightCurve.read_csv(args.csv)
    if args.tmin is not None:
        lc = lc.subset(lc.time_days >= args.tmin)
    fits = []
    for model in ["single", "broken"] if args.model == "auto" else [args.model]:
        try:
            fits.append(fit_lightcurve(lc, model, sys_err=args.sys_err))
        except (RuntimeError, ValueError) as exc:
            print(f"{model} fit failed: {exc}", file=sys.stderr)
    if not fits:
        return 1
    for fit in fits:
        print(fit.summary())
    if len(fits) > 1:
        comparison = compare_models(*fits)
        print()
        print(comparison.summary())
        print(f"\nPreferred: {comparison.best.model_name}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import plot_lightcurve

        best = compare_models(*fits).best
        alt = next((f for f in fits if f is not best), None)
        plot_lightcurve(lc, best, alternative=alt, theme=args.theme, savepath=args.plot)
        print(f"Saved {args.plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p2pl",
        description="Photons to Power Laws: GRB afterglow photometry and light-curve physics.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="simulate, reduce, fit, and validate a synthetic afterglow")
    demo.add_argument("--out", default="p2pl_demo", help="output directory (default: p2pl_demo)")
    demo.add_argument("--figures", default=None, help="figure directory (default: OUT/figures)")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    demo.add_argument("--animate", action="store_true", help="also render an animated GIF")
    demo.set_defaults(func=_cmd_demo)

    sim = sub.add_parser("simulate", help="write a synthetic campaign (FITS frames + catalog) to disk")
    sim.add_argument("out")
    sim.add_argument("--seed", type=int, default=42)
    sim.set_defaults(func=_cmd_simulate)

    fit = sub.add_parser("fit", help="fit power laws to a light-curve CSV")
    fit.add_argument("csv", help="CSV with time_s and flux_ujy/flux_err_ujy (or mag/mag_err), optional band")
    fit.add_argument("--model", choices=("auto", "single", "broken"), default="auto")
    fit.add_argument("--tmin", type=float, default=None, help="ignore points before this time (days)")
    fit.add_argument("--sys-err", type=float, default=0.0, help="fractional systematic error floor")
    fit.add_argument("--plot", default=None, help="save a figure to this path")
    fit.add_argument("--theme", choices=("light", "dark"), default="light")
    fit.set_defaults(func=_cmd_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
