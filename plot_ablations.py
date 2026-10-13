#!/usr/bin/env python3
"""CSV-driven 2x3 PepMetaStack ablation figure.

All six analyses are visualization matrices, not rescaled raw-metric heatmaps:
  - Cells contain ORIGINAL metric values (mean and, where reported, SD).
  - Cell color represents percentage improvement in the mean relative to the
    highlighted reference model for THAT panel, using the correct metric direction.
  - A shared, zero-centered, nonlinear color scale keeps marginal changes pale;
    it is NOT a significance test. Larger changes saturate at +/- 15%.

Usage:
    python scripts/plot_ablations.py
    python scripts/plot_ablations.py --dpi 600 --no-individual

Requirements: matplotlib, numpy (no seaborn, pandas or LaTeX runtime needed).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path.cwd()  # Run from your project root
DATA_DIR = ROOT / "results" / "ablation_results_data"
FIG_DIR = ROOT / "results" / "plots" / "ablation_results"

METRICS = {
    "rmse": (r"RMSE $\downarrow$", -1),
    "kendall_tau": (r"Kendall $\tau\uparrow$", 1),
    "r2": (r"$R^2\uparrow$", 1),
    "pearson_r": (r"Pearson $r\uparrow$", 1),
    "spearman_rho": (r"Spearman $\rho\uparrow$", 1),
    "mae": (r"MAE $\downarrow$", -1),
}
FOUR = ("rmse", "kendall_tau", "r2", "pearson_r")
SIX = ("pearson_r", "spearman_rho", "kendall_tau", "rmse", "mae", "r2")

@dataclass(frozen=True)
class Panel:
    tag: str
    name: str
    filename: str
    metrics: tuple[str, ...]
    reference: str
    labels: dict[str, str]
    filename_stub: str

PANELS = (
    Panel("a", "PLM representations and fusion", "01_plm_representations.csv", FOUR,
          "ProtT5", {
              "ESM-2 320-d": "ESM-2 (320-d)",
              "ESM-2 640-d": "ESM-2 (640-d)",
              "ESM-2 1280-d": "ESM-2 (1280-d)",
              "ProstT5": "ProstT5",
              "ProtT5": "ProtT5",
              "ProtT5+ESM, Concatenation": "ProtT5+ESM (concat.)",
              "ProtT5+ESM, Cross Attention": "ProtT5+ESM (cross-attn.)",
              "AMP-fine-tuned ProtT5": "AMP-tuned ProtT5",
          }, "01_plm_representations"),
    Panel("b", "Biological feature ablation", "02_biological_features.csv", FOUR,
          "PLM", {}, "02_biological_features"),
    Panel("c", "Individual base predictors", "03_base_predictors.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "PepMetaStack (full)",
              "HistGradientBoosting": "HistGB",
          }, "03_base_predictors"),
    Panel("d", "Leave-one-predictor-out", "04_leave_one_out.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "Full ensemble",
              "w/o XGBoost": "Remove XGBoost",
              "w/o Ridge": "Remove Ridge",
              "w/o MLP": "Remove MLP",
              "w/o KNN": "Remove KNN",
              "w/o HistGradientBoosting": "Remove HistGB",
              "w/o Extra Trees": "Remove Extra Trees",
              "w/o SVR": "Remove SVR",
          }, "04_leave_one_out"),
    Panel("e", "Predictor families and ensembles", "05_predictor_families.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "PepMetaStack (full)",
              "Heterogeneous-3 (SVR+ET+MLP)": "Hetero-3: SVR+ET+MLP",
              "Heterogeneous-2 (SVR+ET)": "Hetero-2: SVR+ET",
              "Distance-based (SVR+KNN)": "Distance: SVR+KNN",
              "Tree-based (ET+HGB+XGB)": "Trees: ET+HGB+XGB",
              "Heterogeneous-3 (Ridge+XGB+MLP)": "Hetero-3: Ridge+XGB+MLP",
          }, "05_predictor_families"),
    Panel("f", "Meta-regressor comparison", "06_meta_regressors.csv", SIX,
          "RidgeCV (Ours)", {
              "Linear Regression": "Linear regression",
              "RidgeCV (Ours)": "RidgeCV (ours)",
          }, "06_meta_regressors"),
)

# A colorblind-friendlier diverging ramp: coral = worse, off-white = unchanged,
# teal = better. Matplotlib's SymLogNorm prevents 0.1% changes from looking huge.
COLORMAP = LinearSegmentedColormap.from_list(
    "pepmetastack_delta", ["#ad4949", "#efc6c0", "#f8f8f6", "#b9dadd", "#126b7a"]
)
NORM = SymLogNorm(linthresh=0.65, linscale=0.85, vmin=-15, vmax=15, base=10, clip=True)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",       # Editable vector text in SVG
        "pdf.fonttype": 42,         # Embed TrueType in PDF
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "axes.unicode_minus": False,
    })


def read_csv(panel: Panel, input_dir: Path) -> list[dict]:
    filepath = input_dir / panel.filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing CSV: {filepath}. Check --data-dir and CSV filenames.")
    with filepath.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or sum(r["model"] == panel.reference for r in rows) != 1:
        raise ValueError(f"Missing or repeated reference {panel.reference!r} in {filepath}")
    for row in rows:
        for metric in panel.metrics:
            key = f"{metric}_mean"
            if not row.get(key):
                raise ValueError(f"Missing metric {key!r} for {row['model']} in {filepath}")
            row[key] = float(row[key])
            sdkey = f"{metric}_sd"
            row[sdkey] = float(row[sdkey]) if row.get(sdkey) else None
    return rows


def oriented_percent_change(mean: float, ref: float, direction: int) -> float:
    """Positive always means an improvement, even for error metrics."""
    if ref == 0:
        raise ZeroDivisionError("Relative change is undefined for a reference of zero")
    return direction * (mean - ref) / abs(ref) * 100.0


def text_color(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return "#ffffff" if luminance < 0.54 else "#1b2b35"


def draw_panel(ax: mpl.axes.Axes, panel: Panel, rows: list[dict], *,
               standalone: bool = False) -> None:
    metrics, m, n = panel.metrics, len(panel.metrics), len(rows)
    ref_row = next(row for row in rows if row["model"] == panel.reference)
    left_space = 2.8 if m == 4 else 2.72

    ax.set_xlim(-left_space, m + 0.01)
    ax.set_ylim(n + 0.05, -0.86)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    labelfont = 7.25 if standalone else 6.9
    valuefont = 8.2 if standalone else 7.5
    sdfont = 6.9 if standalone else 6.25
    if m == 6:
        labelfont -= 0.15
        valuefont -= 0.65
        sdfont -= 0.6

    # Very light alternating row guides, visually less obtrusive than a table grid.
    for rindex, row in enumerate(rows):
        is_ref = row["model"] == panel.reference
        if rindex % 2 == 1:
            ax.add_patch(Rectangle((-left_space, rindex), left_space, 1,
                                   color="#fafbfc", lw=0, zorder=0))
        if is_ref:
            ax.add_patch(Rectangle((-left_space, rindex), left_space, 1,
                                   color="#eaf0f2", lw=0, zorder=0))
        display = panel.labels.get(row["model"], row["model"])
        ax.text(-left_space + 0.12, rindex + 0.51, display,
                va="center", ha="left", fontsize=labelfont,
                fontweight="bold" if is_ref else "regular",
                color="#193847" if is_ref else "#34434e", clip_on=False)
        for c, metric in enumerate(metrics):
            val = row[metric + "_mean"]
            sd = row[metric + "_sd"]
            _, direction = METRICS[metric]
            change = oriented_percent_change(val, ref_row[metric+"_mean"], direction)
            facecolor = COLORMAP(NORM(change))
            ax.add_patch(Rectangle((c, rindex), 1, 1,
                                   facecolor=facecolor, edgecolor="white",
                                   linewidth=1.45, zorder=1))
            fg = text_color(facecolor)
            if sd is not None:
                ax.text(c + .5, rindex + .365, f"{val:.3f}",
                        ha="center", va="center", fontsize=valuefont,
                        fontweight="bold" if is_ref else "medium",
                        color=fg, zorder=3)
                ax.text(c + .5, rindex + .735, f"±{sd:.3f}",
                        ha="center", va="center", fontsize=sdfont,
                        color=fg, zorder=3)
            else:
                ax.text(c + .5, rindex + .52, f"{val:.3f}",
                        ha="center", va="center", fontsize=valuefont + .2,
                        fontweight="bold" if is_ref else "medium",
                        color=fg, zorder=3)
        if is_ref:
            ax.add_patch(Rectangle((-left_space + .015, rindex + .015),
                                   left_space + m - .03, .97,
                                   facecolor="none", edgecolor="#516875",
                                   linewidth=1.15, zorder=4))

    for col, metric in enumerate(metrics):
        label = METRICS[metric][0]
        if m == 6:
            # More compact headers at the same font size.
            label = {
                "pearson_r": r"$r\uparrow$", "spearman_rho": r"$\rho\uparrow$",
                "kendall_tau": r"$\tau\uparrow$", "rmse": r"RMSE$\downarrow$",
                "mae": r"MAE$\downarrow$", "r2": r"$R^2\uparrow$",
            }[metric]
        else:
            label = {
                "rmse": r"RMSE$\downarrow$", "kendall_tau": r"$\tau\uparrow$",
                "r2": r"$R^2\uparrow$", "pearson_r": r"$r\uparrow$",
            }[metric]
        ax.text(col + .5, -.43, label, va="center", ha="center",
                fontsize=8 if standalone else 7.4,
                fontweight="bold", color="#243747")
    ax.set_title(f"({panel.tag})  {panel.name}", loc="left", pad=6,
                 fontsize=10 if standalone else 9.5,
                 color="#183847", fontweight="bold")


def make_colorbar(fig, position: list[float], *, standalone: bool = False):
    cax = fig.add_axes(position)
    mapper = mpl.cm.ScalarMappable(norm=NORM, cmap=COLORMAP)
    cb = fig.colorbar(mapper, cax=cax, orientation="horizontal", extend="both")
    ticks = [-15, -5, -1, 0, 1, 5, 15]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["-15", "-5", "-1", "0", "+1", "+5", "+15"])
    cb.ax.tick_params(labelsize=7 if not standalone else 8, length=2,
                      width=.5, pad=1.5, colors="#314550")
    cb.outline.set_linewidth(.5)
    cb.outline.set_edgecolor("#b5c1c6")
    cb.set_label("Change from highlighted reference (%)   ← worse   |   better →",
                 fontsize=7.9 if not standalone else 9,
                 color="#263f4a", labelpad=3)
    return cb


def save_all(fig, output_dir: Path, stem: str, dpi: int) -> None:
    for extension in ("pdf", "svg", "png"):
        out = output_dir / f"{stem}.{extension}"
        options = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(out, format=extension, facecolor="white", **options)
        print(f"Saved {out.relative_to(ROOT)}")


def make_combined(data, output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(8.6, 9.65), facecolor="white")
    gs = fig.add_gridspec(3, 2, left=.028, right=.988, bottom=.135, top=.966,
                          wspace=.077, hspace=.31, height_ratios=[10.0, 9.0, 7.5])
    for idx, (panel, rows) in enumerate(data):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        draw_panel(ax, panel, rows)
    make_colorbar(fig, [.195, .067, .61, .015])
    fig.text(.5, .028,
             "Color compares metric means with each panel's outlined reference; it does not indicate statistical significance."
             "   CV cells: mean over five folds (top) ± SD (bottom).",
             ha="center", va="center", fontsize=7.0, color="#475963")
    save_all(fig, output_dir, "pepmetastack_ablations_2x3", dpi)
    plt.close(fig)


def make_standalone(data, output_dir: Path, dpi: int) -> None:
    for panel, rows in data:
        height = {9: 4.0, 8: 3.8, 6: 3.25}[len(rows)]
        fig = plt.figure(figsize=(6.55, height), facecolor="white")
        ax = fig.add_axes([.035, .21, .935, .68])
        draw_panel(ax, panel, rows, standalone=True)
        make_colorbar(fig, [.18, .105, .64, .025], standalone=True)
        fig.text(.5, .035, "Outlined row = reference; colored cells = directional change in mean.",
                 ha="center", fontsize=7.4, color="#52636b")
        save_all(fig, output_dir, panel.filename_stub, dpi)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing the six CSV datasets")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR,
                        help="Directory for PDF, SVG, and PNG figures")
    parser.add_argument("--dpi", type=int, default=600,
                        help="PNG resolution (PDF and SVG remain vector)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Create only the combined 2x3 panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    data = [(panel, read_csv(panel, args.data_dir)) for panel in PANELS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_combined(data, args.output_dir, args.dpi)
    if not args.no_individual:
        make_standalone(data, args.output_dir, args.dpi)
    print("Done: 6 CSV datasets -> 2x3 figure + 6 individual figures")


if __name__ == "__main__":
    main()
