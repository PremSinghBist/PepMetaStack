#!/usr/bin/env python3
"""PepMetaStack: PLM representations and fusion, horizontal two-panel figure.

Reads ONLY ``01_plm_representations.csv`` (from the supplied ablation scripts).

    (a) Four metric-specific dot/whisker tracks: CV mean +/- SD.
    (b) Heatmap: directional percentage change vs ProtT5, with raw mean +/- SD
        printed inside each cell.

A positive heatmap percentage indicates BETTER performance (for RMSE,
smaller values are better; for correlations/R², larger values are better).

Example:
    python plot_plm_representations_fusion_2panel.py \
        --data-dir results/ablation_results_data \
        --output-dir results/plots/plm_representations \
        --width 9.2 --height 4.1 --dpi 600

Outputs a single figure as PNG, PDF, and SVG. No other CSVs or training needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


METRICS = (
    ("rmse", r"RMSE $\downarrow$", -1),
    ("kendall_tau", r"Kendall $\tau\uparrow$", +1),
    ("r2", r"$R^2\uparrow$", +1),
    ("pearson_r", r"Pearson $r\uparrow$", +1),
)

REFERENCE = "ProtT5"
ALIASES = {
    "ESM-2 320-d": "ESM-2 (320-d)",
    "ESM-2 640-d": "ESM-2 (640-d)",
    "ESM-2 1280-d": "ESM-2 (1280-d)",
    "ProstT5": "ProstT5",
    "ProtT5": "ProtT5",
    "ProtT5+ESM, Concatenation": "ProtT5+ESM (concat.)",
    "ProtT5+ESM, Cross Attention": "ProtT5+ESM (cross-attn.)",
    "AMP-fine-tuned ProtT5": "AMP-tuned ProtT5",
}

TEAL = "#087e83"
GRAY = "#758494"
DARK = "#253b49"
GRID = "#e6eaee"
COLORMAP = LinearSegmentedColormap.from_list(
    "plm_delta", ["#ad4949", "#efc6c0", "#f8f8f6", "#b9dadd", "#126b7a"]
)
NORM = SymLogNorm(linthresh=0.65, linscale=0.85, vmin=-15, vmax=15,
                  base=10, clip=True)


def read_plm_csv(path: Path) -> pd.DataFrame:
    """Read only the PLM experiment, preserving original CSV model order."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing PLM CSV: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = ["model"] + [f"{metric}_mean" for metric, _, _ in METRICS]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    if df.empty or df["model"].isna().any() or df["model"].duplicated().any():
        raise ValueError("Model labels must be unique and nonempty")
    if (df["model"] == REFERENCE).sum() != 1:
        raise ValueError(f"Exactly one reference row named '{REFERENCE}' is required")

    for metric, _, _ in METRICS:
        mean_col, sd_col = f"{metric}_mean", f"{metric}_sd"
        df[mean_col] = pd.to_numeric(df[mean_col], errors="raise")
        if not np.isfinite(df[mean_col]).all():
            raise ValueError(f"Nonfinite values in {mean_col}")
        if sd_col not in df.columns:
            df[sd_col] = np.nan
        df[sd_col] = pd.to_numeric(df[sd_col], errors="coerce")
        if (df[sd_col].dropna() < 0).any():
            raise ValueError(f"Negative standard deviations in {sd_col}")
    df["short_label"] = df["model"].map(ALIASES).fillna(df["model"])
    return df


def metric_limits(values: np.ndarray, sds: np.ndarray) -> tuple[float, float]:
    sds = np.nan_to_num(sds, nan=0)
    minimum = float(np.min(values - sds))
    maximum = float(np.max(values + sds))
    span = max(maximum - minimum, 0.006)
    return minimum - 0.10 * span, maximum + 0.10 * span


def contrast_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.54 else "#1b2b35"


def make_figure(df: pd.DataFrame, outdir: Path, filename: str,
                width: float = 9.2, height: float = 4.1,
                dpi: int = 600) -> dict[str, Path]:
    if width <= 0 or height <= 0 or dpi <= 0:
        raise ValueError("Width, height and dpi must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    ref = df.loc[df["model"] == REFERENCE].iloc[0]
    n = len(df)

    rc = {
        "font.family": "DejaVu Sans", "font.size": 7.4,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "figure.facecolor": "white",
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(width, height), facecolor="white")

        # Two equal horizontal panels: left dot/whisker; right metric heatmap.
        # Coordinates are normalized by figure width and height.
        left_x, panel_w = 0.025, 0.462
        right_x = 0.516
        plot_y, plot_h = 0.205, 0.675
        left_label_w = panel_w * 0.365
        right_label_w = panel_w * 0.380

        fig.text(left_x, 0.957, "(a)",
                 ha="left", va="top", fontsize=9, weight="bold", color=DARK)
        fig.text(right_x, 0.957, "(b)",
                 ha="left", va="top", fontsize=9, weight="bold", color=DARK)

        # ---------- (a) metric-specific dot-and-whisker tracks ----------
        left_label_ax = fig.add_axes([left_x, plot_y, left_label_w, plot_h])
        left_label_ax.set_xlim(0, 1)
        left_label_ax.set_ylim(n - .55, -.75)
        left_label_ax.axis("off")

        for i, row in df.iterrows():
            is_ref = row["model"] == REFERENCE
            left_label_ax.text(
                0.98, i, row["short_label"], ha="right", va="center",
                color=TEAL if is_ref else DARK, fontsize=6.25,
                weight="bold" if is_ref else "normal",
            )

        # Reserve a small inner gap between the label area and metric tracks.
        left_data_x = left_x + left_label_w + 0.007
        left_data_w = panel_w - left_label_w - 0.007
        track_gap = 0.005
        track_w = (left_data_w - 3 * track_gap) / 4

        for k, (metric, title, _) in enumerate(METRICS):
            ax = fig.add_axes([
                left_data_x + k * (track_w + track_gap),
                plot_y, track_w, plot_h,
            ])
            means = df[f"{metric}_mean"].to_numpy(float)
            sds = df[f"{metric}_sd"].to_numpy(float)
            lim_lo, lim_hi = metric_limits(means, sds)
            ax.axvline(float(ref[f"{metric}_mean"]), color=TEAL,
                       linewidth=0.9, ls=":", alpha=.8, zorder=1)

            for i, row in df.iterrows():
                mean = float(row[f"{metric}_mean"])
                sd = float(row[f"{metric}_sd"])
                color = TEAL if row["model"] == REFERENCE else GRAY
                if np.isfinite(sd) and sd > 0:
                    ax.errorbar(mean, i, xerr=sd, fmt="none", color=color,
                                elinewidth=0.9, capsize=1.7, capthick=0.75,
                                zorder=3)
                ax.scatter(mean, i, color=color, s=22 if color == TEAL else 15,
                           edgecolors="white", linewidths=.35, zorder=4)
                ax.axhline(i, color=GRID, linewidth=.45, zorder=0)
            ax.set_ylim(n - .55, -.75)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(title, fontsize=6.6, weight="bold", color=DARK, pad=8)
            # Numeric endpoints show each metric's own axis range.
            ax.set_xlabel(f"{lim_lo:.2f}–{lim_hi:.2f}", fontsize=5.7,
                          labelpad=4, color="#536776")
            for spine in ("top", "left", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_linewidth(.5)
            ax.spines["bottom"].set_color("#a6b3b9")

        # ---------- (b) raw mean +/- SD with directional color encoding ----------
        right_label_ax = fig.add_axes([right_x, plot_y, right_label_w, plot_h])
        right_label_ax.set_xlim(0, 1)
        right_label_ax.set_ylim(n, -0.90)
        right_label_ax.axis("off")
        right_cell_x = right_x + right_label_w
        right_cell_w = panel_w - right_label_w
        cell_ax = fig.add_axes([right_cell_x, plot_y, right_cell_w, plot_h])
        cell_ax.set_xlim(0, 4)
        cell_ax.set_ylim(n, -0.90)
        cell_ax.axis("off")

        for i, row in df.iterrows():
            is_ref = row["model"] == REFERENCE
            bg = "#eaf0f2" if is_ref else ("#fafbfc" if i % 2 else "white")
            right_label_ax.add_patch(Rectangle((0, i), 1, 1, fc=bg, ec="none"))
            right_label_ax.text(.025, i + .5, row["short_label"],
                                ha="left", va="center", fontsize=6.2,
                                color=DARK, weight="bold" if is_ref else "normal")

            for j, (metric, _, direction) in enumerate(METRICS):
                val = float(row[f"{metric}_mean"])
                sd = float(row[f"{metric}_sd"])
                reference = float(ref[f"{metric}_mean"])
                if reference == 0:
                    raise ValueError(f"ProtT5 {metric}_mean is zero; percent change undefined")
                change = direction * (val - reference) / abs(reference) * 100
                face = COLORMAP(NORM(change))
                cell_ax.add_patch(Rectangle((j, i), 1, 1, fc=face,
                                            ec="white", lw=.75))
                color = contrast_color(face)
                if np.isfinite(sd):
                    cell_ax.text(j+.5, i+.35, f"{val:.3f}", ha="center",
                                 va="center", color=color, fontsize=6.5,
                                 weight="bold" if is_ref else "normal")
                    cell_ax.text(j+.5, i+.74, f"±{sd:.3f}", ha="center",
                                 va="center", color=color, fontsize=5.6)
                else:
                    cell_ax.text(j+.5, i+.52, f"{val:.3f}", ha="center",
                                 va="center", color=color, fontsize=6.5,
                                 weight="bold" if is_ref else "normal")
            if is_ref:
                right_label_ax.add_patch(Rectangle((.005, i+.02), .99, .96,
                                                    fill=False, ec="#516875", lw=.6))
                cell_ax.add_patch(Rectangle((.005, i+.02), 3.99, .96,
                                            fill=False, ec="#516875", lw=.6))

        for j, (_, title, _) in enumerate(METRICS):
            cell_ax.text(j+.5, -.43, title, va="center", ha="center",
                         fontsize=6.45, weight="bold", color=DARK)

        # Color scale belongs only to heatmap, not to panel (a).
        cax = fig.add_axes([right_x + .089, .102, panel_w - .165, .022])
        mapper = mpl.cm.ScalarMappable(norm=NORM, cmap=COLORMAP)
        cb = fig.colorbar(mapper, cax=cax, orientation="horizontal", extend="both")
        cb.set_ticks([-15, -5, -1, 0, 1, 5, 15])
        cb.set_ticklabels(["-15", "-5", "-1", "0", "+1", "+5", "+15"])
        cb.ax.tick_params(labelsize=6, length=1.3, pad=2)
        cb.outline.set_linewidth(.4)
        cb.set_label("Change from ProtT5 (%)  ← worse  |  better →",
                     fontsize=6.4, labelpad=3)

        # Journal-friendly vector files and 600-dpi PNG.
        outputs = {}
        for ext in ("png", "pdf", "svg"):
            target = outdir / f"{filename}.{ext}"
            fig.savefig(target, dpi=dpi if ext == "png" else None,
                        facecolor="white")
            outputs[ext] = target
            print(f"Saved {target}")
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("results/ablation_results_data"))
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional direct path to 01_plm_representations.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/plots/plm_representations"))
    parser.add_argument("--filename", default="pepmetastack_plm_representations_2panel")
    parser.add_argument("--width", type=float, default=9.2)
    parser.add_argument("--height", type=float, default=2.6)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    csv_path = args.csv if args.csv else args.data_dir / "01_plm_representations.csv"
    df = read_plm_csv(csv_path)
    make_figure(df, args.output_dir, args.filename,
                width=args.width, height=args.height, dpi=args.dpi)


if __name__ == "__main__":
    '''
    
    python plot_plm_representations_fusion_2panel.py \
        --data-dir results/ablation_results_data \
        --output-dir results/plots/plm_representations \
        --width 9.2 \
        --height 2.6 \
        --dpi 600

    '''
    
    main()
