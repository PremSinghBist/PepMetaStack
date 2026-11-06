#!/usr/bin/env python3
"""Plot only the PepMetaStack PLM representation/fusion heatmap.

Colors encode directional percentage change relative to ProtT5 (positive = better),
while cells display absolute cross-validation mean and standard deviation.

Example:
    python plot_plm_representations_heatmap.py \
        --data-dir results/ablation_results_data \
        --output-dir results/plots/plm_representations \
        --width 6.6 --height 3.5 --dpi 600
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
DARK = "#253b49"
COLORMAP = LinearSegmentedColormap.from_list(
    "plm_delta", ["#ad4949", "#efc6c0", "#f8f8f6", "#b9dadd", "#126b7a"]
)
NORM = SymLogNorm(linthresh=0.65, linscale=0.85, vmin=-15, vmax=15,
                  base=10, clip=True)


def read_plm_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing PLM CSV: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["model"] + [f"{metric}_mean" for metric, _, _ in METRICS]
    missing = [column for column in required if column not in df.columns]
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


def contrast_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.54 else "#1b2b35"


def make_figure(df: pd.DataFrame, outdir: Path, filename: str,
                width: float = 6.6, height: float = 3.5,
                dpi: int = 600) -> dict[str, Path]:
    if width <= 0 or height <= 0 or dpi <= 0:
        raise ValueError("Width, height and dpi must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    ref = df.loc[df["model"] == REFERENCE].iloc[0]
    n = len(df)
    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "figure.facecolor": "white",
    }):
        fig = plt.figure(figsize=(width, height), facecolor="white")
        # Single heatmap: labels on left, four metric cells on right.
        left_x, label_w = 0.04, 0.32
        plot_y, plot_h = 0.25, 0.64
        cell_x, cell_w = left_x + label_w, 0.60
        label_ax = fig.add_axes([left_x, plot_y, label_w, plot_h])
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(n, -0.90)
        label_ax.axis("off")
        cell_ax = fig.add_axes([cell_x, plot_y, cell_w, plot_h])
        cell_ax.set_xlim(0, len(METRICS))
        cell_ax.set_ylim(n, -0.90)
        cell_ax.axis("off")

        for i, row in df.reset_index(drop=True).iterrows():
            is_ref = row["model"] == REFERENCE
            bg = "#eaf0f2" if is_ref else ("#fafbfc" if i % 2 else "white")
            label_ax.add_patch(Rectangle((0, i), 1, 1, fc=bg, ec="none"))
            label_ax.text(.025, i+.5, row["short_label"],
                          ha="left", va="center", fontsize=8,
                          color=DARK, weight="bold" if is_ref else "normal")
            for j, (metric, _, direction) in enumerate(METRICS):
                val = float(row[f"{metric}_mean"])
                sd = float(row[f"{metric}_sd"])
                baseline = float(ref[f"{metric}_mean"])
                if baseline == 0:
                    raise ValueError(f"ProtT5 {metric}_mean is zero; percent change undefined")
                pct_change = direction * (val - baseline) / abs(baseline) * 100
                face = COLORMAP(NORM(pct_change))
                cell_ax.add_patch(Rectangle((j, i), 1, 1, fc=face, ec="white", lw=.8))
                fg = contrast_color(face)
                if np.isfinite(sd):
                    cell_ax.text(j+.5, i+.35, f"{val:.3f}", ha="center",
                                 va="center", fontsize=8, color=fg,
                                 weight="bold" if is_ref else "normal")
                    cell_ax.text(j+.5, i+.75, f"±{sd:.3f}", ha="center",
                                 va="center", fontsize=7.3, color=fg)
                else:
                    cell_ax.text(j+.5, i+.5, f"{val:.3f}", ha="center",
                                 va="center", fontsize=8, color=fg,
                                 weight="bold" if is_ref else "normal")
            if is_ref:
                label_ax.add_patch(Rectangle((.005, i+.02), .99, .96,
                                             fill=False, ec="#516875", lw=.7))
                cell_ax.add_patch(Rectangle((.005, i+.02), 3.99, .96,
                                            fill=False, ec="#516875", lw=.7))

        for j, (_, title, _) in enumerate(METRICS):
            cell_ax.text(j+.5, -.43, title, va="center", ha="center",
                         fontsize=8, weight="bold", color=DARK)

        cax = fig.add_axes([0.27, 0.105, 0.49, 0.035])
        mapper = mpl.cm.ScalarMappable(norm=NORM, cmap=COLORMAP)
        cb = fig.colorbar(mapper, cax=cax, orientation="horizontal", extend="both")
        cb.set_ticks([-15, -5, -1, 0, 1, 5, 15])
        cb.set_ticklabels(["-15", "-5", "-1", "0", "+1", "+5", "+15"])
        cb.ax.tick_params(labelsize=7, length=1.5, pad=2)
        cb.outline.set_linewidth(.5)
        cb.set_label("Change from ProtT5 (%)   ← worse  |  better →",
                     fontsize=8, labelpad=4)

        outputs = {}
        for ext in ("png", "pdf", "svg"):
            target = outdir / f"{filename}.{ext}"
            fig.savefig(target, dpi=dpi if ext == "png" else None,
                        facecolor="white", bbox_inches="tight", pad_inches=.12)
            outputs[ext] = target
            print(f"Saved {target}")
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("results/ablation_results_data"))
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional direct path to 01_plm_representations.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/plots/plm_representations"))
    parser.add_argument("--filename", default="pepmetastack_plm_heatmap")
    parser.add_argument("--width", type=float, default=6.6)
    parser.add_argument("--height", type=float, default=3.5)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    csv_path = args.csv if args.csv else args.data_dir / "01_plm_representations.csv"
    make_figure(read_plm_csv(csv_path), args.output_dir, args.filename,
                width=args.width, height=args.height, dpi=args.dpi)


if __name__ == "__main__":
    '''
    python plot_plm_representations_heatmap.py \
    --data-dir results/ablation_results_data \
    --output-dir results/plots/plm_representations \
    --width 6.6 \
    --height 2.6 \
    --dpi 600
    '''
    main()
