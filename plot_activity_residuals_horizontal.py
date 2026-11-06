
#!/usr/bin/env python3
"""Publication-quality horizontal predicted-vs-experimental
and residual plots.

Input CSV columns: y_true, y_pred
Output: PNG, PDF, SVG
Residual = experimental - predicted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, r2_score


def plot_activity_residuals(
    prediction_csv: str | Path,
    output_dir: str | Path,
    *,
    filename: str = "experimental_predicted_residuals",
    activity_label: str = "Activity",
    width_in: float = 7.0,
    height_in: float = 3.1,
    dpi: int = 600,
) -> dict[str, Path]:

    prediction_csv = Path(prediction_csv)
    output_dir = Path(output_dir)

    if width_in <= 0 or height_in <= 0:
        raise ValueError("Figure dimensions must be positive.")

    if dpi <= 300:
        raise ValueError("DPI must be greater than 300.")

    if not prediction_csv.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {prediction_csv}"
        )

    df = pd.read_csv(prediction_csv)

    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain {sorted(required)}"
        )

    actual = pd.to_numeric(
        df["y_true"], errors="raise"
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        df["y_pred"], errors="raise"
    ).to_numpy(dtype=float)

    if len(actual) < 2:
        raise ValueError("At least 2 data points are needed.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Nonfinite values found.")

    residuals = actual - predicted

    rmse = float(
        np.sqrt(mean_squared_error(actual, predicted))
    )
    r2 = float(r2_score(actual, predicted))

    pearson = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )

    settings = {
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif", "Times New Roman", "Times"
        ],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": dpi,
        "svg.fonttype": "none",
    }

    with plt.rc_context(settings):

        # HORIZONTAL TWO-PANEL LAYOUT
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(width_in, height_in),
            gridspec_kw={"width_ratios": [1, 1]},
        )

        color = "#20668D"

        # -------------------------------------------
        # (a) Experimental vs Predicted
        # -------------------------------------------
        ax1.scatter(
            actual,
            predicted,
            s=10,
            alpha=0.60,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))

        span = max(hi - lo, 1e-9)
        pad = 0.05 * span
        bounds = (lo - pad, hi + pad)

        ax1.plot(
            bounds,
            bounds,
            color="#333333",
            linestyle="--",
            linewidth=0.95,
            zorder=3,
        )

        ax1.set_xlim(bounds)
        ax1.set_ylim(bounds)

        # Equal scaling for ideal prediction line
        ax1.set_aspect("equal", adjustable="box")

        ax1.set_xlabel(
            f"Experimental {activity_label}"
        )
        ax1.set_ylabel(
            f"Predicted {activity_label}"
        )

        ax1.text(
            0.03, 1.0, "(a)",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

        stats = (
            f"$R^2$ = {r2:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"$r$ = {pearson:.3f}"
        )

        ax1.text(
            0.97,
            0.05,
            stats,
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "0.80",
                "linewidth": 0.5,
                "alpha": 0.95,
            },
        )

        # -------------------------------------------
        # (b) Predicted vs Residuals
        # -------------------------------------------
        ax2.scatter(
            predicted,
            residuals,
            s=10,
            alpha=0.60,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

        ax2.axhline(
            0,
            color="#333333",
            linewidth=0.95,
            linestyle="--",
            zorder=3,
        )

        ax2.set_xlabel(
            f"Predicted {activity_label}"
        )
        ax2.set_ylabel(
            "Residual (Exp. − Pred.)"
        )

        ax2.text(
            0.03, 1.0, "(b)",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

        max_abs_resid = float(
            np.max(np.abs(residuals))
        )

        ylim = max(1.1 * max_abs_resid, 1e-6)

        ax2.set_ylim(-ylim, ylim)
        ax2.set_xlim(bounds)

        # -------------------------------------------
        # Shared axis formatting
        # -------------------------------------------
        for ax in (ax1, ax2):

            ax.tick_params(
                direction="out",
                length=2.5,
                width=0.65,
            )

            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=5)
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=5)
            )

            ax.grid(
                True,
                color="0.88",
                linestyle=":",
                linewidth=0.45,
                zorder=0,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # -------------------------------------------
        # Horizontal journal layout
        # -------------------------------------------
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.96,
            bottom=0.20,
            wspace=0.36,
        )

        # Match vertical position and height.
        # Keep equal aspect ratio for panel (a).
        fig.canvas.draw()

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()

        ax2.set_position([
            pos2.x0,
            pos1.y0,
            pos2.width,
            pos1.height,
        ])

        # -------------------------------------------
        # Save figures
        # -------------------------------------------
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        outputs = {}

        for ext in ("png", "pdf", "svg"):

            path = output_dir / f"{filename}.{ext}"

            fig.savefig(
                path,
                dpi=dpi,
                facecolor="white",
            )

            outputs[ext] = path

        plt.close(fig)

    print(f"Independent predictions: n={len(actual)}")
    print(
        f"RMSE = {rmse:.4f} | "
        f"R² = {r2:.4f} | "
        f"Pearson r = {pearson:.4f}"
    )

    for ext, path in outputs.items():
        print(f"Saved {ext.upper()}: {path}")

    return outputs


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--filename",
        default="experimental_predicted_residuals",
    )

    parser.add_argument(
        "--activity-label",
        default="Activity",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--width",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--height",
        type=float,
        default=3.1,
    )

    args = parser.parse_args()

    plot_activity_residuals(
        prediction_csv=args.input,
        output_dir=(
            args.output_dir
            or args.input.parent / "figures"
        ),
        filename=args.filename,
        activity_label=args.activity_label,
        dpi=args.dpi,
        width_in=args.width,
        height_in=args.height,
    )


if __name__ == "__main__":
    '''
    python plot_activity_residuals_horizontal.py \
    --input ./results/PepMetaStack_ecoli_with_emb_physio_blosum_sinu/stacking_final_independent_predictions.csv \
    --output-dir ./results/plots/predicted_residuals/horizontal \
    --activity-label "log10 MIC" \
    --dpi 600 \
    --width 7.0 \
    --height 2.4
    '''
    main()
