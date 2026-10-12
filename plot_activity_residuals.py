#!/usr/bin/env python3
"""Publication-sized predicted-vs-experimental and residual plots.

Reads the CSV created by StackingRegressionPipeline.save_results():
    stacking_final_independent_predictions.csv
which contains the columns y_true and y_pred.

Example:
    python plot_activity_residuals.py \
        --input results/base_predictors/base_predictor_01/stacking_final_independent_predictions.csv \
        --output-dir results/base_predictors/base_predictor_01/figures \
        --activity-label 'log10 MIC' --dpi 600

Outputs one single-column, vertically arranged two-panel figure in PNG, PDF, and SVG.
Residual = experimental - predicted (positive means underprediction).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures on servers without a display.
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
    width_in: float = 3.35,
    height_in: float = 5.0,
    dpi: int = 600,
) -> dict[str, Path]:
    """Make a single-column 2-panel journal figure from existing predictions."""
    prediction_csv = Path(prediction_csv)
    output_dir = Path(output_dir)
    if width_in <= 0 or height_in <= 0:
        raise ValueError("Figure dimensions must be positive.")
    if dpi <= 300:
        raise ValueError("DPI must be greater than 300 (recommended: 600).")
    if not prediction_csv.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    df = pd.read_csv(prediction_csv)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {sorted(required)}; found {list(df.columns)}")
    try:
        actual = pd.to_numeric(df["y_true"], errors="raise").to_numpy(dtype=float)
        predicted = pd.to_numeric(df["y_pred"], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("y_true and y_pred must be numeric") from exc
    if len(actual) < 2:
        raise ValueError("At least 2 data points are needed")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Nonfinite values found in prediction CSV")

    residuals = actual - predicted
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))
    pearson = (float(np.corrcoef(actual, predicted)[0, 1])
               if np.std(actual) > 0 and np.std(predicted) > 0 else float("nan"))

    # Journal-readable typography at the final printed width.
    settings = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
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
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(width_in, height_in),
            gridspec_kw={"height_ratios": [1.07, 0.85]},#55.7%, 44.3%
        )

        # (a) Experimental versus predicted, with perfect agreement y=x.
        ax1.scatter(
            actual, predicted, s=10, alpha=0.60,
            color="#20668D", edgecolors="none", rasterized=True,
            zorder=2,
        )
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        span = max(hi - lo, 1e-9)
        pad = 0.05 * span #5% 
        bounds = (lo - pad, hi + pad) #boundary x, y
        ax1.plot(bounds, bounds, color="#333333", linestyle="--",
                 linewidth=0.95, label="Ideal prediction", zorder=3) #forces symmetric, square axes
        ax1.set_xlim(bounds)
        ax1.set_ylim(bounds)
        ax1.set_aspect("equal", adjustable="box") 
        ax1.set_xlabel(f"Experimental {activity_label}")
        ax1.set_ylabel(f"Predicted {activity_label}")
        ax1.text(0.03, 1.0, "(a)", transform=ax1.transAxes,
                 va="top", ha="left", fontsize=8, fontweight="bold") #3% from left, 97% from bottom
        stats = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\n$r$ = {pearson:.3f}"
        ax1.text(0.97, 0.05, stats, transform=ax1.transAxes,
                 ha="right", va="bottom", fontsize=7,
                 bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                       "edgecolor": "0.80", "linewidth": 0.5, "alpha": 0.95}) #97% from left, 5% from bottom

        # (b) Residuals against predictions: residual = experimental - predicted.
        ax2.scatter(
            predicted, residuals, s=10, alpha=0.60,
            color="#20668D", edgecolors="none", rasterized=True,
            zorder=2,
        )
        ax2.axhline(0, color="#333333", linewidth=0.95,
                    linestyle="--", zorder=3)
        ax2.set_xlabel(f"Predicted {activity_label}")
        ax2.set_ylabel("Residual (Exp. − Pred.)")
        ax2.text(0.03, 1.0, "(b)", transform=ax2.transAxes,
                 va="top", ha="left", fontsize=8, fontweight="bold")#use axes coordinates instead of data coordinates
        max_abs_resid = float(np.max(np.abs(residuals)))
        ylim = max(1.1 * max_abs_resid, 1e-6) #Add 10% margin to the largest absolute residual, but at least 1e-6
        ax2.set_ylim(-ylim, ylim) #symmetric around zero
        ax2.set_xlim(bounds)

        for ax in (ax1, ax2):
            ax.tick_params(direction="out", length=2.5, width=0.65) #tick marks outside the axes, 2.5 points long, 0.65 points wide
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) #Maxbins = 5, but may be fewer if the data range is small
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(True, color="0.88", linestyle=":", linewidth=0.45, zorder=0)#grid lines, dim gray| 1->white, zorder=0 most backward 
            ax.spines["top"].set_visible(False) #hide top line of the ax (Spine: Line)
            ax.spines["right"].set_visible(False)

        fig.subplots_adjust(left=0.18, right=0.97, top=0.98,
                            bottom=0.10, hspace=0.32) #margin: left 18%,  hspace=32% of the height of the figure (vertical space between the two subplots)
        
        # Match the horizontal position and width of both subplots
        fig.canvas.draw()

        pos1 = ax1.get_position() #Fetch coordinates [x0, y0, width, height]
        pos2 = ax2.get_position()

        # Adjust the position of ax2 to match ax1's x0 and width, while keeping its y0 and height unchanged
        ax2.set_position([
            pos1.x0,
            pos2.y0,
            pos1.width,
            pos2.height
        ])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for ext in ("png", "pdf", "svg"):
            path = output_dir / f"{filename}.{ext}"
            fig.savefig(path, dpi=dpi, facecolor="white") #facecolor means the background color of fig
            outputs[ext] = path
        plt.close(fig)

    print(f"Independent predictions: n={len(actual)}")
    print(f"RMSE = {rmse:.4f} | R² = {r2:.4f} | Pearson r = {pearson:.4f}")
    print("\n".join(f"Saved {ext.upper()}: {path}" for ext, path in outputs.items()))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="CSV with y_true and y_pred columns")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: figures subdirectory beside input CSV")
    parser.add_argument("--filename", default="experimental_predicted_residuals")
    parser.add_argument("--activity-label", default="Activity",
                        help="Use the actual target scale, e.g. 'log10 MIC'")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=3.35,
                        help="Figure width in inches (LLNCS single column ~3.35)")
    parser.add_argument("--height", type=float, default=5.0)
    args = parser.parse_args()
    plot_activity_residuals(
        prediction_csv=args.input,
        output_dir=args.output_dir or args.input.parent / "figures",
        filename=args.filename,
        activity_label=args.activity_label,
        dpi=args.dpi,
        width_in=args.width,
        height_in=args.height,
    )


if __name__ == "__main__":
    '''
    python plot_activity_residuals.py \
    --input ./results/PepMetaStack_ecoli_with_emb_physio_blosum_sinu/stacking_final_independent_predictions.csv \
    --output-dir ./results/plots/predicted_residuals \
    --activity-label "log10 MIC" \
    --dpi 600
    --width 3.35 \
    --height 5.0
    
    '''
    
    main()
