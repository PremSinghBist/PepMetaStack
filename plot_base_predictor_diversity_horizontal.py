
#!/usr/bin/env python3
"""Figure 3: Base-predictor diversity and complementary predictions.

Horizontal (1 x 2) journal figure: (c) left, (d) right.

Uses held-out predictions from five-fold cross-validation.

Expected directory structure:
    A1_individual/
        svr/fold_1_predictions.csv
        extra_trees/fold_1_predictions.csv
        hist_gb/fold_1_predictions.csv
        knn/fold_1_predictions.csv
        mlp/fold_1_predictions.csv
        ... similarly for folds 2–5.

Each file requires: sequence, y_true, y_pred
Optional: Fold

(c) Compact lower-triangular residual correlation heatmap.
(d) Percentage of peptides predicted more accurately than SVR.

Excludes Ridge and XGBoost.

Outputs PNG (600 DPI), PDF, SVG, and analysis CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Models included in the diversity analysis
# ---------------------------------------------------------
MODELS = {
    "svr": "SVR",
    "extra_trees": "Extra Trees",
    "hist_gb": "HistGB",
    "knn": "KNN",
    "mlp": "MLP",
}

BASELINE = "svr"


# ---------------------------------------------------------
# 1. Load and align held-out predictions
# ---------------------------------------------------------
def load_aligned_validation_predictions(
    input_root: Path,
    n_folds: int
) -> pd.DataFrame:

    all_folds = []

    for fold in range(1, n_folds + 1):

        joined = None

        for model in MODELS:

            path = (
                input_root
                / "A1_individual"
                / model
                / f"fold_{fold}_predictions.csv"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing prediction file: {path}"
                )

            data = pd.read_csv(path)

            required = {"sequence", "y_true", "y_pred"}
            missing = required - set(data.columns)

            if missing:
                raise ValueError(
                    f"{path}: missing columns {sorted(missing)}"
                )

            if len(data) < 2:
                raise ValueError(
                    f"{path}: insufficient validation samples"
                )

            if data["sequence"].isna().any():
                raise ValueError(
                    f"{path}: missing peptide sequences"
                )

            if "Fold" in data.columns:
                fold_values = pd.to_numeric(
                    data["Fold"], errors="coerce"
                ).to_numpy()

                if not np.all(fold_values == fold):
                    raise ValueError(
                        f"{path}: incorrect Fold column"
                    )

            y_true = pd.to_numeric(
                data["y_true"], errors="raise"
            ).to_numpy(dtype=float)

            y_pred = pd.to_numeric(
                data["y_pred"], errors="raise"
            ).to_numpy(dtype=float)

            if not (
                np.isfinite(y_true).all()
                and np.isfinite(y_pred).all()
            ):
                raise ValueError(
                    f"{path}: nonfinite predictions or targets"
                )

            sequences = data["sequence"].astype(str).to_numpy()

            if joined is None:

                joined = pd.DataFrame({
                    "Fold": fold,
                    "sequence": sequences,
                    "y_true": y_true,
                    model: y_pred,
                })

            else:

                if len(joined) != len(data):
                    raise ValueError(
                        f"{path}: different number of samples"
                    )

                if not np.array_equal(
                    joined["sequence"].to_numpy(),
                    sequences
                ):
                    raise ValueError(
                        f"{path}: sequence ordering differs"
                    )

                if not np.allclose(
                    joined["y_true"].to_numpy(),
                    y_true,
                    atol=1e-7,
                    rtol=1e-6
                ):
                    raise ValueError(
                        f"{path}: experimental targets differ"
                    )

                joined[model] = y_pred

        all_folds.append(joined)

        print(
            f"Fold {fold}: {len(joined)} aligned peptides"
        )

    return pd.concat(all_folds, ignore_index=True)


# ---------------------------------------------------------
# 2. Compute residual correlations and complementarity
# ---------------------------------------------------------
def analyze_diversity(
    aligned: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    truth = aligned["y_true"].to_numpy(dtype=float)

    # Signed residual = experimental - predicted
    errors = np.column_stack([
        truth - aligned[model].to_numpy(dtype=float)
        for model in MODELS
    ])

    # Pairwise Pearson correlations of residual vectors
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(
            errors,
            rowvar=False
        )

    corr_df = pd.DataFrame(
        correlation,
        index=list(MODELS),
        columns=list(MODELS)
    )

    corr_df.index.name = "model"

    # Compare each predictor with SVR peptide by peptide
    baseline_error = np.abs(
        truth - aligned[BASELINE].to_numpy(dtype=float)
    )

    rows = []

    for model in MODELS:

        if model == BASELINE:
            continue

        candidate_error = np.abs(
            truth - aligned[model].to_numpy(dtype=float)
        )

        # Positive difference means candidate is more accurate
        difference = baseline_error - candidate_error

        tolerance = 1e-10

        rows.append({
            "model": model,
            "label": MODELS[model],
            "n_peptides": len(truth),
            "wins_over_svr_pct":
                100.0 * np.mean(difference > tolerance),
            "losses_to_svr_pct":
                100.0 * np.mean(difference < -tolerance),
            "ties_pct":
                100.0 * np.mean(
                    np.abs(difference) <= tolerance
                ),
            "mean_abs_error":
                float(np.mean(candidate_error)),
            "svr_mean_abs_error":
                float(np.mean(baseline_error)),
            "mean_abs_error_difference_svr_minus_model":
                float(np.mean(difference)),
        })

    return corr_df, pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. Generate publication-quality figure
# ---------------------------------------------------------
def plot_diversity(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    filename: str = "figure3_base_predictor_diversity",
    width_in: float = 7.2,
    height_in: float = 3.5,
    dpi: int = 600,
    folds: int = 5,
) -> dict[str, Path]:

    if width_in <= 0 or height_in <= 0:
        raise ValueError(
            "Figure dimensions must be positive"
        )

    if dpi <= 300:
        raise ValueError(
            "DPI must be greater than 300"
        )

    if folds < 1:
        raise ValueError(
            "Number of folds must be positive"
        )

    input_root = Path(input_root)
    output_dir = Path(output_dir)

    aligned = load_aligned_validation_predictions(
        input_root, folds
    )

    corr_df, win_df = analyze_diversity(aligned)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    settings = {
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times"
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

        # Side-by-side layout: heatmap (c) left, SVR complementarity (d) right.
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(width_in, height_in),
            gridspec_kw={"width_ratios": [1.0, 1.0]},
        )

        # =================================================
        # Panel (c): Compact lower-triangular heatmap
        # =================================================

        corr_array = corr_df.to_numpy(dtype=float)

        names = [
            "SVR", "ET", "HGB", "KNN", "MLP"
        ]

        # Remove original first row (SVR) and last
        # column (MLP).
        #
        # Original: 5 x 5
        # Result:   4 x 4
        #
        # All 10 unique pairwise correlations remain.
        corr_plot = corr_array[1:, :-1]

        n = len(names) - 1

        # Mask strict upper triangle.
        #
        # k=1 is important here:
        # The diagonal of the shifted 4x4 matrix contains
        # correlations between DISTINCT predictors.
        mask = np.triu(
            np.ones_like(corr_plot, dtype=bool),
            k=1
        )

        mask |= ~np.isfinite(corr_plot)

        masked = np.ma.array(
            corr_plot,
            mask=mask
        )

        palette = plt.get_cmap("RdBu_r").copy()
        palette.set_bad("white")

        ax1.imshow(
            masked,
            cmap=palette,
            vmin=-1,
            vmax=1,
            interpolation="none",
            aspect="auto",
            rasterized=True,
            zorder=1,
        )

        # Column labels: SVR, ET, HGB, KNN
        ax1.set_xticks(
            np.arange(n),
            labels=names[:-1]
        )

        # Row labels: ET, HGB, KNN, MLP
        ax1.set_yticks(
            np.arange(n),
            labels=names[1:]
        )

        ax1.set_xlim(-0.5, n - 0.5)
        ax1.set_ylim(n - 0.5, -0.5)

        ax1.tick_params(
            axis="both",
            which="both",
            length=0
        )

        # Annotate only the 10 visible correlations
        for i in range(n):

            for j in range(i + 1):

                value = corr_plot[i, j]

                label = (
                    f"{value:.2f}"
                    if np.isfinite(value)
                    else "NA"
                )

                text_color = (
                    "white"
                    if np.isfinite(value)
                    and abs(value) >= 0.65
                    else "#222222"
                )

                ax1.text(
                    j, i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=text_color,
                    zorder=4,
                )

                # Draw only populated cell boundaries
                ax1.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=3,
                    )
                )

        # Place panel label in unused upper-right area
        ax1.text(
            0.98, 0.98,
            "(a)",
            transform=ax1.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

        ax1.set_xlabel("Base predictor (Column)")
        ax1.set_ylabel("Base predictor (Row)")

        for spine in ax1.spines.values():
            spine.set_visible(False)

        # =================================================
        # Panel (d): Complementarity relative to SVR
        # =================================================

        ordered = win_df.sort_values(
            "wins_over_svr_pct",
            ascending=True
        )

        y = np.arange(len(ordered))

        values = ordered[
            "wins_over_svr_pct"
        ].to_numpy(dtype=float)

        ax2.barh(
            y,
            values,
            height=0.58,
            color="#20668D",
            alpha=0.88,
            linewidth=0,
            zorder=3,
        )

        short_names = {
            "Extra Trees": "ET",
            "HistGB": "HGB"
        }

        labels = [
            short_names.get(name, name)
            for name in ordered["label"]
        ]

        ax2.set_yticks(
            y,
            labels=labels
        )

        ax2.set_xlim(0, 100)
        ax2.set_ylim(
            -0.65,
            len(y) - 0.35
        )

        ax2.set_xlabel(
            "Peptides with lower error than SVR (%)"
        )

        ax2.xaxis.set_major_locator(
            MaxNLocator(
                nbins=5,
                integer=True
            )
        )

        # Value annotations
        for yi, value in zip(y, values):

            right_text = min(
                float(value) + 1.5,
                98.0
            )

            ax2.text(
                right_text,
                yi,
                f"{value:.1f}%",
                va="center",
                fontsize=7,
            )

        ax2.text(
            0.98, 0.98,
            "(b)",
            transform=ax2.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 1,
            },
            zorder=6,
        )

        ax2.tick_params(
            direction="out",
            length=2.5,
            width=0.65
        )

        ax2.tick_params(
            axis="y",
            length=0
        )

        ax2.grid(
            axis="x",
            color="0.88",
            linestyle=":",
            linewidth=0.45,
            zorder=0
        )

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # =================================================
        # Horizontal journal layout and vertical alignment
        # =================================================

        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            top=0.965,
            bottom=0.20,
            wspace=0.36,
        )

        # The heatmap is square (4 x 4), and the two panels share
        # identical axes height. Keep the chart widths flexible.
        # Do not apply the old vertical set_position() block here.

        # =================================================
        # Save high-resolution figures
        # =================================================

        outputs = {}

        for ext in ("png", "pdf", "svg"):

            path = output_dir / f"{filename}.{ext}"

            fig.savefig(
                path,
                dpi=dpi,
                facecolor="white"
            )

            outputs[ext] = path

        plt.close(fig)

    # -----------------------------------------------------
    # Save analysis data
    # -----------------------------------------------------

    aligned_file = (
        output_dir
        / f"{filename}_aligned_validation_predictions.csv"
    )

    corr_file = (
        output_dir
        / f"{filename}_residual_correlation.csv"
    )

    win_file = (
        output_dir
        / f"{filename}_complementarity_vs_svr.csv"
    )

    aligned.to_csv(
        aligned_file,
        index=False
    )

    corr_df.to_csv(corr_file)

    win_df.to_csv(
        win_file,
        index=False
    )

    print(
        f"\nValidation peptides: "
        f"{len(aligned)} across {folds} folds"
    )

    print(
        "Models: " + ", ".join(MODELS.values())
    )

    print("Excluding Ridge and XGBoost")

    print("\nComplementarity relative to SVR:")

    print(
        win_df[
            [
                "label",
                "wins_over_svr_pct",
                "mean_abs_error"
            ]
        ].to_string(index=False)
    )

    for ext, path in outputs.items():
        print(f"Saved {ext.upper()}: {path}")

    return outputs


# ---------------------------------------------------------
# 4. Command-line interface
# ---------------------------------------------------------
def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "results/base_precitors_ablation"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots")
    )

    parser.add_argument(
        "--filename",
        default="figure3_base_predictor_diversity"
    )

    parser.add_argument(
        "--width",
        type=float,
        default=7.2
    )

    parser.add_argument(
        "--height",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5
    )

    args = parser.parse_args()

    plot_diversity(
        input_root=args.input_root,
        output_dir=args.output_dir,
        filename=args.filename,
        width_in=args.width,
        height_in=args.height,
        dpi=args.dpi,
        folds=args.folds,
    )


if __name__ == "__main__":
    '''
    python plot_base_predictor_diversity_horizontal.py \
        --input-root results/base_precitors_ablation \
        --output-dir results/plots/base_predictor_diversity_horizontal \
        --width 7.2 \
        --height 2.6 \
        --dpi 600

    
    '''
    main()
