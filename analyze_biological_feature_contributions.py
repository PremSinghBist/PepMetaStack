#!/usr/bin/env python3
"""PepMetaStack: contribution of ProtT5 embeddings and biological priors.

Evaluate *predefined* representation configurations with the same five outer
validation folds as the reference StackingRegressionPipeline, without scoring
or selecting on the independent test set.

P = physicochemical descriptors
B = BLOSUM-derived descriptors
S = sinusoidal positional encodings

Experiments (9 distinct configurations):
    P+B+S                biological priors alone (no ProtT5)
    ProtT5               PLM representation alone
    ProtT5+P, ProtT5+B, ProtT5+S
    ProtT5+P+B, ProtT5+P+S, ProtT5+B+S
    ProtT5+P+B+S         full representation

Figure (single-column, two vertically aligned panels):
    (a) Five-fold CV RMSE (mean +/- SD) for key feature combinations.
    (b) Paired fold-wise change in RMSE on removing PLM/P/B/S from full.
        Positive delta = an ablation worsens RMSE.



Requires numpy, pandas, scipy, scikit-learn, matplotlib and all dependencies
of the existing pipeline. Each model is trained with its ORIGINAL settings;
stack-n-jobs controls only parallelism in the outer StackingRegressor.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# pipeline. The RidgeCV *meta-learner* is retained as in PepMetaStack.
FEATURE_KEYS = ("PLM", "P", "B", "S")
CONFIGS = {
    "priors_only": ("P", "B", "S"),
    "plm_only": ("PLM",),
    "plm_p": ("PLM", "P"),
    "plm_b": ("PLM", "B"),
    "plm_s": ("PLM", "S"),
    "without_s": ("PLM", "P", "B"),
    "without_b": ("PLM", "P", "S"),
    "without_p": ("PLM", "B", "S"),
    "full": ("PLM", "P", "B", "S"),
}

DISPLAY_CONFIGS = (
    ("priors_only", "P+B+S"),
    ("plm_only", "ProtT5"),
    ("plm_p", "ProtT5+P"),
    ("plm_b", "ProtT5+B"),
    ("plm_s", "ProtT5+S"),
    ("full", "ProtT5+P+B+S"),
)
REMOVAL_CONFIGS = (
    ("priors_only", "w/o ProtT5"),
    ("without_p", "w/o P"),
    ("without_b", "w/o B"),
    ("without_s", "w/o S"),
)
METRICS = ("RMSE", "MAE", "R2", "Pearson_R", "Spearman_R", "Kendall_Tau")


def import_reference_pipeline(path: Path):
    """Import the user's source without running its __main__ experiment."""
    path = path.expanduser().resolve() 
    if not path.is_file():
        raise FileNotFoundError(f"Reference pipeline not found: {path}")
    # Source imports data_util and the local biological-feature loader modules.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("pepmetastack_reference_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "StackingRegressionPipeline"):
        raise AttributeError(f"{path} does not define StackingRegressionPipeline")
    if not hasattr(module, "data_util") or not hasattr(module.data_util, "get_cv_split"):
        raise AttributeError("Reference module must import data_util.get_cv_split")
    return module


def prepare_component_arrays(dataset: pd.DataFrame, feature_maps: dict, split_name: str):
    if not {"sequence", "value"}.issubset(dataset.columns):
        raise ValueError(f"{split_name}: data must have sequence and value columns")
    if dataset.empty:
        raise ValueError(f"{split_name}: empty dataset")
    if dataset["sequence"].isna().any():
        raise ValueError(f"{split_name}: sequences contain nulls")
    sequences = dataset["sequence"].tolist()
    if any(not isinstance(s, str) or not s.strip() for s in sequences):
        raise ValueError(f"{split_name}: sequences must be nonempty strings")
    y = pd.to_numeric(dataset["value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"{split_name}: nonfinite activity values")

    blocks = {}
    for name in FEATURE_KEYS:
        mapping = feature_maps[name]
        missing = [s for s in sequences if s not in mapping]
        if missing:
            raise KeyError(f"{split_name}: {name} missing for {len(missing)} peptides; examples {missing[:3]}")
        block = np.stack([np.asarray(mapping[s], dtype=np.float32).ravel() for s in sequences])
        if not np.isfinite(block).all() or block.ndim != 2 or block.shape[1] == 0:
            raise ValueError(f"{split_name}: {name} features are invalid")
        blocks[name] = block
    return blocks, y


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = (float(np.corrcoef(y_true, y_pred)[0, 1])
               if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman = float(spearmanr(y_true, y_pred).statistic)
        kendall = float(kendalltau(y_true, y_pred).statistic)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_R": pearson,
        "Spearman_R": spearman,
        "Kendall_Tau": kendall,
    }


def run_experiments(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    module = import_reference_pipeline(args.pipeline_script)
    cls = module.StackingRegressionPipeline

    # All representations loaded once, using the EXACT loaders in your pipeline.
    pipe = cls(
        embedding_files={"prott5": args.embedding_file},
        physio_path=args.physio_file,
        blosum_path=args.blosum_file,
        sinu_path=args.sinu_file,
        random_state=args.seed,
        use_physio=True,
        use_blosum=True,
        use_sinusoidal=True,
        use_pfeature=False,
    )
    feature_maps = {
        "PLM": pipe.embeddings["prott5"],
        "P": pipe.extra_features["physio"],
        "B": pipe.extra_features["blosum"],
        "S": pipe.extra_features["sinusoidal"],
    }

    fold_records: list[dict] = []
    prediction_records: list[dict] = []
    test_identity = None
    development_identity = None
    for fold in range(1, args.folds + 1):
        train, validation, test = module.data_util.get_cv_split(
            input_csv_path=str(args.dataset_csv), fold=fold
        )
        train = train.copy().reset_index(drop=True)
        validation = validation.copy().reset_index(drop=True)
        test = test.copy().reset_index(drop=True)

        # Guard against changes in the reference split across ablation folds.
        current_test = Counter(zip(test["sequence"].astype(str), test["value"].astype(str)))
        current_development = Counter(zip(
            pd.concat([train, validation])["sequence"].astype(str),
            pd.concat([train, validation])["value"].astype(str),
        ))
        if test_identity is None:
            test_identity, development_identity = current_test, current_development
        elif test_identity != current_test or development_identity != current_development:
            raise ValueError("Test/development membership changed across folds")

        dev_sequences = set(train["sequence"].astype(str)) | set(validation["sequence"].astype(str))
        if dev_sequences & set(test["sequence"].astype(str)):
            raise ValueError("Exact sequence overlap with independent test data")
        fold_overlap = set(train["sequence"].astype(str)) & set(validation["sequence"].astype(str))
        if fold_overlap:
            warnings.warn(
                f"Fold {fold}: {len(fold_overlap)} identical sequences occur in "
                "both training and validation; check split leakage.", stacklevel=2
            )

        train_blocks, y_train = prepare_component_arrays(train, feature_maps, f"fold {fold} training")
        val_blocks, y_val = prepare_component_arrays(validation, feature_maps, f"fold {fold} validation")
        print(f"\nOuter fold {fold}/{args.folds}: {len(y_train)} train, {len(y_val)} validation")

        for config, components in CONFIGS.items():
            X_train = np.concatenate([train_blocks[k] for k in components], axis=1)
            X_val = np.concatenate([val_blocks[k] for k in components], axis=1)
            # Fresh model every configuration and every outer fold. All hyperparameters
            # match the provided reference class; RidgeCV remains the meta-learner.
            model = pipe.build_stacking_ensemble()
            if args.stack_n_jobs is not None and hasattr(model, "set_params"):
                model.set_params(n_jobs=args.stack_n_jobs)
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_val), dtype=float).ravel()
            if len(pred) != len(y_val) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid predictions for {config} in fold {fold}")
            metrics = score(y_val, pred)
            fold_records.append({
                "Fold": fold, "configuration": config,
                "components": "+".join(components), "n_validation": len(y_val),
                "n_features": X_train.shape[1], **metrics,
            })
            prediction_records.extend({
                "Fold": fold, "configuration": config,
                "sequence": s, "y_true": float(t), "y_pred": float(p),
            } for s, t, p in zip(validation["sequence"], y_val, pred))
            print(f"  {config:<14} dim={X_train.shape[1]:4d}  "
                  f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")

    fold_df = pd.DataFrame.from_records(fold_records)
    pred_df = pd.DataFrame.from_records(prediction_records)
    return fold_df, pred_df


def make_summaries(folds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Fold", "configuration", *METRICS}
    missing = required - set(folds_df.columns)
    if missing:
        raise ValueError(f"Fold metric CSV lacks: {sorted(missing)}")
    if folds_df.duplicated(["Fold", "configuration"]).any():
        raise ValueError("Duplicate fold/configuration metrics")
    pivot = folds_df.pivot(index="Fold", columns="configuration", values="RMSE")
    if set(pivot.columns) != set(CONFIGS) or pivot.isna().any().any():
        raise ValueError("All nine configurations must be evaluated in every fold")
    if len(pivot) < 2:
        raise ValueError("At least two paired validation folds are needed for SD")

    summary = []
    for config, components in CONFIGS.items():
        subset = folds_df.loc[folds_df["configuration"] == config]
        row = {"configuration": config, "components": "+".join(components),
               "folds": int(len(subset))}
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
        summary.append(row)
    summary_df = pd.DataFrame(summary)

    paired = []
    reference = pivot["full"].to_numpy(float)
    for config, removed_label in REMOVAL_CONFIGS:
        difference = pivot[config].to_numpy(float) - reference
        paired.append({
            "removed": removed_label, "configuration": config,
            "n_folds": len(difference),
            "delta_RMSE_mean": float(np.mean(difference)),
            "delta_RMSE_sd": float(np.std(difference, ddof=1)),
            "fold_deltas": ";".join(f"{v:.7f}" for v in difference),
        })
    paired_df = pd.DataFrame(paired)
    return summary_df, paired_df


def plot_figure(summary_df: pd.DataFrame, paired_df: pd.DataFrame,
                output_dir: Path, *, filename: str, width: float, height: float,
                dpi: int) -> dict[str, Path]:
    if width <= 0 or height <= 0 or dpi <= 300:
        raise ValueError("Width/height must be positive; dpi must be > 300")
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = summary_df.set_index("configuration")
    removal = paired_df.set_index("configuration")
    if not set(CONFIGS).issubset(lookup.index):
        raise ValueError("Summary is missing feature configurations")

    settings = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.65, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": dpi,
    }
    with plt.rc_context(settings):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(width, height),
            gridspec_kw={"height_ratios": [1.07, 0.85]},
        )

        # (a) Overall predictive accuracy of predefined representations.
        labels = [label for _, label in DISPLAY_CONFIGS]
        means = np.array([lookup.loc[c, "RMSE_mean"] for c, _ in DISPLAY_CONFIGS], float)
        sds = np.array([lookup.loc[c, "RMSE_sd"] for c, _ in DISPLAY_CONFIGS], float)
        y = np.arange(len(labels))
        ax1.axvline(
            float(lookup.loc["plm_only", "RMSE_mean"]),
            linestyle="--", linewidth=0.75, color="0.7", zorder=1,
        )
        for i, (mean, sd) in enumerate(zip(means, sds)):
            is_full = DISPLAY_CONFIGS[i][0] == "full"
            color = "#20668D" if is_full else "#6A8494"
            ax1.errorbar(
                mean, i, xerr=sd, fmt="o", color=color,
                markersize=5.3 if is_full else 4.2,
                markeredgecolor="white", markeredgewidth=0.45,
                capsize=2.5, capthick=0.8, elinewidth=0.8,
                zorder=4 if is_full else 3,
            )
        ax1.set_yticks(y, labels=labels)
        ax1.invert_yaxis()
        span = max(float(np.ptp(means)), 0.005)
        lo = float(np.min(means - sds)) - max(0.07 * span, 0.005)
        hi = float(np.max(means + sds)) + max(0.08 * span, 0.008)
        ax1.set_xlim(lo, hi)
        ax1.set_xlabel("Validation RMSE (lower is better)")
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes,
                 va="top", ha="left", fontsize=8, fontweight="bold",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 0.7})
        ax1.grid(axis="x", color="0.88", linestyle=":", linewidth=0.45, zorder=0)

        # (b) Paired fold-wise increase in RMSE when removing a feature.
        removed = [label for _, label in REMOVAL_CONFIGS]
        delta = np.array([removal.loc[c, "delta_RMSE_mean"] for c, _ in REMOVAL_CONFIGS], float)
        delta_sd = np.array([removal.loc[c, "delta_RMSE_sd"] for c, _ in REMOVAL_CONFIGS], float)
        y2 = np.arange(len(removed))
        colors = ["#20668D" if d >= 0 else "#B16C54" for d in delta]
        ax2.barh(y2, delta, height=0.58, color=colors, alpha=0.88, zorder=3)
        ax2.errorbar(delta, y2, xerr=delta_sd, fmt="none", ecolor="#292929",
                     elinewidth=0.85, capsize=2.2, capthick=0.75, zorder=4)
        ax2.axvline(0, linewidth=0.9, color="#333333", zorder=5)
        ax2.set_yticks(y2, labels=removed)
        ax2.invert_yaxis()
        # Preserve true zero without wasting half the plotting region when
        # removing ProtT5 has a much larger effect than removing P, B or S.
        min_effect = min(0.0, float(np.min(delta - delta_sd)))
        max_effect = max(0.0, float(np.max(delta + delta_sd)))
        data_span = max(max_effect - min_effect, 0.006)
        ax2.set_xlim(min_effect - 0.07 * data_span,
                     max_effect + 0.19 * data_span)
        # Numeric labels ensure small effects stay interpretable next to PLM.
        label_offset = 0.012 * data_span
        for index, effect in enumerate(delta):
            ax2.text(effect + (label_offset if effect >= 0 else -label_offset),
                     index, f"{effect:+.3f}", fontsize=6.2,
                     va="center", ha="left" if effect >= 0 else "right",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                           "pad": 0.2}, zorder=6)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax2.set_xlabel(r"$\Delta$RMSE vs. full (positive = worse)")
        ax2.text(0.0, 1.02, "(b)", transform=ax2.transAxes,
                 va="bottom", ha="left", fontsize=8, fontweight="bold",
                 clip_on=False,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 0.7})
        ax2.grid(axis="x", color="0.88", linestyle=":", linewidth=0.45, zorder=0)

        for ax in (ax1, ax2):
            ax.tick_params(direction="out", length=2.5, width=0.65)
            ax.tick_params(axis="y", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Same explicit subplot alignment method as previous PepMetaStack figures.
        fig.subplots_adjust(left=0.34, right=0.97, top=0.97,
                            bottom=0.12, hspace=0.47)
        fig.canvas.draw()
        pos1, pos2 = ax1.get_position(), ax2.get_position()
        ax2.set_position([pos1.x0, pos2.y0, pos1.width, pos2.height])

        outputs = {}
        for extension in ("png", "pdf", "svg"):
            path = output_dir / f"{filename}.{extension}"
            fig.savefig(path, dpi=dpi, facecolor="white")
            outputs[extension] = path
        plt.close(fig)
    return outputs


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pipeline-script", type=Path, default=Path("stacking_regression_pipeline.py"),
                        help="Python file defining your existing StackingRegressionPipeline")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/ecoli_five_fold_dataset.csv"))
    parser.add_argument("--embedding-file", type=Path,
                        default=Path("data/prott5/prot_t5_xl_uniref50_ecoli.npz"))
    parser.add_argument("--physio-file", type=Path, default=Path("data/ecoli_phyiochem.csv"))
    parser.add_argument("--blosum-file", type=Path, default=Path("data/ecoli_blosum62_features.csv"))
    parser.add_argument("--sinu-file", type=Path, default=Path("data/ecoli_sinusoidal_encoding.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/biological_feature_analysis"))
    parser.add_argument("--filename", default="figure_biological_feature_contributions")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stack-n-jobs", type=int, default=1,
                        help="Parallelism of StackingRegressor (1 avoids nested overload)")
    parser.add_argument("--width", type=float, default=3.35)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--plot-only", action="store_true",
                        help="Read prior fold metrics and regenerate figure without training")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.output_dir / "biological_feature_cv_folds.csv"
    if args.plot_only:
        if not folds_path.is_file():
            raise FileNotFoundError(f"Cannot plot without saved fold metrics: {folds_path}")
        folds_df = pd.read_csv(folds_path)
    else:
        folds_df, predictions = run_experiments(args)
        folds_df.to_csv(folds_path, index=False)
        predictions.to_csv(args.output_dir / "biological_feature_oof_predictions.csv", index=False)

    summary, paired = make_summaries(folds_df)
    summary.to_csv(args.output_dir / "biological_feature_cv_summary.csv", index=False)
    paired.to_csv(args.output_dir / "biological_feature_paired_delta_rmse.csv", index=False)
    paths = plot_figure(summary, paired, args.output_dir,
                        filename=args.filename, width=args.width, height=args.height, dpi=args.dpi)
    print("\nRMSE (five-fold mean +/- SD):")
    for row in summary.itertuples():
        print(f"  {row.configuration:14s}: {row.RMSE_mean:.4f} +/- {row.RMSE_sd:.4f}")
    print("\nChange from full model on removing each component (paired fold RMSE):")
    for row in paired.itertuples():
        print(f"  {row.removed:12s}: {row.delta_RMSE_mean:+.4f} +/- {row.delta_RMSE_sd:.4f}")
    print("\n".join(f"Saved {format.upper()}: {path}" for format, path in paths.items()))


if __name__ == "__main__":
    '''
        Usage:
            python analyze_biological_feature_contributions.py \
            --pipeline-script ./ridge_regression_embedding_guided.py \
            --dataset-csv ./data/ecoli_five_fold_dataset.csv \
            --embedding-file ./data/prott5/prot_t5_xl_uniref50_ecoli.npz \
            --physio-file ./data/ecoli_phyiochem.csv \
            --blosum-file ./data/ecoli_blosum62_features.csv \
            --sinu-file ./data/ecoli_sinusoidal_encoding.csv \
            --output-dir ./results/plots/biological_feature_analysis \
            --dpi 600 --width 3.35 --height 5.0

        To regenerate the figure without retraining:
            python analyze_biological_feature_contributions.py \\
            --plot-only \\
            --output-dir ./results/plots/biological_feature_analysis

        Outputs:
            biological_feature_cv_folds.csv
            biological_feature_cv_summary.csv
            biological_feature_paired_delta_rmse.csv
            biological_feature_oof_predictions.csv   (after training)
            figure_biological_feature_contributions.{png,pdf,svg}
    
    '''
    
    cli()
