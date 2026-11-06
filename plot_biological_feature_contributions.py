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

Figure (simple, journal-style, two panels):
    (c) Three horizontal bars: effect on RMSE of removing P, B or S
        individually from the full ProtT5+P+B+S representation.
        Error bars show the SD of paired five-fold RMSE differences.
    (d) One four-point line: mean five-fold validation RMSE (with SD)
        along the selected backward-elimination path, from the full
        representation down to ProtT5 alone.

The elimination path is chosen greedily using minimum mean validation RMSE
at each stage, with ProtT5 retained. It is descriptive/exploratory, not a
nested-CV estimate of selected-model generalization.

At each step, all remaining one-group removals are compared by mean
validation RMSE. The candidate with the LOWEST RMSE is selected. This is an
exploratory CV-based selection path, not an unbiased estimate of performance
for the selected representation (nested CV is needed for that).


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
from matplotlib.ticker import MaxNLocator, FuncFormatter
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


def build_backward_path(folds_df: pd.DataFrame):
    """Greedy backward elimination of P/B/S, always retaining ProtT5.

    Uses existing paired outer-fold metrics for each possible representation.
    A candidate's mean validation RMSE determines the next group to drop.
    Ties are settled in alphabetical order for reproducibility.
    """
    required = {"Fold", "configuration", "RMSE"}
    if not required.issubset(folds_df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(folds_df.columns))}")
    if folds_df.duplicated(["Fold", "configuration"]).any():
        raise ValueError("Duplicate Fold/configuration rows")
    pivot = folds_df.pivot(index="Fold", columns="configuration", values="RMSE")
    if set(CONFIGS) - set(pivot.columns) or pivot.isna().any().any():
        raise ValueError("Every predefined configuration is needed in each fold")
    if len(pivot) < 2 or not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError("Two or more complete, finite folds are needed")
    reverse = {frozenset(v): key for key, v in CONFIGS.items()}
    active = frozenset(("PLM", "P", "B", "S"))
    selected, candidates = [], []
    full_values = pivot["full"].to_numpy(float)
    previous_values = full_values
    for step in range(4):
        config = reverse[active]
        values = pivot[config].to_numpy(float)
        name = "+".join("ProtT5" if k == "PLM" else k
                        for k in FEATURE_KEYS if k in active)
        selected.append({
            "step": step, "configuration": config, "features": name,
            "removed_at_step": "none" if step == 0 else removed,
            "RMSE_mean": float(values.mean()),
            "RMSE_sd": float(values.std(ddof=1)),
            "delta_from_full_mean": float((values-full_values).mean()),
            "delta_from_previous_mean": float((values-previous_values).mean()) if step else 0.0,
        })
        if step == 3:
            break
        options = []
        for group in ("P", "B", "S"):
            if group not in active:
                continue
            remaining = active - {group}
            candidate_config = reverse[remaining]
            candidate_values = pivot[candidate_config].to_numpy(float)
            options.append((float(candidate_values.mean()), group,
                            remaining, candidate_config, candidate_values))
        options.sort(key=lambda item: (item[0], item[1]))
        best = options[0]
        for mean, group, remaining, candidate_config, candidate_values in options:
            candidates.append({
                "step": step + 1,
                "parent_configuration": config,
                "remove": group,
                "candidate_configuration": candidate_config,
                "candidate_features": "+".join(
                    "ProtT5" if k == "PLM" else k
                    for k in FEATURE_KEYS if k in remaining),
                "RMSE_mean": mean,
                "RMSE_sd": float(candidate_values.std(ddof=1)),
                "selected": group == best[1],
            })
        previous_values = values
        removed, active = best[1], best[2]
    return pd.DataFrame(selected), pd.DataFrame(candidates)


def build_step_effects(folds_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate *paired* fold differences for every chosen removal.

    This is not the difference of independent summary SDs: each difference
    uses RMSE values from the exact same outer validation fold.
    """
    pivot = folds_df.pivot(index="Fold", columns="configuration", values="RMSE")
    full = pivot[str(path_df.iloc[0]["configuration"])].to_numpy(float)
    records = []
    for step in range(1, len(path_df)):
        current = path_df.iloc[step]
        previous = path_df.iloc[step - 1]
        current_values = pivot[str(current["configuration"])].to_numpy(float)
        parent_values = pivot[str(previous["configuration"])].to_numpy(float)
        marginal = current_values - parent_values
        cumulative = current_values - full
        records.append({
            "step": step,
            "removed": str(current["removed_at_step"]),
            "from_features": str(previous["features"]),
            "to_features": str(current["features"]),
            "from_configuration": str(previous["configuration"]),
            "to_configuration": str(current["configuration"]),
            "n_folds": len(marginal),
            "delta_from_previous_mean": float(np.mean(marginal)),
            "delta_from_previous_sd": float(np.std(marginal, ddof=1)),
            "delta_from_full_mean": float(np.mean(cumulative)),
            "delta_from_full_sd": float(np.std(cumulative, ddof=1)),
            "fold_deltas_from_previous": ";".join(f"{v:.9g}" for v in marginal),
            "fold_deltas_from_full": ";".join(f"{v:.9g}" for v in cumulative),
        })
    return pd.DataFrame(records)


def plot_figure(path_df: pd.DataFrame, paired_df: pd.DataFrame,
                output_dir: Path, *, filename: str, width: float,
                height: float, dpi: int,
                panel_labels=("c", "d")) -> dict[str, Path]:
    """Two easy-to-read panels: leave-one-out impact and selected elimination path.

    (c) ΔRMSE vs full = RMSE(without group) - RMSE(full), using *paired*
        fold differences. Positive: the removal worsens predictive accuracy.
    (d) Greedy path of mean five-fold RMSE +/- between-fold SD. ProtT5 is fixed.
    """
    if width <= 0 or height <= 0 or dpi <= 0:
        raise ValueError("Width, height and dpi must be positive")
    if len(panel_labels) != 2 or any(not str(s).strip() for s in panel_labels):
        raise ValueError("Exactly two nonempty panel letters are required")
    if len(path_df) != 4:
        raise ValueError("Expected four stages: full plus three removals")

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = paired_df.set_index("removed")
    required = ("w/o P", "w/o B", "w/o S")
    if not set(required).issubset(lookup.index):
        raise ValueError("Paired summaries must contain w/o P, B and S")

    BLUE = "#256886"
    NEGATIVE = "#b66b4c"
    GREY = "#7c929e"
    GRID = "#e0e6e9"
    TEXT = "#293c45"
    settings = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0, "axes.labelsize": 8.1,
        "axes.titlesize": 9.1, "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.3,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": dpi,
    }

    with plt.rc_context(settings):
        fig, (ax_c, ax_d) = plt.subplots(
            1, 2, figsize=(width, height),
            gridspec_kw={"width_ratios": [1.00, 1.16]},
        )

        # (c) Single-group ablations relative to the same full-model baseline.
        groups = ("P", "B", "S")
        means = np.array([float(lookup.loc[f"w/o {g}", "delta_RMSE_mean"])
                          for g in groups])
        sds = np.array([float(lookup.loc[f"w/o {g}", "delta_RMSE_sd"])
                        for g in groups])
        if not np.isfinite(means).all() or not np.isfinite(sds).all():
            raise ValueError("Nonfinite removal effects or SD values")
        yy = np.arange(3)
        colors = [BLUE if mu >= 0 else NEGATIVE for mu in means]
        ax_c.barh(yy, means, height=0.48, color=colors, alpha=0.86,
                  edgecolor="none", zorder=3)
        ax_c.errorbar(means, yy, xerr=sds, fmt="none", ecolor=TEXT,
                      elinewidth=0.9, capsize=2.5, capthick=0.8, zorder=4)
        ax_c.axvline(0, color="#555f65", lw=0.9, zorder=2)
        # ax_c.set_yticks(yy, [f"Remove {g}" for g in groups])
        ax_c.set_yticks(yy, [f"w/o {g}" for g in groups])
        ax_c.set_ylim(2.65, -0.65)
        ax_c.set_xlabel(r"Change in RMSE vs. full ($\Delta$RMSE)")
        # ax_c.set_title(f"({panel_labels[0]})", loc="left",
        #                fontweight="bold", pad=12)
        
        
        ax_c.text(
            0.02, 0.98, "(c)",
            transform=ax_c.transAxes,
            va="top", ha="left",
            fontsize=8,
            fontweight="bold",
            zorder=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.93,
                "pad": 0.7
            }
        )

        
        
        ax_c.grid(axis="x", color=GRID, ls=":", lw=0.7, zorder=0)
        ax_c.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_c.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:+.3f}"))
        low = min(0.0, float(np.min(means - sds)))
        high = max(0.0, float(np.max(means + sds)))
        span = max(high - low, 0.015)
        ax_c.set_xlim(low - 0.18 * span, high + 0.25 * span)
        for i, v in enumerate(means):
            offset = 0.025 * span
            ax_c.text(v + (offset if v >= 0 else -offset), i,
                      f"{v:+.3f}", ha="left" if v >= 0 else "right",
                      va="center", fontsize=7.1, color=TEXT, zorder=5,
                      bbox={"facecolor": "white", "alpha": 0.85,
                            "edgecolor": "none", "pad": 0.4})
        first_removed = str(path_df.iloc[1]["removed_at_step"])
        # Highlight which ablation led to the first greedy elimination.
        for tick, g in zip(ax_c.get_yticklabels(), groups):
            '''if g == first_removed:
                tick.set_fontweight("bold")
                tick.set_color(BLUE)'''
        # ax_c.text(0.0, 1.018,
        #           f"First removed: {first_removed}  |  Positive = worse",
        #           transform=ax_c.transAxes, color="#556c76",
        #           fontsize=6.8, ha="left", va="bottom")

        # (d) A four-point line shows how the selected feature set changes.
        xs = np.arange(4)
        rmse = path_df["RMSE_mean"].to_numpy(dtype=float)
        rmse_sd = path_df["RMSE_sd"].to_numpy(dtype=float)
        if not np.isfinite(rmse).all() or not np.isfinite(rmse_sd).all():
            raise ValueError("Nonfinite selected-path RMSE or SD")
        ax_d.errorbar(xs, rmse, yerr=rmse_sd, fmt="o-", color=BLUE,
                      lw=1.7, markersize=6, mfc=BLUE, mec="white", mew=0.75,
                      ecolor=GREY, elinewidth=1.1, capsize=3.5,
                      capthick=0.9, zorder=4)
        ax_d.set_xticks(xs)
        xtick_labels = ["Full\nP+B+S"]
        for index in range(1, 4):
            removed = str(path_df.iloc[index]["removed_at_step"])
            active = str(path_df.iloc[index]["features"])
            retained = active.replace("ProtT5+", "")
            xtick_labels.append(f"Remove {removed}\n{retained}")
        ax_d.set_xticklabels(xtick_labels, fontsize=7.0)
        ax_d.set_xlim(-0.22, 3.22)
        ax_d.set_ylabel("Validation RMSE (lower is better)")
        # ax_d.set_title(f"({panel_labels[1]})", loc="left", #Backward-elimination path
        #                fontweight="bold", pad=12)
        
        ax_d.text(
            0.02, 0.98, "(d)",
            transform=ax_d.transAxes,
            va="top", ha="left",
            fontsize=8,
            fontweight="bold",
            zorder=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.93,
                "pad": 0.7
            }
        )

        
        
        ax_d.grid(axis="y", color=GRID, ls=":", lw=0.7, zorder=0)
        ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
        lo2 = float(np.min(rmse-rmse_sd))
        hi2 = float(np.max(rmse+rmse_sd))
        span2 = max(hi2-lo2, 0.012)
        ax_d.set_ylim(lo2 - .25*span2, hi2 + .30*span2)
        for xx, mu, err in zip(xs, rmse, rmse_sd):
            ax_d.text(xx, mu + err + 0.045 * span2, f"{mu:.3f}",
                      ha="center", va="bottom", fontsize=7.2,
                      fontweight="bold" if xx == 0 else "normal", color=TEXT)
        # ax_d.text(0.0, 1.018, "ProtT5 retained at every step",
        #           transform=ax_d.transAxes, color="#556c76",
        #           fontsize=6.8, ha="left", va="bottom")

        for ax in (ax_c, ax_d):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", direction="out", length=2.5, width=0.65)
            ax.tick_params(axis="y", length=0)

        # Avoid small-font captions inside the figure; describe the method in
        # the manuscript caption. Make enough room for two-line step labels.
        fig.subplots_adjust(left=.115, right=.985, bottom=.225,
                            top=.83, wspace=.42)

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
    parser.add_argument("--filename", default="figure_backward_elimination_ablation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stack-n-jobs", type=int, default=1,
                        help="Parallelism of StackingRegressor (1 avoids nested overload)")
    parser.add_argument("--width", type=float, default=7.2)
    parser.add_argument("--height", type=float, default=2.8)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--panel-labels", nargs=2, default=("c", "d"),
                        metavar=("LEFT", "RIGHT"),
                        help="Panel letters (e.g. --panel-labels a b for a standalone figure)")
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
    path_df, candidate_df = build_backward_path(folds_df)
    path_df.to_csv(args.output_dir / "biological_feature_backward_path.csv", index=False)
    candidate_df.to_csv(args.output_dir / "biological_feature_backward_candidates.csv", index=False)
    step_df = build_step_effects(folds_df, path_df)
    step_df.to_csv(args.output_dir / "biological_feature_backward_step_effects.csv", index=False)
    paths = plot_figure(path_df, paired, args.output_dir,
                        filename=args.filename, width=args.width, height=args.height,
                        dpi=args.dpi, panel_labels=args.panel_labels)
    print("\nGreedy backward elimination (ProtT5 fixed):")
    for row in path_df.itertuples():
        print(f"  Step {row.step}: {row.features:<22} "
              f"RMSE={row.RMSE_mean:.4f} +/- {row.RMSE_sd:.4f} "
              f"(removed: {row.removed_at_step})")
    print("\nRMSE (five-fold mean +/- SD):")
    for row in summary.itertuples():
        print(f"  {row.configuration:14s}: {row.RMSE_mean:.4f} +/- {row.RMSE_sd:.4f}")
    print("\nChange from full model on removing each component (paired fold RMSE):")
    for row in paired.itertuples():
        print(f"  {row.removed:12s}: {row.delta_RMSE_mean:+.4f} +/- {row.delta_RMSE_sd:.4f}")
    print("\n".join(f"Saved {format.upper()}: {path}" for format, path in paths.items()))


if __name__ == "__main__":
    '''
    python plot_biological_feature_contributions.py \
    --plot-only \
    --output-dir ./results/plots/biological_feature_analysis
    '''
    
    cli()
