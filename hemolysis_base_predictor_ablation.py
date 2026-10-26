#!/usr/bin/env python3
"""Validation-only, leave-one-base-predictor-out ablation of PepMetaStack.

The fixed PepMetaStack comprises exactly five base regressors:
  SVR, KNN, ExtraTrees, HistGradientBoosting, and MLP.
Its meta-regressor is RidgeCV with 15 alphas on logspace(-3, 3).

Experiments: one full PepMetaStack + five leave-one-out variants.
Evaluation: one shared outer KFold split, giving held-out predictions
for every development peptide. RidgeCV stacking uses a separate inner
KFold solely inside each outer-training split. No independent-test file
is read, fitted, or evaluated.

Inputs (unchanged from the source script):
  - development CSV: SEQUENCE and positive μM target
  - ProtT5 embeddings NPZ: original_sequences, embeddings
  - development Pfeature NPZ: Sequence, Pfeature, feature_names

Outputs: ONE CSV with one pooled out-of-fold validation row per experiment.
Metric values, including error metrics, are in the log10(μM) target scale.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Fixed architecture: no unrelated ensemble baselines or extra base models.
BASE_PREDICTORS = ("svr", "knn", "extra_trees", "hist_gb", "mlp")
METRIC_COLUMNS = (
    "Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2", "MSE"
)
OUTPUT_COLUMNS = (
    "Experiment", "Ablated_Predictor", "Base_Predictors", "Evaluation",
    "N_Validation", "Outer_Folds", "Inner_Folds", "Random_State",
    *METRIC_COLUMNS, "Delta_RMSE_vs_Full", "Delta_R2_vs_Full",
)
OUTLIER_WARNING_THRESHOLD = 3.0  # Diagnostic only; predictions are NEVER clipped.


def normalize_sequence(value):
    return str(value).strip().upper()


def build_feature_lookup(sequences, vectors, description):
    if len(sequences) != len(vectors):
        raise ValueError(f"{description}: sequence and feature counts differ")
    lookup = {}
    for seq, vec in zip(sequences, vectors):
        key = normalize_sequence(seq)
        arr = np.asarray(vec, dtype=np.float32)
        if not key or arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"{description}: invalid descriptor for {key!r}")
        if key in lookup:
            if lookup[key].shape != arr.shape or not np.array_equal(lookup[key], arr):
                raise ValueError(f"{description}: conflicting features for {key!r}")
        else:
            lookup[key] = arr
    return lookup


def load_prott5(path):
    # Variable-length residue arrays in this existing NPZ require pickle.
    # Only load NPZ files produced by a trusted source.
    with np.load(path, allow_pickle=True) as npz:
        for key in ("original_sequences", "embeddings"):
            if key not in npz.files:
                raise ValueError(f"ProtT5 file missing {key}: {path}")
        lookup = build_feature_lookup(
            npz["original_sequences"], npz["embeddings"], "ProtT5"
        )
    result = {}
    for seq, arr in lookup.items():
        if arr.ndim == 1:
            pooled = arr
        elif arr.ndim == 2 and arr.shape[0] > 0:
            pooled = np.mean(arr, axis=0)
        else:
            raise ValueError(f"ProtT5: invalid embedding shape for {seq}: {arr.shape}")
        if pooled.ndim != 1 or not np.isfinite(pooled).all():
            raise ValueError(f"ProtT5: invalid mean-pooled vector for {seq}")
        result[seq] = pooled.astype(np.float32)
    return result


def load_pfeatures(path):
    with np.load(path, allow_pickle=False) as npz:
        for key in ("Sequence", "Pfeature", "feature_names"):
            if key not in npz.files:
                raise ValueError(f"Pfeature file missing {key}: {path}")
        values = np.asarray(npz["Pfeature"], dtype=np.float32)
        names = np.asarray(npz["feature_names"], dtype=str)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Pfeature matrix and feature_names are inconsistent")
        return build_feature_lookup(npz["Sequence"], values, "Pfeature")


def load_development(args):
    dataset_path = Path(args.development)
    for item in (dataset_path, Path(args.prott5), Path(args.pfeature_development)):
        if not item.is_file():
            raise FileNotFoundError(f"Required input file not found: {item}")

    frame = pd.read_csv(dataset_path)
    for col in (args.sequence_column, args.target_column):
        if col not in frame.columns:
            raise ValueError(f"Development CSV lacks {col!r}; columns={frame.columns.tolist()}")
    if frame.empty or frame[args.sequence_column].isna().any():
        raise ValueError("Development dataset is empty or has missing sequences")
    sequences = frame[args.sequence_column].map(normalize_sequence).tolist()
    if any(not x for x in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("Development sequences must be nonempty and unique to avoid leakage")

    targets = pd.to_numeric(frame[args.target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(targets).all() or np.any(targets <= 0):
        raise ValueError(f"The {args.target_column!r} regression target must be finite and >0")
    y = np.log10(targets)

    prott5 = load_prott5(args.prott5)
    pfeatures = load_pfeatures(args.pfeature_development)
    blocks = []
    for label, lookup in (("ProtT5", prott5), ("Pfeature", pfeatures)):
        missing = [seq for seq in sequences if seq not in lookup]
        if missing:
            raise KeyError(f"{label}: {len(missing)} missing peptide vectors; e.g. {missing[:5]}")
        # Enforces identical dimensions for all development peptides.
        block = np.vstack([lookup[seq].reshape(-1) for seq in sequences])
        if not np.isfinite(block).all():
            raise ValueError(f"{label}: nonfinite features found")
        blocks.append(block)
        print(f"  {label:8s}: {block.shape}", flush=True)

    X = np.hstack(blocks).astype(np.float32, copy=False)
    print(f"  Development: {X.shape}; target=log10({args.target_column})", flush=True)
    return X, y, sequences


def create_base_predictor(name, args):
    if name == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
        ])
    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=7, weights="distance", metric="cosine", n_jobs=args.n_jobs
            )),
        ])
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators, max_depth=16,
            random_state=args.seed, n_jobs=args.n_jobs,
        )
    if name == "hist_gb":
        return HistGradientBoostingRegressor(
            max_iter=args.n_estimators, l2_regularization=1.0,
            random_state=args.seed,
        )
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64), activation="relu", alpha=0.01,
                max_iter=args.mlp_max_iter, early_stopping=True,
                random_state=args.seed,
            )),
        ])
    raise ValueError(f"Invalid base predictor: {name}")


def create_pepmetastack(remaining_bases, args):
    # For every ablation, the RidgeCV meta-learning algorithm is IDENTICAL.
    return StackingRegressor(
        estimators=[(name, create_base_predictor(name, args)) for name in remaining_bases],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 15)),
        cv=KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed),
        n_jobs=1,  # Avoid nested Joblib process pools; base estimators use --n-jobs.
        passthrough=False,
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) < 2 or not np.isfinite(y_pred).all():
        raise ValueError("Validation predictions must be finite and properly aligned")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = float(pearsonr(y_true, y_pred).statistic)
        rho = float(spearmanr(y_true, y_pred).statistic)
        tau = float(kendalltau(y_true, y_pred).statistic)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "Pearson_R": r,
        "Spearman_R": rho,
        "Kendall_Tau": tau,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
    }


def run_validation(bases, X, y, sequences, folds, args, name):
    predictions = np.full(y.shape, np.nan, dtype=np.float64)
    for fold_no, (training_indices, validation_indices) in enumerate(folds, start=1):
        model = create_pepmetastack(bases, args)
        model.fit(X[training_indices], y[training_indices])
        pred = np.asarray(model.predict(X[validation_indices]), dtype=np.float64).ravel()
        if len(pred) != len(validation_indices) or not np.isfinite(pred).all():
            raise ValueError(f"{name}: invalid predictions in validation fold {fold_no}")
        predictions[validation_indices] = pred

        errors = np.abs(y[validation_indices] - pred)
        max_local = int(np.argmax(errors))
        if errors[max_local] >= OUTLIER_WARNING_THRESHOLD:
            idx = int(validation_indices[max_local])
            print(
                f"  WARNING fold {fold_no}: extreme absolute error "
                f"{errors[max_local]:.4f} at row={idx}, peptide={sequences[idx]}; "
                f"actual={y[idx]:.4f}, predicted={pred[max_local]:.4f}",
                flush=True,
            )
        print(f"  Validation fold {fold_no}/{args.outer_folds} complete", flush=True)

    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{name}: at least one peptide has no OOF validation prediction")
    return regression_metrics(y, predictions)


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
        temp, index=False, float_format="%.10g"
    )
    temp.replace(path)


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Validation-only PepMetaStack ablation (SVR, KNN, ExtraTrees, HistGB, MLP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--development", default="./data/hemolysis/cross_val_dataset.csv")
    p.add_argument("--prott5", default="./data/hemolysis/hemolysis_prott5.npz")
    p.add_argument(
        "--pfeature-development",
        default="./data/hemolysis/pfeatures/crossvalidation_pfeatures.npz",
    )
    p.add_argument("--sequence-column", default="SEQUENCE")
    p.add_argument("--target-column", default="μM")
    p.add_argument(
        "--ablate", default=",".join(BASE_PREDICTORS),
        help="Select leave-one-out ablations from the FIXED five predictors; 'none' runs full only",
    )
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--mlp-max-iter", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", default="./result/hemolysis_ablation/pepmetastack_validation_ablation.csv",
        help="The ONLY output file (one row per full/ablated configuration)",
    )
    args = p.parse_args()
    if args.ablate.strip().lower() == "none":
        args.ablations = []
    else:
        names = [s.strip().lower() for s in args.ablate.split(",")]
        if any(not s for s in names):
            p.error("--ablate contains an empty name or repeated comma")
        if len(set(names)) != len(names):
            p.error("--ablate contains duplicate entries")
        if not set(names).issubset(BASE_PREDICTORS):
            p.error(f"--ablate supports only: {','.join(BASE_PREDICTORS)}")
        # Stable order regardless of --ablate user ordering.
        args.ablations = [s for s in BASE_PREDICTORS if s in names]
    if args.outer_folds < 2 or args.inner_folds < 2:
        p.error("--outer-folds and --inner-folds must be >=2")
    if args.n_estimators < 1 or args.mlp_max_iter < 1 or args.n_jobs == 0:
        p.error("--n-estimators and --mlp-max-iter must be positive; --n-jobs cannot be zero")
    return args


def main():
    args = parse_arguments()
    print("Loading development/validation data only...", flush=True)
    X, y, sequences = load_development(args)
    if len(y) < args.outer_folds:
        raise ValueError("Not enough development rows for the specified outer folds")
    min_outer_train = len(y) - int(np.ceil(len(y) / args.outer_folds))
    if min_outer_train < args.inner_folds:
        raise ValueError("Insufficient outer-training samples for the specified inner folds")

    folds = list(KFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed
    ).split(X))
    experiments = [("PepMetaStack_Full", "", list(BASE_PREDICTORS))]
    experiments.extend(
        (f"PepMetaStack_Without_{removed}", removed,
         [name for name in BASE_PREDICTORS if name != removed])
        for removed in args.ablations
    )
    print(f"\nRunning {len(experiments)} fixed-architecture configurations:", flush=True)
    for label, _, remaining in experiments:
        print(f"  {label}: {', '.join(remaining)}", flush=True)

    rows = []
    full_metrics = None
    for idx, (label, removed, remaining) in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] {label}", flush=True)
        metrics = run_validation(remaining, X, y, sequences, folds, args, label)
        if full_metrics is None:
            full_metrics = metrics
        row = {
            "Experiment": label,
            "Ablated_Predictor": removed,
            "Base_Predictors": "+".join(remaining),
            "Evaluation": "CV_OOF_Validation",
            "N_Validation": len(y),
            "Outer_Folds": args.outer_folds,
            "Inner_Folds": args.inner_folds,
            "Random_State": args.seed,
            **metrics,
            # Positive values mean removing the predictor worsened performance.
            "Delta_RMSE_vs_Full": metrics["RMSE"] - full_metrics["RMSE"],
            "Delta_R2_vs_Full": full_metrics["R2"] - metrics["R2"],
        }
        rows.append(row)
        save_csv(rows, args.output)  # Checkpoint after every completed experiment.
        print(
            f"  CV_OOF R={metrics['Pearson_R']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"R2={metrics['R2']:.4f}", flush=True,
        )
    print(f"\nSaved {len(rows)} validation-only rows to {Path(args.output).resolve()}")
    print("No independent-test data were loaded or evaluated.")


if __name__ == "__main__":
    '''
    python hemolysis_base_predictor_ablation.py \
        --ablate svr,knn,extra_trees,hist_gb,mlp \
        --outer-folds 5 \
        --inner-folds 5 \
        --n-estimators 200 \
        --n-jobs 4

    '''
    
    main()
