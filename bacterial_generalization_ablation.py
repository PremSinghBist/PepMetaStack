#!/usr/bin/env python3
"""Five-species MIC regression: ProtT5 vs ESM-2, with biological feature fusion.

Input:  data/grampa.csv (bacterium, sequence, value); value is ALREADY
        log-transformed/normalized MIC. Duplicate species+sequence targets are
        averaged in log space, equivalent to geometric averaging in the original
        MIC space when the transformation is affine in log(MIC).

Output: bacterial_generalization_ablation/ (overridable)

This is a WITHIN-SPECIES experiment: five species are modeled separately.
Each model uses pooled PLM + 32 physicochemical + 20 BLOSUM62 + 32 sinusoidal
features. Validation metrics are computed on five REPEATED 80:20 splits,
with identical splits used for ProtT5 and ESM-2.

Dependencies:
  pip install numpy pandas scipy scikit-learn biopython modlamp torch \
              transformers sentencepiece tqdm

Usage:
  python bacterial_generalization_ablation.py --input data/grampa.csv \
      --output bacterial_generalization_ablation

  # Custom seeds, reduced resource use:
  python bacterial_generalization_ablation.py --seeds 42 12 128 \
      --device cuda --prott5-batch-size 2 --esm2-batch-size 2 --jobs 1

First execution downloads pretrained model weights from Hugging Face unless
cached locally. ProtT5 XL and ESM2-650M are very large; a CUDA GPU is strongly
recommended. Cached mean-pooled embeddings are reused on subsequent runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

LOG = logging.getLogger("bacterial_ablation")
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
DIM_PE = 32
MODEL_IDS = {
    "prott5": "Rostlab/prot_t5_xl_uniref50",
    "esm2": "facebook/esm2_t33_650M_UR50D",
}
EXPECTED_DIMS = {"prott5": 1024, "esm2": 1280}
METRICS = ["Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2"]


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/grampa.csv"))
    p.add_argument("--output", type=Path, default=Path("results/bacterial_generalization_ablation"))
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--min-sequences", type=int, default=30,
                   help="Minimum distinct peptide sequences per selected species.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 12, 128, 256, 512],
                   help="Random seeds for repeated 80:20 holdouts; >=2 for sample SD.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--prott5-batch-size", type=int, default=2)
    p.add_argument("--esm2-batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1000,
                   help="Discard longer peptides; avoids transformer truncation.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel jobs for sklearn; 1 minimizes RAM/GPU contention.")
    p.add_argument("--local-files-only", action="store_true",
                   help="Never download model weights; use HF cache only.")
    p.add_argument("--models", nargs="+", choices=["prott5", "esm2"],
                   default=["prott5", "esm2"])
    p.add_argument("--prepare-only", action="store_true",
                   help="Write cleaned/aggregated data and features; skip PLMs and fitting.")
    return p.parse_args()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def species_name(value) -> str | None:
    """Canonicalize leading binomial name; ignore strains/serotypes.

    Abbreviated common species are expanded where unambiguous from epithet.
    Unknown abbreviations are retained, not guessed.
    """
    if pd.isna(value):
        return None
    raw = re.sub(r"\s+", " ", str(value).strip())
    if not raw:
        return None
    normalized = re.sub(r"[._-]+", " ", raw).lower()
    if re.match(r"^(?:escherichia\s+coli|e\s+coli|ecoli)(?:\b|$)", normalized):
        return "Escherichia coli"
    parts = re.findall(r"[A-Za-z]+", raw)
    if len(parts) < 2:
        return None
    genus, epithet = parts[0].lower(), parts[1].lower()
    expansions = {
        ("s", "aureus"): "Staphylococcus",
        ("s", "epidermidis"): "Staphylococcus",
        ("s", "pneumoniae"): "Streptococcus",
        ("p", "aeruginosa"): "Pseudomonas",
        ("k", "pneumoniae"): "Klebsiella",
        ("a", "baumannii"): "Acinetobacter",
        ("e", "faecalis"): "Enterococcus",
        ("b", "subtilis"): "Bacillus",
    }
    genus = expansions.get((genus, epithet), genus.capitalize())
    # Avoid grouping generic labels such as "Gram positive" as a species.
    if genus.lower() in {"gram", "multiple", "unknown", "mixed", "not", "various"}:
        return None
    return f"{genus} {epithet}"


def clean_and_aggregate(path: Path, out: Path, top_n: int, min_sequences: int, max_length: int):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV missing: {path}. Supply --input PATH.")
    df = pd.read_csv(path, low_memory=False)
    needed = {"bacterium", "sequence", "value"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing CSV columns: {sorted(needed - set(df.columns))}")
    start = len(df)
    df = df[["bacterium", "sequence", "value"]].copy()
    df["species"] = df["bacterium"].map(species_name)
    df["sequence"] = (df["sequence"].astype("string")
                      .str.upper().str.replace(r"\s+", "", regex=True).str.strip())
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    valid = (df["species"].notna() & df["sequence"].notna() &
             df["sequence"].str.fullmatch(f"[{CANONICAL}]+", na=False) &
             df["sequence"].str.len().between(1, max_length).fillna(False) &
             np.isfinite(df["value"].to_numpy(dtype=np.float64)))
    df = df.loc[valid].copy()
    df = df[df["species"] != "Escherichia coli"].copy()
    if df.empty:
        raise ValueError("No valid non-E. coli canonical peptide records remain after filtering.")

    # IMPORTANT: Because 'value' is already log-transformed/normalized,
    # mean(value) is the correct target corresponding to a geometric mean of
    # raw MICs, PROVIDED 'value' is affine in log(MIC). Never exponentiate a
    # potentially standardized/normalized log value without knowing its scale.
    grouped = (df.groupby(["species", "sequence"], sort=True)
               .agg(value=("value", "mean"), n_measurements=("value", "size"),
                    target_sd=("value", "std"))
               .reset_index())
    grouped.to_csv(out / "all_non_ecoli_aggregated.csv", index=False)

    ranking = (grouped.groupby("species")
               .agg(unique_sequences=("sequence", "size"),
                    original_measurements=("n_measurements", "sum"))
               .reset_index()
               .sort_values(["unique_sequences", "species"], ascending=[False, True]))
    ranking.to_csv(out / "species_ranking.csv", index=False)
    eligible = ranking[ranking["unique_sequences"] >= min_sequences]
    if len(eligible) < top_n:
        raise ValueError(f"Only {len(eligible)} species have >= {min_sequences} unique sequences; "
                         f"need {top_n}. Check {out / 'species_ranking.csv'} "
                         "or lower --min-sequences / --top-n.")
    selected = eligible.head(top_n)["species"].tolist()
    subset = grouped[grouped["species"].isin(selected)].copy()
    subset.to_csv(out / "selected_five_species_aggregated.csv", index=False)
    LOG.info("Input rows: %d | valid non-E. coli records: %d | aggregated pairs: %d",
             start, len(df), len(grouped))
    LOG.info("Selected top %d species by UNIQUE sequences: %s", top_n, selected)
    for s in selected:
        c = ranking.loc[ranking["species"] == s].iloc[0]
        LOG.info("  %-35s sequences=%d measurements=%d", s,
                 c["unique_sequences"], c["original_measurements"])
    return subset, selected


def biological_features(sequences: list[str], out: Path):
    """Reproduce the supplied feature generators, WITHOUT global scaling leakage.

    Physicochemical: modlamp GlobalDescriptor.calculate_all() + 20 AA
    frequencies + Eisenberg hydrophobic moment/global = 32 features.
    BLOSUM62: average of canonical substitution matrix rows = 20.
    PE: sine/cosine encodings averaged across each peptide's valid positions = 32.
    """
    try:
        from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
    except ImportError as exc:
        raise RuntimeError("modlamp is required. Install: pip install modlamp") from exc

    LOG.info("Computing biological features for %d unique sequences", len(sequences))
    global_desc = GlobalDescriptor(sequences)
    global_desc.calculate_all()
    globals_df = pd.DataFrame(global_desc.descriptor, columns=global_desc.featurenames)
    comp = PeptideDescriptor(sequences)
    comp.count_aa()
    comp_df = pd.DataFrame(comp.descriptor, columns=comp.featurenames)
    eis = PeptideDescriptor(sequences, "eisenberg")
    eis.calculate_moment()
    moment_df = pd.DataFrame(eis.descriptor, columns=["Eisenberg_Moment"])
    eis.calculate_global()
    global_eis_df = pd.DataFrame(eis.descriptor, columns=["Eisenberg_Global"])
    physio_raw = pd.concat([globals_df, comp_df, moment_df, global_eis_df], axis=1)
    physio_raw = physio_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if physio_raw.shape[1] != 32:
        raise RuntimeError(f"Expected 32 physicochemical features, got {physio_raw.shape[1]}. "
                           "Check installed modlamp version.")

    mat = substitution_matrices.load("BLOSUM62")
    b = np.asarray([[mat[a, z] for z in BLOSUM_ORDER] for a in BLOSUM_ORDER],
                   dtype=np.float32)
    row_index = {aa: j for j, aa in enumerate(BLOSUM_ORDER)}
    blosum = np.stack([b[[row_index[aa] for aa in seq]].mean(axis=0)
                       for seq in sequences]).astype(np.float32)

    frequencies = np.exp(-np.arange(0, DIM_PE, 2, dtype=np.float64)
                         * np.log(10000.0) / DIM_PE)
    pe = np.empty((len(sequences), DIM_PE), dtype=np.float32)
    pe_by_length = {}
    for i, seq in enumerate(sequences):
        l = len(seq)
        if l not in pe_by_length:
            angle = np.arange(l, dtype=np.float64)[:, None] * frequencies[None, :]
            pos = np.empty((l, DIM_PE), dtype=np.float64)
            pos[:, 0::2] = np.sin(angle)
            pos[:, 1::2] = np.cos(angle)
            pe_by_length[l] = pos.mean(axis=0).astype(np.float32)
        pe[i] = pe_by_length[l]

    phys = physio_raw.to_numpy(dtype=np.float32)
    if not all(np.isfinite(x).all() for x in (phys, blosum, pe)):
        raise ValueError("Non-finite biological features after processing")
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for filename, arr, columns in [
        ("physicochemical_raw.csv", phys, list(physio_raw.columns)),
        ("blosum62.csv", blosum, [f"blosum62_{a}" for a in BLOSUM_ORDER]),
        ("sinusoidal.csv", pe, [f"pe_{i}" for i in range(DIM_PE)]),
    ]:
        saved = pd.DataFrame(arr, columns=columns)
        saved.insert(0, "sequence", sequences)
        saved.to_csv(feature_dir / filename, index=False)
    return np.concatenate((phys, blosum, pe), axis=1).astype(np.float32)


def get_device(which: str):
    import torch
    if which == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device("cuda" if (which == "cuda" or (which == "auto" and torch.cuda.is_available()))
                        else "cpu")


def plm_embeddings(model_name: str, sequences: list[str], out: Path, args):
    """Load OR compute mean-pooled residue-only language model features."""
    cache = out / "features" / f"{model_name}_mean_pooled.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as cached:
            saved_seq = cached["sequences"].tolist()
            saved_embeddings = cached["embeddings"].astype(np.float32)
        if saved_seq == sequences and saved_embeddings.shape == (len(sequences), EXPECTED_DIMS[model_name]):
            LOG.info("Loaded %s cache: %s", model_name, cache)
            return saved_embeddings
        LOG.warning("Existing %s embedding cache does not match sequences; regenerating", model_name)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("PLM extraction needs torch, transformers, sentencepiece and tqdm") from exc

    device = get_device(args.device)
    hf_id = MODEL_IDS[model_name]
    LOG.info("Loading %s on %s: %s", model_name, device, hf_id)
    if model_name == "prott5":
        tokenizer = T5Tokenizer.from_pretrained(
            hf_id, do_lower_case=False, local_files_only=args.local_files_only)
        model = T5EncoderModel.from_pretrained(hf_id, local_files_only=args.local_files_only)
        batch_size = args.prott5_batch_size
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=args.local_files_only)
        model = AutoModel.from_pretrained(hf_id, local_files_only=args.local_files_only)
        batch_size = args.esm2_batch_size
    if batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    model = model.eval().to(device)
    if device.type == "cuda":
        model = model.half()  # smaller GPU memory footprint for inference only

    outputs = np.empty((len(sequences), EXPECTED_DIMS[model_name]), dtype=np.float32)
    progress = tqdm(total=len(sequences), desc=f"{model_name} embeddings", unit="peptide")
    offset = 0
    while offset < len(sequences):
        this_batch = min(batch_size, len(sequences) - offset)
        curr = sequences[offset:offset + this_batch]
        preprocessed = [re.sub(r"[UZOB]", "X", s) for s in curr]
        if model_name == "prott5":
            preprocessed = [" ".join(s) for s in preprocessed]
        try:
            toks = tokenizer(preprocessed, add_special_tokens=True, padding=True,
                             truncation=False, return_tensors="pt")
            toks = {key: val.to(device) for key, val in toks.items()}
            with torch.inference_mode():
                hidden = model(**toks).last_hidden_state
            for j, seq in enumerate(curr):
                residue_length = len(seq)
                n_tokens = int(toks["attention_mask"][j].sum().item())
                if n_tokens < residue_length + (1 if model_name == "prott5" else 2):
                    raise RuntimeError(f"Unexpected token count for {model_name} sequence {seq[:25]}")
                # ProtT5: [residues] + eos. ESM2: cls + [residues] + eos.
                start = 0 if model_name == "prott5" else 1
                outputs[offset + j] = (hidden[j, start:start + residue_length]
                                       .float().mean(dim=0).cpu().numpy())
            del toks, hidden
            offset += this_batch
            progress.update(this_batch)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or this_batch == 1:
                raise
            batch_size = max(1, this_batch // 2)
            LOG.warning("CUDA OOM on %s; reducing batch size to %d", model_name, batch_size)
            gc.collect()
            torch.cuda.empty_cache()
    progress.close()
    if not np.isfinite(outputs).all():
        raise ValueError(f"Non-finite {model_name} embeddings")
    np.savez_compressed(cache, sequences=np.asarray(sequences), embeddings=outputs)
    LOG.info("Saved %s embeddings %s: %s", model_name, outputs.shape, cache)
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def build_model(seed: int, n_train: int, jobs: int):
    """Same five base estimators and RidgeCV stack as provided reference code."""
    inner_splits = min(5, n_train)
    if inner_splits < 2:
        raise ValueError("Insufficient training samples")
    # kNN must support the smallest internal CV training fold.
    min_internal_fit = n_train - int(np.ceil(n_train / inner_splits))
    neighbors = min(7, min_internal_fit)
    estimators = [
        ("svr", Pipeline([("scaler", StandardScaler()),
                          ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf"))])),
        ("extra_trees", ExtraTreesRegressor(n_estimators=150, max_depth=16,
                                             random_state=seed, n_jobs=jobs)),
        ("hist_gb", HistGradientBoostingRegressor(max_iter=150,
                                                   l2_regularization=1.0,
                                                   random_state=seed)),
        ("knn", Pipeline([("scaler", StandardScaler()),
                          ("model", KNeighborsRegressor(
                              n_neighbors=neighbors, weights="distance", metric="cosine",
                              n_jobs=jobs))])),
        ("mlp", Pipeline([("scaler", StandardScaler()),
                          ("model", MLPRegressor(hidden_layer_sizes=(128, 64),
                                                 activation="relu", alpha=0.01,
                                                 max_iter=300, early_stopping=True,
                                                 random_state=seed))])),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 15)),
        cv=KFold(n_splits=inner_splits, shuffle=True, random_state=seed),
        n_jobs=jobs,
    )


def score_regression(truth: np.ndarray, pred: np.ndarray) -> dict:
    if len(truth) < 2:
        raise ValueError("Validation set must contain at least two examples")
    def correlation(method):
        if np.std(truth) == 0 or np.std(pred) == 0:
            return np.nan
        return float(method(truth, pred).statistic)
    return {
        "Pearson_R": correlation(pearsonr),
        "Spearman_R": correlation(spearmanr),
        "Kendall_Tau": correlation(kendalltau),
        "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
        "MAE": float(mean_absolute_error(truth, pred)),
        "R2": float(r2_score(truth, pred)),
    }


def run_evaluations(data: pd.DataFrame, species: list[str], sequences: list[str],
                    bio: np.ndarray, args):
    out = args.output
    index = {seq: j for j, seq in enumerate(sequences)}
    scores = []
    all_predictions = []
    out_splits = out / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    result_dir = out / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        plm = plm_embeddings(model_name, sequences, out, args)
        feature_matrix = np.concatenate([plm, bio], axis=1)
        LOG.info("%s full feature dimension: %d", model_name, feature_matrix.shape[1])
        for bacterial_species in species:
            frame = (data[data["species"] == bacterial_species].copy()
                     .sort_values("sequence").reset_index(drop=True))
            chosen = np.asarray([index[seq] for seq in frame["sequence"]], dtype=int)
            X = feature_matrix[chosen]
            y = frame["value"].to_numpy(dtype=np.float64)
            for repeat, seed in enumerate(args.seeds, 1):
                # Splits rely only on species/seed, not model, so the paired
                # ProtT5 and ESM2 experiments use EXACTLY the same validation set.
                train_idx, val_idx = train_test_split(
                    np.arange(len(frame)), train_size=0.8, test_size=0.2,
                    random_state=seed, shuffle=True)
                if len(train_idx) < 10 or len(val_idx) < 2:
                    raise ValueError(f"Too few peptides in {bacterial_species} for 80:20 split")
                split_dir = out_splits / slugify(bacterial_species) / f"seed_{seed}"
                split_dir.mkdir(parents=True, exist_ok=True)
                if not (split_dir / "train.csv").exists():
                    frame.iloc[train_idx].to_csv(split_dir / "train.csv", index=False)
                    frame.iloc[val_idx].to_csv(split_dir / "validation.csv", index=False)
                LOG.info("Fitting %s | %s | split %d/%d seed=%d train=%d val=%d",
                         model_name, bacterial_species, repeat, len(args.seeds), seed,
                         len(train_idx), len(val_idx))
                model = build_model(seed, len(train_idx), args.jobs)
                model.fit(X[train_idx], y[train_idx])
                pred = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
                metrics = score_regression(y[val_idx], pred)
                scores.append({"species": bacterial_species, "embedding": model_name,
                               "repeat": repeat, "seed": seed,
                               "train_n": len(train_idx), "validation_n": len(val_idx),
                               "feature_dim": X.shape[1], **metrics})
                pred_df = frame.iloc[val_idx][["species", "sequence", "value"]].copy()
                pred_df = pred_df.rename(columns={"value": "y_true"})
                pred_df["y_pred"] = pred
                pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
                pred_df["absolute_error"] = np.abs(pred_df["residual"])
                pred_df["embedding"] = model_name
                pred_df["seed"] = seed
                all_predictions.append(pred_df)
                # Checkpoint continually in case a later GPU/model run fails.
                pd.DataFrame(scores).to_csv(result_dir / "validation_per_split.csv", index=False)
                pd.concat(all_predictions, ignore_index=True).to_csv(
                    result_dir / "validation_predictions.csv", index=False)
                del model
        del plm, feature_matrix
        gc.collect()
    return pd.DataFrame(scores)


def write_summaries(detail: pd.DataFrame, output: Path, seeds: list[int]):
    if detail.empty:
        return
    directory = output / "results"
    directory.mkdir(parents=True, exist_ok=True)
    summary = (detail.groupby(["species", "embedding"], as_index=False)[METRICS]
               .agg(["mean", "std"]))
    summary.columns = ["species", "embedding"] + [f"{metric}_{stat}"
                      for metric, stat in summary.columns.to_list()[2:]]
    summary["n_repeats"] = detail.groupby(["species", "embedding"]).size().to_numpy()
    summary.to_csv(directory / "validation_mean_sd.csv", index=False)
    pretty = summary[["species", "embedding", "n_repeats"]].copy()
    for metric in METRICS:
        pretty[metric] = summary.apply(
            lambda r: f"{r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f}", axis=1)
    pretty.to_csv(directory / "validation_mean_sd_formatted.csv", index=False)
    # Direct ablation comparison on matched species and random splits.
    # Positive ESM2_minus_ProtT5 favors ESM2 for correlations/R2;
    # negative favors ESM2 for error metrics (RMSE/MAE).
    if set(detail["embedding"].unique()) >= {"prott5", "esm2"}:
        paired = detail.pivot_table(index=["species", "seed"], columns="embedding",
                                    values=METRICS, aggfunc="first")
        records = []
        for species in detail["species"].unique():
            if species not in paired.index.get_level_values("species"):
                continue
            frame = paired.loc[species]
            for metric in METRICS:
                delta = (frame[(metric, "esm2")] - frame[(metric, "prott5")]).dropna()
                if len(delta):
                    records.append({"species": species, "metric": metric,
                                    "ESM2_minus_ProtT5_mean": delta.mean(),
                                    "ESM2_minus_ProtT5_sd": delta.std(ddof=1),
                                    "paired_splits": len(delta)})
        pd.DataFrame(records).to_csv(directory / "paired_model_comparison.csv", index=False)
    if len(seeds) < 2:
        LOG.warning("Only one seed: SD is undefined; use >=2 --seeds for mean±SD.")
    LOG.info("\n%s", pretty.to_string(index=False))
    LOG.info("Saved %s, %s", directory / "validation_mean_sd.csv",
             directory / "validation_mean_sd_formatted.csv")


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 < args.top_n or args.min_sequences < 10 or args.max_length < 1:
        raise ValueError("Invalid --top-n, --min-sequences, or --max-length")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) == 0:
        raise ValueError("Supply >=1 distinct random seed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "features").mkdir(exist_ok=True)
    try:
        records, selected = clean_and_aggregate(args.input, args.output, args.top_n,
                                               args.min_sequences, args.max_length)
        seqs = sorted(records["sequence"].unique().tolist())
        LOG.info("Unique peptide sequences across selected species: %d", len(seqs))
        settings = {"input": str(args.input), "selected_species": selected,
                    "seeds": args.seeds, "train_fraction": 0.8,
                    "validation_fraction": 0.2, "embedding_models": args.models,
                    "feature_fusion": "mean-pooled PLM + raw physico32 + BLOSUM20 + PE32",
                    "target": "arithmetic mean of previously log-normalized MIC, per species+sequence",
                    "selection": "top by distinct sequences after cleaning and deduplication",
                    "max_sequence_length": args.max_length,
                    "interpretation": "within-species repeated 80:20 validation; NOT held-out-species generalization"}
        (args.output / "experiment_config.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8")
        bio = biological_features(seqs, args.output)
        if args.prepare_only:
            LOG.info("--prepare-only selected: preprocessing complete, skipping model evaluation")
            return
        per_split = run_evaluations(records, selected, seqs, bio, args)
        write_summaries(per_split, args.output, args.seeds)
    except Exception:
        LOG.exception("Experiment failed. Any previously generated output/checkpoints are retained.")
        raise


if __name__ == "__main__":
    '''
        python bacterial_generalization_ablation.py \
    --input data/grampa.csv \
    --output results/bacterial_generalization_ablation
    '''
    
    main()
