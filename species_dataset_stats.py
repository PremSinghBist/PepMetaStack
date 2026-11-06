#!/usr/bin/env python3
"""Summarize aggregated species-level MIC data in one CSV (no random seeds).

Expected input columns:
    species,sequence,value,n_measurements,target_sd

Each row represents one unique (species, peptide sequence) pair.
The target 'value' is summarized as stored (no back-transformation).
"""
import argparse
import math
from pathlib import Path

import pandas as pd


def summarize(input_path: Path, output_dir: Path, validation_fraction: float = 0.20) -> Path:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")

    df = pd.read_csv(input_path)
    required = {"species", "sequence", "value", "n_measurements", "target_sd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    df["species"] = df["species"].astype("string").str.strip()
    df["sequence"] = df["sequence"].astype("string").str.strip()
    for col in ("value", "n_measurements", "target_sd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["species", "sequence", "value", "n_measurements"]].isna().any().any():
        raise ValueError("Missing or invalid species, sequence, value, or n_measurements")
    if (df["sequence"].str.len() == 0).any():
        raise ValueError("Empty peptide sequence encountered")
    if (df["n_measurements"] < 1).any() or (df["n_measurements"] % 1 != 0).any():
        raise ValueError("n_measurements must contain positive integers")
    if df.duplicated(["species", "sequence"]).any():
        raise ValueError("Duplicate (species, sequence) rows: aggregate them first")

    rows = []
    for species, group in df.groupby("species", sort=True):
        n = len(group)
        if n < 2:
            raise ValueError(f"Too few peptides to split for {species}: {n}")
        # Matches sklearn's train_test_split(..., test_size=0.20) sample counts.
        val_n = math.ceil(n * validation_fraction)
        train_n = n - val_n
        if not train_n:
            raise ValueError(f"No training peptides remain for {species}")
        lengths = group["sequence"].str.len()
        y = group["value"]
        rows.append({
            "species": species,
            "total_peptides": n,
            "original_measurements": int(group["n_measurements"].sum()),
            "peptides_with_replicates": int((group["n_measurements"] > 1).sum()),
            "train_n": train_n,
            "validation_n": val_n,
            "sequence_length_mean": lengths.mean(),
            "sequence_length_median": lengths.median(),
            "sequence_length_min": lengths.min(),
            "sequence_length_max": lengths.max(),
            "target_mean": y.mean(),
            "target_sd": y.std(ddof=1),
            "target_median": y.median(),
            "target_min": y.min(),
            "target_max": y.max(),
        })

    summary = pd.DataFrame(rows).sort_values("total_peptides", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "species_summary.csv"
    summary.to_csv(result, index=False, float_format="%.6f")
    print(summary[["species", "total_peptides", "original_measurements", "train_n", "validation_n"]].to_string(index=False))
    print(f"\nSaved: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("selected_five_species_aggregated.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/species_stats"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    summarize(args.input, args.output, args.validation_fraction)


if __name__ == "__main__":
    '''
    python species_dataset_stats.py \
    --input ./results/bacterial_generalization_ablation/selected_five_species_aggregated.csv \
    --output results/species_stats
    
    '''
    main()
