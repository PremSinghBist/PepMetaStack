from Bio.Align import substitution_matrices
import numpy as np
import pandas as pd
from typing import Dict

# --------------------------------------------------
# 1. Load BLOSUM62 once
# --------------------------------------------------
BLOSUM62 = substitution_matrices.load("BLOSUM62")

# --------------------------------------------------
# 2. Canonical amino acids and filtered matrix
# --------------------------------------------------
CANONICAL_AAS = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX = {aa: i for i, aa in enumerate(CANONICAL_AAS)}

def get_canonical_blosum62(matrix: "SubstitutionMatrix") -> np.ndarray:
    """
    Filter BLOSUM62 to canonical 20 amino acids.
    
    Returns:
        20x20 np.ndarray
    """
    filtered = np.array([[matrix[a1][a2] for a2 in CANONICAL_AAS] for a1 in CANONICAL_AAS], dtype=np.float32)
    return filtered

CANONICAL_BLOSUM = get_canonical_blosum62(BLOSUM62)

# --------------------------------------------------
# 3. Sequence-level BLOSUM62 composition
# --------------------------------------------------
def blosum62_composition(seq: str, blosum_mat: np.ndarray = CANONICAL_BLOSUM) -> np.ndarray:
    """
    Compute 20-dim BLOSUM62 composition vector for a sequence.
    
    Steps:
    1. Convert sequence to uppercase
    2. Ignore non-canonical amino acids
    3. Extract BLOSUM62 row for each residue
    4. Average over sequence
    """
    seq = seq.upper()
    
    # Get row indices of canonical residues
    idx = [AA_TO_IDX[aa] for aa in seq if aa in AA_TO_IDX]
    
    if not idx:
        return np.zeros(len(CANONICAL_AAS), dtype=np.float32)
    
    # Select rows and compute mean
    rows = blosum_mat[idx, :]
    return np.mean(rows, axis=0)

def compute_blosum62_features(input_seqs_csv: str, output_csv: str):
    """
    Read sequences from input CSV, compute BLOSUM62 features,
    and save to output CSV.
    
    Input CSV must contain a column named 'sequence'.
    """
    df = pd.read_csv(input_seqs_csv)

    sequences = df["sequence"].astype(str).tolist()

    # Compute features
    blosum_features = np.vstack([
        blosum62_composition(seq) for seq in sequences
    ])  # shape: (N, 20)

    # Create column names
    blosum_cols = [f"blosum62_{aa}" for aa in CANONICAL_AAS]

    # Build final DataFrame
    features_df = pd.DataFrame(
        blosum_features,
        columns=blosum_cols
    )

    features_df.insert(0, "sequence", sequences)

    features_df.to_csv(output_csv, index=False)

def load_blosum62_features(csv_path: str) -> pd.DataFrame:
    """
    Load BLOSUM62 features from CSV.
    returns: dict {sequence: feature_vector}
    """
    df = pd.read_csv(csv_path)
    blosum_cols = [f"blosum62_{aa}" for aa in CANONICAL_AAS]
    seqs = df["sequence"].tolist()
    features = df[blosum_cols].values.astype(np.float32)

    assert len(seqs) == features.shape[0], "Mismatch in number of sequences and features."

    blosum_dict = {
        seq: features[i]
        for i, seq in enumerate(seqs)
    }

    print(f"Loaded BLOSUM62 features from {csv_path}, shape: {features.shape}")
    print(features.dtype)
    return blosum_dict

# --------------------------------------------------
# Hemolysis-specific BLOSUM62 feature extraction
# --------------------------------------------------

def read_hemolysis_sequences_for_blosum(csv_path: str):
    """
    Read hemolysis sequences from CSV.

    The input CSV must contain a column named 'SEQUENCE'.
    """
    df = pd.read_csv(csv_path)

    if "SEQUENCE" not in df.columns:
        raise ValueError(
            f"'SEQUENCE' column not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = (
        df["SEQUENCE"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(f"Loaded {len(sequences)} sequences from {csv_path}")

    return sequences


def extract_hemolysis_blosum62_features(sequences):
    """
    Compute sequence-level BLOSUM62 composition features
    for hemolysis peptide sequences.

    Returns:
        DataFrame with:
            SEQUENCE
            blosum62_A ... blosum62_V
    """

    sequences = [
        str(seq).strip().upper()
        for seq in sequences
    ]

    # Compute 20-dimensional BLOSUM62 representation
    blosum_features = np.vstack([
        blosum62_composition(seq)
        for seq in sequences
    ])

    # Feature names
    blosum_cols = [
        f"blosum62_{aa}"
        for aa in CANONICAL_AAS
    ]

    # Build DataFrame
    features_df = pd.DataFrame(
        blosum_features,
        columns=blosum_cols
    )

    # Keep sequence as first column
    features_df.insert(
        0,
        "SEQUENCE",
        sequences
    )

    print(
        f"Extracted {len(blosum_cols)} BLOSUM62 features."
    )
    print(
        f"Feature matrix shape: {features_df.shape}"
    )

    return features_df


def save_hemolysis_blosum62_features(
    cross_val_csv: str,
    independent_csv: str,
    save_path: str = "data/hemolysis/hemolysis_blosum62_features.csv"
):
    """
    Compute BLOSUM62 features for both the hemolysis
    cross-validation and independent-test datasets.

    The two datasets are combined into a single feature CSV.
    """

    # ------------------------------------------
    # Read sequences
    # ------------------------------------------
    cross_val_sequences = read_hemolysis_sequences_for_blosum(
        cross_val_csv
    )

    independent_sequences = read_hemolysis_sequences_for_blosum(
        independent_csv
    )

    print("\n========================================")
    print("Hemolysis BLOSUM62 feature extraction")
    print("========================================")
    print(
        f"Cross-validation sequences : "
        f"{len(cross_val_sequences)}"
    )
    print(
        f"Independent-test sequences : "
        f"{len(independent_sequences)}"
    )

    # ------------------------------------------
    # Check sequence overlap
    # ------------------------------------------
    cross_set = set(cross_val_sequences)
    independent_set = set(independent_sequences)

    overlap = cross_set.intersection(independent_set)

    if overlap:
        print(
            f"WARNING: {len(overlap)} sequences occur "
            f"in both datasets."
        )
    else:
        print("No sequence overlap detected.")

    # ------------------------------------------
    # Combine while preserving order
    # ------------------------------------------
    all_sequences = list(
        dict.fromkeys(
            cross_val_sequences +
            independent_sequences
        )
    )

    print(
        f"Total unique sequences       : "
        f"{len(all_sequences)}"
    )

    # ------------------------------------------
    # Extract BLOSUM62 features
    # ------------------------------------------
    print("\nExtracting BLOSUM62 features...")

    features_df = extract_hemolysis_blosum62_features(
        all_sequences
    )

    # ------------------------------------------
    # Save
    # ------------------------------------------
    from pathlib import Path

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        save_path,
        index=False
    )

    print("\nBLOSUM62 features saved to:")
    print(save_path)

    print(
        f"Final feature matrix: "
        f"{features_df.shape[0]} sequences × "
        f"{features_df.shape[1] - 1} features"
    )

    print("========================================")

    return features_df

# --------------------------------------------------
# 4. Test
# --------------------------------------------------
if __name__ == "__main__0":
    #Ecoli
    input_csv = "data/ecoli_clustered_dataset_splits.csv"
    output_csv = "data/ecoli_blosum62_features.csv"

    
 

    compute_blosum62_features(input_seqs_csv=input_csv, output_csv=output_csv)
    
    blosum_dict = load_blosum62_features(csv_path=output_csv)
    print(list(blosum_dict.items())[:2])
    

if __name__ == "__main__":
    # ==================================================
    # NEW: Hemolysis BLOSUM62 extraction
    # ==================================================

    CROSS_VAL_HEMOLYSIS = (
        "data/hemolysis/cross_val_dataset.csv"
    )

    INDEPENDENT_HEMOLYSIS = (
        "data/hemolysis/independent_dataset.csv"
    )

    HEMOLYSIS_BLOSUM_PATH = (
        "data/hemolysis/hemolysis_blosum62_features.csv"
    )

    save_hemolysis_blosum62_features(
        cross_val_csv=CROSS_VAL_HEMOLYSIS,
        independent_csv=INDEPENDENT_HEMOLYSIS,
        save_path=HEMOLYSIS_BLOSUM_PATH
    )
