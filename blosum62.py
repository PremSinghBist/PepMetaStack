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
# 4. Test
# --------------------------------------------------
if __name__ == "__main__":
    #Ecoli
    input_csv = "data/ecoli_clustered_dataset_splits.csv"
    output_csv = "data/ecoli_blosum62_features.csv"

    
 

    compute_blosum62_features(input_seqs_csv=input_csv, output_csv=output_csv)
    
    blosum_dict = load_blosum62_features(csv_path=output_csv)
    print(list(blosum_dict.items())[:2])
