import numpy as np
import pandas as pd

import numpy as np

import experimental_config 

cfg = experimental_config.ExperimentConfig()


def positional_encoding_global(pe_padded, pe_mask):
    """
    Compute a single vector per sequence by averaging positional encodings
    over valid (non-padded) positions.

    Args:
        pe_padded : np.ndarray of shape (B, L, d)
                    Padded positional encodings.
        pe_mask   : np.ndarray of shape (B, L)
                    1 for real tokens, 0 for padding.

    Returns:
        np.ndarray of shape (B, d)
        One global positional embedding per sequence.
    """
    # Expand mask to broadcast over the embedding dimension
    mask = pe_mask[..., None]  # shape (B, L, 1)

    # Sum only over real tokens
    summed = np.sum(pe_padded, axis=1)  # (B, L, d) -> (B, d)

    # Count number of real tokens per sequence | 
    counts = np.sum(mask, axis=1) + 1e-8  # shape (B, 1), epsilon to avoid div by zero

    # Compute the average over real tokens
    return summed / counts  # shape (B, d)



def pad_positional_encodings(pe_list, max_len=cfg.seq_max_len):
    '''Pad a list of positional encoding matrices to a common length.
    Args:
        pe_list: list of np.ndarray, each of shape (L_i, d)
        max_len: int, maximum sequence length for padding
        Returns:
            padded: (N, max_len, D)
            mask:   (N, max_len)
    '''
    dim = pe_list[0].shape[1] 
    total_seqs = len(pe_list)

    padded = np.zeros((total_seqs, max_len, dim), dtype=np.float32) 
    mask   = np.zeros((total_seqs, max_len), dtype=np.float32) 

    for index, pe in enumerate(pe_list):
        # Slice the positional encoding if it exceeds max_len
        seq_len = min(pe.shape[0], max_len) 
        
        # Pull only up to max_len tokens from the positional encoding matrix
        padded[index, :seq_len, :] = pe[:seq_len, :] 
        mask[index, :seq_len] = 1.0   

    return padded, mask


def sinusoidal_position_encoding(length, d_model):
    """
    Create sinusoidal positional encoding matrix for a given sequence of lenght L.
    
    Args:
        length (int): sequence length
        d_model (int): embedding dimension
    
    Returns:
        (length, d_model) numpy array
    """
    pos = np.arange(length)[:, np.newaxis]             # (L, 1)
    i = np.arange(d_model)[np.newaxis, :]              # (1, D)

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates

    pe = np.zeros((length, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # even dims
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # odd dims

    return pe


def extract_position_encoding(csv_file):
    df = pd.read_csv(csv_file)
    sequences = df['sequence'].to_list()

    print(f"Loaded {len(sequences)} sequences")

    # Step 1: token-level positional encodings
    pe_list = [] #list of (L, D) arrays
    for seq in sequences:
        pe = sinusoidal_position_encoding(len(seq), cfg.sinusoidal_feature_dim)
        pe_list.append(pe)

    # Step 2: pad into batch
    pe_padded, pe_mask = pad_positional_encodings(pe_list, max_len=cfg.seq_max_len)
    # Step 3: pool to one vector per peptide
    pe_global = positional_encoding_global(pe_padded, pe_mask)

    print("Token PE shape:", pe_padded.shape)   # (B, 128, D)
    print("Global PE shape:", pe_global.shape)  # (B, D)

    return pe_global, sequences



def save_position_encoding(pe_global, sequences, save_path):
    """ Save global positional encodings to a CSV file """
    B, d = pe_global.shape
    col_names = [f"pe_{i}" for i in range(d)]
    df = pd.DataFrame(pe_global, columns=col_names)
    df.insert(0, 'sequence', sequences)
    df.to_csv(save_path, index=False)
    print(f"Saved positional encodings to {save_path}")

def load_sinusoidal_encoding(csv_path):
    '''
    Load sinusoidal positional encodings from a CSV file.
    
    Returns a dictionary mapping sequences to their positional encodings.
    Key: sequence (str)
    Value: positional encoding (np.ndarray of shape (32,))

    '''
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].to_list()      # original sequences
    pe_global = df.drop(columns='sequence').to_numpy(dtype=np.float32)  # shape (B, d)
    print(f"Loaded sinusoidal positional encodings from {csv_path}")
    print("Global PE shape:", pe_global.shape)  # (B, D)
    assert pe_global.shape[0] == len(sequences) 

    encoding_dict = {seq: pe_global[i] for i, seq in enumerate(sequences)}
    return encoding_dict

def extract_onehot_encoding(sequences:list):
    '''
    Extract one-hot positional encodings for a list of sequences.
    
    Args:
        sequences: list of sequences (str)
        returns:
            onehot_global: np.ndarray of shape (B, 20)
    '''
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}   #Amino acid : Index   

    onehot_list = []
    for seq in sequences:
        onehot = np.zeros((len(seq), 20))
        for i, aa in enumerate(seq):
            if aa in aa_to_index:
                onehot[i, aa_to_index[aa]] = 1
        onehot_list.append(onehot)

    # Pad to same length
    max_len = max(len(seq) for seq in onehot_list)
    padded_onehot = np.zeros((len(onehot_list), max_len, 20))
    for i, oh in enumerate(onehot_list):
        padded_onehot[i, :oh.shape[0], :] = oh

    # Global average pooling
    onehot_global = np.mean(padded_onehot, axis=1)  # (B, 20)

    return onehot_global

# --------------------------------------------------
# Hemolysis-specific positional encoding
# --------------------------------------------------

def read_hemolysis_sequences_for_position_encoding(csv_path):
    """
    Read hemolysis peptide sequences from CSV.

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


def extract_hemolysis_position_encoding(sequences):
    """
    Extract global sinusoidal positional encodings for
    hemolysis peptide sequences.

    Each sequence is represented by a single global positional
    encoding vector obtained by averaging valid positional
    encodings across its residues.

    Returns:
        pe_global : np.ndarray of shape (N, d)
        sequences : list of sequences
    """

    sequences = [
        str(seq).strip().upper()
        for seq in sequences
    ]

    # --------------------------------------------------
    # Step 1: Generate residue-level positional encodings
    # --------------------------------------------------
    pe_list = []

    for seq in sequences:
        pe = sinusoidal_position_encoding(
            len(seq),
            cfg.sinusoidal_feature_dim
        )
        pe_list.append(pe)

    # --------------------------------------------------
    # Step 2: Pad to common maximum sequence length
    # --------------------------------------------------
    pe_padded, pe_mask = pad_positional_encodings(
        pe_list,
        max_len=cfg.seq_max_len
    )

    # --------------------------------------------------
    # Step 3: Global average pooling
    # --------------------------------------------------
    pe_global = positional_encoding_global(
        pe_padded,
        pe_mask
    )

    print(
        f"Token PE shape   : {pe_padded.shape}"
    )
    print(
        f"Global PE shape  : {pe_global.shape}"
    )

    return pe_global, sequences


def save_hemolysis_position_encoding(
    cross_val_csv,
    independent_csv,
    save_path="data/hemolysis/hemolysis_sinusoidal_encoding.csv"
):
    """
    Compute and save global sinusoidal positional encodings
    for both hemolysis cross-validation and independent-test
    datasets.

    The output contains:
        SEQUENCE
        pe_0
        pe_1
        ...
        pe_(d-1)
    """

    # --------------------------------------------------
    # Read sequences
    # --------------------------------------------------
    cross_val_sequences = (
        read_hemolysis_sequences_for_position_encoding(
            cross_val_csv
        )
    )

    independent_sequences = (
        read_hemolysis_sequences_for_position_encoding(
            independent_csv
        )
    )

    print("\n========================================")
    print("Hemolysis positional encoding")
    print("========================================")

    print(
        f"Cross-validation sequences : "
        f"{len(cross_val_sequences)}"
    )

    print(
        f"Independent-test sequences : "
        f"{len(independent_sequences)}"
    )

    # --------------------------------------------------
    # Check overlap
    # --------------------------------------------------
    cross_set = set(cross_val_sequences)
    independent_set = set(independent_sequences)

    overlap = cross_set.intersection(
        independent_set
    )

    if overlap:
        print(
            f"WARNING: {len(overlap)} sequences occur "
            f"in both datasets."
        )
    else:
        print("No sequence overlap detected.")

    # --------------------------------------------------
    # Combine while preserving order
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Extract positional encodings
    # --------------------------------------------------
    print("\nExtracting sinusoidal positional encodings...")

    pe_global, processed_sequences = (
        extract_hemolysis_position_encoding(
            all_sequences
        )
    )

    # --------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------
    d = pe_global.shape[1]

    col_names = [
        f"pe_{i}"
        for i in range(d)
    ]

    features_df = pd.DataFrame(
        pe_global,
        columns=col_names
    )

    features_df.insert(
        0,
        "SEQUENCE",
        processed_sequences
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    from pathlib import Path

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        save_path,
        index=False
    )

    print("\nPositional encodings saved to:")
    print(save_path)

    print(
        f"Final feature matrix: "
        f"{features_df.shape[0]} sequences × "
        f"{features_df.shape[1] - 1} features"
    )

    print("========================================")

    return features_df

if __name__ == "__main__0":
    #Ecoli
    save_path = './data/ecoli_sinusoidal_encoding.csv'
    pe_global, sequences = extract_position_encoding('./data/ecoli_clustered_dataset_splits.csv')
    save_position_encoding(pe_global, sequences, save_path)
    
    # To load later:
    # encoding_dict = load_sinusoidal_encoding(sinusoidal_encoding_path)
    

if __name__ == "__main__":
    # ==================================================
    # NEW: Hemolysis positional encoding
    # ==================================================

    CROSS_VAL_HEMOLYSIS = (
        "./data/hemolysis/cross_val_dataset.csv"
    )

    INDEPENDENT_HEMOLYSIS = (
        "./data/hemolysis/independent_dataset.csv"
    )

    HEMOLYSIS_PE_PATH = (
        "./data/hemolysis/hemolysis_sinusoidal_encoding.csv"
    )

    save_hemolysis_position_encoding(
        cross_val_csv=CROSS_VAL_HEMOLYSIS,
        independent_csv=INDEPENDENT_HEMOLYSIS,
        save_path=HEMOLYSIS_PE_PATH
    )