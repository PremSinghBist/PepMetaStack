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

if __name__ == "__main__":
    #Ecoli
    save_path = './data/ecoli_sinusoidal_encoding.csv'
    pe_global, sequences = extract_position_encoding('./data/ecoli_clustered_dataset_splits.csv')
    save_position_encoding(pe_global, sequences, save_path)
    
    # To load later:
    # encoding_dict = load_sinusoidal_encoding(sinusoidal_encoding_path)
    