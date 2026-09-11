from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Eisenberg hydrophobicity values
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -1.50, "E": -1.50, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": -1.52, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26
}

CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0}

PHYSIO_LABELS = [
    "Length",
    "Molecular Weight",
    "Net Charge",
    "Charge Density",
    "Isoelectric Point (pI)",
    "Instability Index",
    "Aromaticity",
    "Aliphatic Index",
    "Boman Index",
    "Hydrophobic Ratio",

    # Amino acid composition (fraction)
    "Ala (A)", "Cys (C)", "Asp (D)", "Glu (E)", "Phe (F)",
    "Gly (G)", "His (H)", "Ile (I)", "Lys (K)", "Leu (L)",
    "Met (M)", "Asn (N)", "Pro (P)", "Gln (Q)", "Arg (R)",
    "Ser (S)", "Thr (T)", "Val (V)", "Trp (W)", "Tyr (Y)",

    # Hydrophobic moment descriptors
    "Eisenberg Hydrophobic Moment",
    "Eisenberg Hydrophobicity (Global)"
]

def extract_amp_features(sequences):
    """
    Extracts physicochemical features, capturing descriptors before 
    they are overwritten by subsequent modlamp calls.
    """
    # 1. Global Descriptors (MW, Charge, pI, etc.)
    glob = GlobalDescriptor(sequences)
    glob.calculate_all()
    df = pd.DataFrame(glob.descriptor, columns=glob.featurenames)

    # 2. Peptidic Descriptors (Eisenberg Scale)
    pep = PeptideDescriptor(sequences, 'eisenberg')
    
    # Calculate and capture Moment
    pep.calculate_moment() 
    df['Hydrophobic_Moment'] = pep.descriptor.flatten() 
    
    # Calculate and capture Global Hydrophobicity
    pep.calculate_global() 
    df['Global_Hydrophobicity'] = pep.descriptor.flatten() 
    
    # 3. Clean and Scale
    df = df.fillna(0)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    
    return scaled_array, df.columns.tolist()

def extract_amp_features_full(sequences):
    """
    Extracts AMP physicochemical features compatible with current modlamp version.
    
    Features:
    1. Global descriptors: MW, Charge, ChargeDensity, pI, Aliphatic Index, Aromaticity, Instability, Boman, Hydrophobic Ratio, Length
    2. Amino acid composition: 20 AA frequencies
    3. Eisenberg scale descriptors: hydrophobic moment and global average
    """

    df_list = []

    # ─── 1. Global descriptors ────────────────────────────────
    glob = GlobalDescriptor(sequences)
    glob.calculate_all()  # MW, Charge, pI, etc.
    df_list.append(pd.DataFrame(glob.descriptor, columns=glob.featurenames))

    # ─── 2. Amino acid composition (20 residues) ─────────────
    pep_base = PeptideDescriptor(sequences)  # no scale argument
    pep_base.count_aa()  # counts AA frequencies
    df_list.append(pd.DataFrame(pep_base.descriptor, columns=pep_base.featurenames))

    # ─── 3. Eisenberg scale descriptors ──────────────────────
    pep_eis = PeptideDescriptor(sequences, "eisenberg")
    pep_eis.calculate_moment()  # hydrophobic moment
    df_list.append(pd.DataFrame(pep_eis.descriptor, columns=["Eisenberg_Moment"]))

    pep_eis.calculate_global()  # global average
    df_list.append(pd.DataFrame(pep_eis.descriptor, columns=["Eisenberg_Global"]))

    # ─── 4. Combine all features and normalize ───────────────
    df_final = pd.concat(df_list, axis=1).fillna(0)
    scaled_features = StandardScaler().fit_transform(df_final)

    return scaled_features, df_final.columns.tolist()


def compute_save_physio_features(seq_csv_path=None, save_path = None):
    '''
        Computes Physiochemical Features from ecoli_normalized  Sequence CSV file and Saves it.
    '''
    df = pd.read_csv(seq_csv_path) #
    sequences = df['sequence'].to_list()

    # features_scaled, feature_names = extract_amp_features(sequences) #Top 12 features only
    features_scaled, feature_names = extract_amp_features_full(sequences)
    print(f"Feature Names: {feature_names} And Lenght: {len(feature_names)} ")

    # Success Check: Convert back to DataFrame for verification
    physio_df = pd.DataFrame(features_scaled, columns=feature_names)
    physio_df.insert(0, 'sequence', sequences)
    print(f"Extracted {len(feature_names)} features successfully.")
    print(physio_df.head())

    
    physio_df.to_csv(save_path)
    
    print(f"Count of total columns (including sequence): {physio_df.shape[1]} Total: {physio_df.columns.tolist()}")

    print(f"PhysioChem Features saved successfully to: {save_path}.")

def load_physio_features_as_numpy_all(physio_chem_path):
    """
    Returns a dictionary mapping each sequence to a NumPy array of all its physico-chemical features.

    Args:
        physio_chem_path (str): Path to the CSV file containing all physico-chemical features.

    Returns:
        dict: {sequence_str: np.array([all feature values], dtype=np.float32)}
    """

    # Load CSV
    df = pd.read_csv(physio_chem_path)

    # Remove 'Unnamed: 0' if present and set 'sequence' as index
    df = df.set_index('sequence').drop(columns=['Unnamed: 0'], errors='ignore')

    features_dict = {}

    # Iterate over sequences
    for seq, row in df.iterrows():
        feature_row = row.values.astype(np.float32)

        # Check for NaN or infinite values
        if not np.isfinite(feature_row).all():
            raise ValueError(f"Sequence {seq} contains NaN or Inf values.")

        features_dict[seq] = feature_row

    print(f"Loaded physico-chemical features for {len(features_dict)} sequences.")
    return features_dict

def load_physio_features_as_numpy(physio_chem_path):
    '''
    Returns Physico dictionary of Sequence as Key, and Numpy array as its feature values.
    '''
    df = pd.read_csv(physio_chem_path)
    #drop 'Unnamed: 0' to get only the numeric features and set index to sequence column
    #index orientation| keys = sequences value: dict:{col1: value1, col2: val2, col3: val3}
    physio_map = df.set_index('sequence').drop(columns=['Unnamed: 0'], errors='ignore').to_dict('index')

    '''
    Eg:
    Key: YPELQQDLIARLL 
    Value: {'Length': -0.6082551447299205, 'MW': -0.582454124890362, 
    'Charge': -1.8165588249838385, 'ChargeDensity': -1.7817525128767344, 'pI': -2.9343701092948087, 
    'InstabilityInd': 1.0819424035517982, 'Aromaticity': -0.3370577589352167,
    'AliphaticInd': 1.05090662530887, 'BomanInd': -0.0170077023703997,
    'HydrophRatio': 0.262069142615557, 'Hydrophobic_Moment': -0.1620249751930577,
        'Global_Hydrophobicity': 0.2418275906882444
     }

    '''
    features_dict = {}
    #Add just filtered columns | top 12 features
    for seq, feature_row_dict in physio_map.items():
        feature_row = []
        net_charge = feature_row_dict['Charge']
        m_w = feature_row_dict['MW'] #Molecular weight 
        s_len = feature_row_dict['Length'] #Seq Len | new 
        pI = feature_row_dict['pI'] #Isoeletric point 
        g_h = feature_row_dict['Global_Hydrophobicity']
        b_i = feature_row_dict['BomanInd']
        c_d = feature_row_dict['ChargeDensity']
        arm = feature_row_dict['Aromaticity'] #new
        ins_i = feature_row_dict['InstabilityInd'] #new
        h_m = feature_row_dict['Hydrophobic_Moment'] 
        al_i = feature_row_dict['AliphaticInd'] #new
        hyd_r = feature_row_dict['HydrophRatio'] #new 
        

        feature_row.extend([
           net_charge,m_w, s_len, pI, g_h, b_i, c_d, arm, ins_i, h_m, al_i, hyd_r  
        ])
        feature_row = np.array(feature_row, dtype=np.float32)
        has_invalid = ~np.isfinite(feature_row).all()
        if has_invalid:
            raise ValueError("Input Contains Nan or Infinite values.")
        

        features_dict[seq] = feature_row
    
    print(f"Length of Physico Features: {len(features_dict)}")
    return features_dict

def integrate_physio_with_mic(
    physio_csv="/data/prem001/PGAT-ABPp/code/data/phyiochem.csv",
    mic_csv="/data/prem001/PGAT-ABPp/code/data/ecoli_normalized.csv",
    output_csv="/data/prem001/PGAT-ABPp/code/data/physio_with_mic.csv"
):
    """
    Merge physico-chemical features with normalized MIC values based on sequence.

    Returns
    -------
    pd.DataFrame
        Integrated dataframe with all physico-chemical features + normalized_value
    """

    # Load data
    physio_df = pd.read_csv(physio_csv)
    mic_df = pd.read_csv(mic_csv)

    # Sanity check
    required_cols_physio = {"sequence"}
    required_cols_mic = {"sequence", "normalized_value"}

    if not required_cols_physio.issubset(physio_df.columns):
        raise ValueError("physiochem.csv must contain 'sequence' column")

    if not required_cols_mic.issubset(mic_df.columns):
        raise ValueError("ecoli_normalized.csv must contain 'sequence' and 'normalized_value' columns")

    # Merge on sequence
    merged_df = physio_df.merge(
        mic_df[["sequence", "normalized_value"]],
        on="sequence",
        how="left"  # keep all physio rows
    )

    # Report missing MIC values
    missing = merged_df["normalized_value"].isna().sum()
    if missing > 0:
        print(f" Warning: {missing} sequences have no matching MIC values")

    # Save integrated file
    if output_csv is not None:
        merged_df.to_csv(output_csv, index=False)
        print(f" Integrated file saved to: {output_csv}")

    return merged_df

import pandas as pd

def compute_physio_mic_correlation(
    physio_csv,
    mic_csv,
    sequence_col="sequence",
    target_col="log_mic",
    save_path="data/physio_mic_correlation.csv"
):
    """
    Compute absolute Pearson correlation between physico-chemical features and MIC.

    Parameters
    ----------
    physio_csv : str
        CSV file containing sequence + physico-chemical features
    mic_csv : str
        CSV file containing sequence + MIC (log or normalized)
    sequence_col : str
        Column name for peptide sequence
    target_col : str
        Column name for MIC values

    Returns
    -------
    pd.DataFrame
        Sorted table with columns:
        [feature, pearson_corr, abs_corr]
    """

    # ─── Load data ──────────────────────────────────────────
    df_physio = pd.read_csv(physio_csv)
    df_mic = pd.read_csv(mic_csv)

    # ─── Merge on sequence ──────────────────────────────────
    df = pd.merge(
        df_physio,
        df_mic[[sequence_col, target_col]],
        on=sequence_col,
        how="inner"
    )

    # ─── Select numeric physico-chemical features ───────────
    exclude_cols = {sequence_col, target_col, "Unnamed: 0"}
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # ─── Drop missing values ────────────────────────────────
    df = df[feature_cols + [target_col]].dropna()

    # ─── Compute correlations ───────────────────────────────
    records = []

    for col in feature_cols:
        corr = df[col].corr(df[target_col], method="pearson")
        records.append({
            "feature": col,
            "pearson_corr": corr,
            "abs_corr": abs(corr)
        })

    # ─── Sort by absolute correlation ───────────────────────
    corr_df = (
        pd.DataFrame(records)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    corr_df.to_csv(save_path, index=False)
    print(f"Correlation results saved to: {save_path}")

    return corr_df


def save_dict_to_txt(corr_dict, out_path="/data/prem001/PGAT-ABPp/code/data/physio_mic_correlation.txt"):
    with open(out_path, "w") as f:
        for feature, score in corr_dict.items():
            f.write(f"{feature}\t{score:.6f}\n")
        

def _mean_hydro(seq):
    return np.mean([EISENBERG.get(a, 0.0) for a in seq])

def _net_charge(seq):
    return sum([CHARGE.get(a, 0) for a in seq])

def _hydrophobic_moment(seq):
    if len(seq) < 3:
        return 0.0
    pep = PeptideDescriptor([seq], "eisenberg")
    pep.calculate_moment()
    return float(pep.descriptor[0][0])


def compute_nc_terminal_bias_from_sequences(sequences):
    """
    Compute N/C terminal physicochemical asymmetry features.

    Parameters
    ----------
    sequences : list[str]

    Returns
    -------
    pd.DataFrame with columns:
    [sequence, N_charge, C_charge, Delta_charge,
               N_hydro, C_hydro, Delta_hydro,
               N_moment, C_moment, Delta_moment]
    """
    records = []

    for seq in sequences:
        L = len(seq)
        k = max(3, int(0.5 * L))   # 50% split, min 3 aa

        N = seq[:k]
        C = seq[-k:]

        N_charge = _net_charge(N)
        C_charge = _net_charge(C)

        N_h = _mean_hydro(N)
        C_h = _mean_hydro(C)

        N_m = _hydrophobic_moment(N)
        C_m = _hydrophobic_moment(C)

        records.append({
            "sequence": seq,
            "N_charge": N_charge,
            "C_charge": C_charge,
            "Delta_charge": N_charge - C_charge,

            "N_hydro": N_h,
            "C_hydro": C_h,
            "Delta_hydro": N_h - C_h,

            "N_moment": N_m,
            "C_moment": C_m,
            "Delta_moment": N_m - C_m
        })

    return pd.DataFrame(records)

def compute_nc_terminal_bias_from_csv(csv_path, out_csv="data/nc_terminal_bias.csv"):
    """
    Load sequences from CSV, compute N/C terminal bias, and save.

    Input CSV must contain a 'sequence' column.
    """
    df = pd.read_csv(csv_path)
    sequences = df["sequence"].astype(str).tolist()

    bias_df = compute_nc_terminal_bias_from_sequences(sequences)

    bias_df.to_csv(out_csv, index=False)
    print(f"Saved N/C terminal bias features to: {out_csv}")

    return bias_df

def comp_save_full_physio_with_nc_terminal_bias():
    #Save nc_terminal bias data 
    physico_chem_src ="data/phyiochem.csv"  
    nc_terminal_bias_save_path = "data/nc_terminal_bias.csv"  
    nc_df = compute_nc_terminal_bias_from_csv(
        csv_path=physico_chem_src,
        out_csv=nc_terminal_bias_save_path
    )

    #merge with original physio 
    full_phyisioc_with_nc = "data/phyiochem_with_nc_bias.csv"
    physio = pd.read_csv(physico_chem_src)
    physio = physio.merge(nc_df, on="sequence", how="left")

    physio.to_csv(full_phyisioc_with_nc, index=False)


if __name__ == "__main__":
    #Ecoli
    input_seq_csv_path='./data/ecoli_clustered_dataset_splits.csv'
    save_path = 'data/phyiochem_ecoli.csv'


    compute_save_physio_features(input_seq_csv_path, save_path)
    
    



    

    
