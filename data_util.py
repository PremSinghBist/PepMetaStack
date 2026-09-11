from pathlib import Path 
import random
from typing import List, Tuple
import pandas as pd
import numpy as np

from prott5_feature_extractor import load_embeddings
from blosum62 import load_blosum62_features
import physiochem_feature_extractor as PFE
import position_encoding_extractor as PEE

from experimental_config import ExperimentConfig
CFG = ExperimentConfig()


def normalize(features, mean=None, std=None, eps=1e-8):
    """
    Z-score normalization for BLOSUM features.

    features: np.ndarray of shape (N, [?])
    mean, std: computed on training set and reused for val/test

    Preserves realative scales between different feature dimensions.
    Returns:
        features_norm: normalized features
        mean: mean used for normalization (per feature dimension). feautres dim is 20, there will be 20 means.
        std: std used for normalization (per feature dimension)
    """
    if mean is None:
        mean = features.mean(axis=0)   
    if std is None:
        std = features.std(axis=0)    

    features_norm = (features - mean) / (std + eps)
    return features_norm, mean, std

def load_datasets_old(datasets_index=[0, 1, 2, 3, 4], dataset_path=''):
    """
    datasets_index: Which Dataset to load 
    0: First dataset 1: Second dataset and so on.
    Returns: List [(train_set, val_set, test_set), (...)]
    """
    import pickle
    with open(dataset_path, "rb") as f:
        datasets = pickle.load(f)


    max_idx = len(datasets) - 1
    for i in datasets_index:
        if i < 0 or i > max_idx:
            raise ValueError(
                f"Invalid dataset index {i}. Valid range: [0, {max_idx}]"
            )

    print(f"Datasets loaded from {dataset_path}.")
    return [datasets[i] for i in datasets_index]

def load_features(
    embedding_file=None,
    blosum_csv="",
    physio_csv="",
    sinusoidal_csv="",
    dataset_csv="",
    fold_index=1,
):
    """
    Load features and create train/validation/test splits
    for the selected CV fold.

    split == "test"       -> fixed test set
    cv_fold == fold_index -> validation
    other cv_fold values  -> training
    """

    if fold_index not in [1, 2, 3, 4, 5]:
        raise ValueError("fold_index must be between 1 and 5.")

    # Load dataset metadata
    df = pd.read_csv(dataset_csv)

    required = {"sequence", "value", "split", "cv_fold"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    metadata = (
        df.set_index("sequence")[["value", "split", "cv_fold"]]
        .to_dict("index")
    )

    # Load pre-computed features
    seqs, _, embs = load_embeddings(embedding_file)

    blosum_dict = load_blosum62_features(
        csv_path=blosum_csv
    )

    physio_dict = PFE.load_physio_features_as_numpy_all(
        physio_csv
    )

    sinusoidal_dict = PEE.load_sinusoidal_encoding(
        sinusoidal_csv
    )

    # Create splits
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for seq, emb in zip(seqs, embs):

        if seq not in metadata:
            raise ValueError(
                f"{seq} missing in dataset CSV."
            )

        row = metadata[seq]

        # Determine split
        if row["split"] == "test":
            split = "test"
            cv_fold = None

        elif int(row["cv_fold"]) == fold_index:
            split = "val"
            cv_fold = int(row["cv_fold"])

        else:
            split = "train"
            cv_fold = int(row["cv_fold"])

        # ProtT5 embedding
        emb = np.asarray(emb, dtype=np.float32)

        if emb.ndim == 1:
            emb = emb.reshape(-1, CFG.PROTT5_EMB_DIM)

        # Additional features
        physio = np.asarray(
            physio_dict[seq],
            dtype=np.float32
        )

        blosum = np.asarray(
            blosum_dict[seq],
            dtype=np.float32
        )

        sino = np.asarray(
            sinusoidal_dict[seq],
            dtype=np.float32
        )

        splits[split].append(
            {
                "emb": emb,
                "physio": physio,
                "blosum": blosum,
                "sino": sino,
                "seq": seq,
                "mic": row["value"],
                "split": split,
                "cv_fold": cv_fold,
            }
        )

    print(
        f"\nCV Fold {fold_index}: "
        f"Train={len(splits['train'])}, "
        f"Val={len(splits['val'])}, "
        f"Test={len(splits['test'])}"
    )

    return splits

def print_features_stat(splits):
    # -------------------------------
    # Print Basic Statistics
    # -------------------------------
    print("=" * 60)
    print("               DATASET & FEATURE STATISTICS             ")
    print("=" * 60)

    # 1. Split Counts
    n_train, n_val, n_test = (
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    n_total = n_train + n_val + n_test
    print(f"Total Sequences Loaded: {n_total}")
    print(
        f"  ├─ Train : {n_train:5d} ({n_train / n_total * 100:.1f}%)"
        if n_total
        else 0
    )
    print(
        f"  ├─ Val   : {n_val:5d} ({n_val / n_total * 100:.1f}%)"
        if n_total
        else 0
    )
    print(
        f"  └─ Test  : {n_test:5d} ({n_test / n_total * 100:.1f}%)"
        if n_total
        else 0
    )
    print("-" * 60)

    # 2. Feature Dimensions (from first available sample)
    sample = (
        splits["train"][0]
        if splits["train"]
        else (splits["val"][0] if splits["val"] else splits["test"][0])
    )
    print("Feature Shapes per Sequence:")
    print(f"  ├─ Protein Embedding : {sample['emb'].shape}")
    print(f"  ├─ Physio-Chemical   : {sample['physio'].shape}")
    print(f"  ├─ BLOSUM62          : {sample['blosum'].shape}")
    print(f"  └─ Sinusoidal PE     : {sample['sino'].shape}")
    print("-" * 60)

    # 3. Label (MIC) Distribution Statistics
    print("Target (MIC) Distribution:")
    for s_name in ["train", "val", "test"]:
        if len(splits[s_name]) > 0:
            mics = [x["mic"] for x in splits[s_name]]
            print(
                f"  ├─ {s_name.capitalize():5s} -> Mean: {np.mean(mics):.4f} | Std: {np.std(mics):.4f} | Min: {np.min(mics):.4f} | Max: {np.max(mics):.4f}"
            )
    print("=" * 60)
    
# ======================================================================
# Load GMProt fusion embeddings
# ======================================================================

def load_fusion_embeddings(embedding_path):
    """
    Load 128-D GMProt fusion embeddings from a .npy file.

    Parameters
    ----------
    embedding_path : str or Path
        Path to the saved .npy embedding file.

    Returns
    -------
    dict
        Dictionary containing:
            sequence -> 128-D NumPy embedding
    """

    embedding_path = Path(embedding_path)

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file does not exist: {embedding_path}"
        )

    # Load dictionary
    embedding_dict = np.load(
        embedding_path,
        allow_pickle=True
    ).item()

    if not isinstance(embedding_dict, dict):
        raise ValueError(
            "The embedding file does not contain a dictionary."
        )

    # Basic information
    print("\n" + "=" * 60)
    print("GMProt Fusion Embeddings")
    print("=" * 60)

    print(f"Embedding file     : {embedding_path}")
    print(f"Number of sequences: {len(embedding_dict)}")

    if embedding_dict:
        first_sequence = next(iter(embedding_dict))
        first_embedding = np.asarray(
            embedding_dict[first_sequence]
        )

        print(f"Embedding dimension: {first_embedding.shape}")
        print(f"Data type          : {first_embedding.dtype}")
        print(f"First sequence     : {first_sequence}")

        # Verify all embeddings
        dimensions = {
            np.asarray(embedding).shape
            for embedding in embedding_dict.values()
        }

        if len(dimensions) == 1:
            print(f"Embedding shapes    : {dimensions}")
        else:
            print(
                f"Warning: Multiple embedding shapes found: "
                f"{dimensions}"
            )

    print("=" * 60)

    return embedding_dict

def merge_and_resplit_by_counts(data_splits, seed=None):
    """
    Merges train, val, and test splits, shuffles them completely, 
    and resplits into new sets preserving the exact original lengths of each split.
    
  
    returns: 
        new_train, new_val, new_test: Resplit datasets with original lengths.
    """
    if seed is not None:
        random.seed(seed)
        
    print(f"\nMerging and resplitting datasets with seed={seed}...")
        
    # 1. Capture original target lengths
    n_train = len(data_splits['train'])
    n_val = len(data_splits['val'])
    
    # 2. Merge all samples
    combined = list(data_splits['train']) + list(data_splits['val']) + list(data_splits['test'])
    
    # 3. Shuffle completely
    random.shuffle(combined)
    
    # 4. Slice using exact original sample counts
    return {
        'train': combined[:n_train],
        'val': combined[n_train : n_train + n_val],
        'test': combined[n_train + n_val :]
    } 

def create_dataset_split(input_csv, save_csv, random_state=42):
    """
    Shuffle the dataset and create a fixed independent test set
    with 5-fold cross-validation on the remaining development set.

    Dataset:
        Total       = 4,793
        Development = 4,074
        Test        =   719

    The 4,074 development sequences are divided into 5 folds.
    For each CV fold:
        - 4 folds are used for training
        - 1 fold is used for validation
        - 719 test sequences remain fixed and are never used for training

    The output CSV contains:
        - split   : train / validation / test
        - cv_fold : 1 / 2 / 3 / 4 / 5 for development samples,
                    <NA> for test samples

    Fold 1 is assigned as the default validation fold in the
    'split' column.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.

    save_csv : str
        Path where the split CSV will be saved.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled dataset with 'split' and 'cv_fold' columns.
    """

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------

    df = pd.read_csv(input_csv)

    total_size = len(df)
    development_size = 4074
    test_size = 719
    n_folds = 5

    print(f"Total sequences loaded: {total_size}")

    # ---------------------------------------------------------------
    # 2. Validate dataset size
    # ---------------------------------------------------------------

    expected_size = development_size + test_size

    if total_size != expected_size:
        raise ValueError(
            f"Expected {expected_size:,} sequences, "
            f"but found {total_size:,}."
        )

    # ---------------------------------------------------------------
    # 3. Shuffle the entire dataset
    # ---------------------------------------------------------------

    df = (
        df.sample(
            frac=1,
            random_state=random_state
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # 4. Create fixed test set
    #
    # First 4,074 sequences -> development set
    # Last 719 sequences   -> fixed test set
    # ---------------------------------------------------------------

    df["split"] = "test"

    df["cv_fold"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    development_indices = np.arange(
        development_size
    )

    test_indices = np.arange(
        development_size,
        total_size
    )

    # ---------------------------------------------------------------
    # 5. Create 5 CV folds from development set
    # ---------------------------------------------------------------

    rng = np.random.default_rng(random_state)

    rng.shuffle(
        development_indices
    )

    folds = np.array_split(
        development_indices,
        n_folds
    )

    for fold_number, fold_indices in enumerate(
        folds,
        start=1
    ):
        df.loc[
            fold_indices,
            "cv_fold"
        ] = fold_number

    # ---------------------------------------------------------------
    # 6. Set default Train / Validation split
    #
    # Fold 1     -> Validation
    # Folds 2-5  -> Train
    # Test       -> Test
    # ---------------------------------------------------------------

    df.loc[
        df["cv_fold"] == 1,
        "split"
    ] = "validation"

    df.loc[
        df["cv_fold"].isin([2, 3, 4, 5]),
        "split"
    ] = "train"

    # ---------------------------------------------------------------
    # 7. Verify split sizes
    # ---------------------------------------------------------------

    split_counts = df["split"].value_counts()

    expected_train = (
        development_size - len(folds[0])
    )

    expected_validation = len(folds[0])

    if split_counts.get("train", 0) != expected_train:
        raise ValueError(
            "Unexpected training set size."
        )

    if split_counts.get("validation", 0) != expected_validation:
        raise ValueError(
            "Unexpected validation set size."
        )

    if split_counts.get("test", 0) != test_size:
        raise ValueError(
            "Unexpected test set size."
        )

    # ---------------------------------------------------------------
    # 8. Save dataset
    # ---------------------------------------------------------------

    df.to_csv(
        save_csv,
        index=False
    )

    # ---------------------------------------------------------------
    # 9. Display summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Dataset split created successfully")
    print("=" * 60)

    print("\nDefault split:")
    print(
        df["split"]
        .value_counts()
        .reindex(
            ["train", "validation", "test"]
        )
    )

    print("\n5-fold CV distribution:")
    print(
        df["cv_fold"]
        .value_counts()
        .sort_index()
    )

    print("\nCV fold sizes:")

    for fold_number in range(
        1,
        n_folds + 1
    ):
        count = (
            df["cv_fold"] == fold_number
        ).sum()

        print(
            f"Fold {fold_number}: "
            f"{count} sequences"
        )

    print(f"\nRandom seed : {random_state}")
    print(f"Saved to    : {save_csv}")
    print("=" * 60)

    return df

def summarize_cv_metrics(fold_metrics, save_csv):
    """
    Compute mean and standard deviation across CV folds
    and save the summary as a CSV file.

    Parameters
    ----------
    fold_metrics : list of dict
        Metrics from each CV fold.

    save_csv : str
        Output CSV path.

    Returns
    -------
    dict
        Mean and standard deviation for each metric.
    """

    metric_names = [
        "RMSE",
        "R2",
        "Pearson",
        "Kendall",
    ]

    results = []
    summary = {}

    for metric in metric_names:

        values = np.array(
            [fold[metric] for fold in fold_metrics],
            dtype=np.float64
        )

        mean = np.mean(values)
        std = np.std(values, ddof=1)

        summary[metric] = {
            "mean": float(mean),
            "std": float(std),
        }

        results.append({
            "Metric": metric,
            "Mean": mean,
            "Std": std,
        })

    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv, index=False)

    # Print results
    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation Performance")
    print("=" * 60)

    for result in results:
        print(
            f"{result['Metric']:<10}: "
            f"{result['Mean']:.4f} ± {result['Std']:.4f}"
        )

    print("=" * 60)
    print(f"Results saved to: {save_csv}")

    return summary

def get_cv_split(input_csv_path, fold):
    """
    Load the dataset split CSV and return train, validation,
    and fixed test data for a specific CV fold.

    Parameters
    ----------
    input_csv_path : str or Path
        Path to the CSV file created by create_dataset_split().

    fold : int
        Cross-validation fold number (1-5).

    Returns
    -------
    train_df : pd.DataFrame
        Development sequences excluding the selected validation fold.

    val_df : pd.DataFrame
        Sequences belonging to the selected validation fold.

    test_df : pd.DataFrame
        Fixed independent test set.
    """

    # ---------------------------------------------------------------
    # 1. Validate fold
    # ---------------------------------------------------------------

    if fold not in [1, 2, 3, 4, 5]:
        raise ValueError(
            "fold must be between 1 and 5."
        )

    # ---------------------------------------------------------------
    # 2. Load split CSV
    # ---------------------------------------------------------------

    input_csv_path = Path(input_csv_path)

    if not input_csv_path.exists():
        raise FileNotFoundError(
            f"Dataset split file does not exist: "
            f"{input_csv_path}"
        )

    df = pd.read_csv(
        input_csv_path
    )

    # ---------------------------------------------------------------
    # 3. Check required columns
    # ---------------------------------------------------------------

    required_columns = {
        "split",
        "cv_fold"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------------
    # 4. Extract train / validation / test
    # ---------------------------------------------------------------

    train_df = df[
        (df["cv_fold"].notna()) &
        (df["cv_fold"] != fold)
    ].copy()

    val_df = df[
        df["cv_fold"] == fold
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    # ---------------------------------------------------------------
    # 5. Display summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"CV Fold {fold}")
    print("=" * 60)

    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("=" * 60)

    return train_df, val_df, test_df



if __name__ == "__main__":
    create_dataset_split(
        input_csv="./data/ecoli_dataset.csv",
        save_csv="./data/ecoli_five_fold_dataset.csv",
        random_state=42
    )
    
    train_df, val_df, test_df = get_cv_split(
        input_csv_path="./data/ecoli_five_fold_dataset.csv",
        fold=2
    )
    '''load_datasets_old(
        datasets_index=[0, 1, 2, 3, 4], dataset_path="./data/ecoli_prott5_based_datasets.pkl"
    )
    import sys; sys.exit(0)
    
    # Example usage
    splits = load_features(
        normalize_features=True,
        embedding_file=CFG.ecoli_prott5_embedding_file,
        blosum_csv=CFG.ecoli_blosum_csv,
        physio_csv=CFG.ecoli_physio_csv,
        sinusoidal_csv=CFG.ecoli_sinusoidal_csv,
        dataset_csv=CFG.ecoli_dataset_csv
    )
    print_features_stat(splits)'''